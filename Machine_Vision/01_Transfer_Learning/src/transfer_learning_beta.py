# Transfer learning using Keras and Tensorflow.
# Written by Rahul Remanan and MOAD (https://www.moad.computer) machine vision team.
# For more information contact: info@moad.computer
# License: MIT open source license (https://github.com/rahulremanan/python_tutorial/blob/master/LICENSE).
# Example usage: python3 transfer_learning.py --training_directory /home/info/train --validation_directory /home/info/val --batches 20 --epochs 10 --model_file /home/info/transfer_learn_epoch100.model --output_directory /home/info/model --train_model True --load_model True --fine_tune True --test_augmentation False --plot True
import time
import os
import sys
import glob
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import keras
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, RMSprop, Adagrad

IM_WIDTH, IM_HEIGHT = 299, 299 #Fixed size for InceptionV3
DEFAULT_EPOCHS = 100
DEFAULT_BATCHES = 10
FC_SIZE = 1024
NB_LAYERS_TO_FREEZE = 169

sgd = SGD(lr=1e-7, decay=0.5, momentum=1, nesterov=True)
rms = RMSprop(lr=1e-7, rho=0.9, epsilon=1e-08, decay=0.0)
ada = Adagrad(lr=1e-7, epsilon=1e-08, decay=0.0)
optimizer = sgd

def generate_timestamp():
    timestring = time.strftime("%Y_%m_%d-%H_%M_%S")
    print ("Time stamp generated: "+timestring)
    return timestring

timestr = generate_timestamp()

def is_valid_file(parser, arg):
    if not os.path.isfile(arg):
        parser.error("The file %s does not exist..." % arg)
    else:
        return arg
    
def is_valid_dir(parser, arg):
    if not os.path.isdir(arg):
        parser.error("The folder %s does not exist..." % arg)
    else:
        return arg
 
def get_bool_fn(bool_fn):       #Boolean filter
    try:
        load_value = bool(bool_fn)
        return load_value
    except:
        print ("Please check if: --train_model, --load_model, --fine_tune, --test_aug, --plot arguments are True or False statements...")
        print ("Example usage: --train_model True ")

def get_nb_files(directory):
  if not os.path.exists(directory):
    return 0
  cnt = 0
  for r, dirs, files in os.walk(directory):
    for dr in dirs:
      cnt += len(glob.glob(os.path.join(r, dr + "/*")))
  return cnt

def setup_to_transfer_learn(model, base_model, optimizer):
  for layer in base_model.layers:
    layer.trainable = False
  model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
  return model

def add_new_last_layer(base_model, nb_classes):
  """Add last layer to the convnet"""
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(FC_SIZE, activation='relu')(x) #new FC layer, random init
  predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer
  model = Model(inputs=base_model.input, outputs=predictions)
  return model

def setup_to_finetune(model, optimizer):
  """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers"""
  for layer in model.layers[:NB_LAYERS_TO_FREEZE]:
     layer.trainable = False
  for layer in model.layers[NB_LAYERS_TO_FREEZE:]:
     layer.trainable = True
  model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
  return model

def train(args):
  """Use transfer learning and fine-tuning to train a network on a new dataset"""
  nb_train_samples = get_nb_files(args.train_dir[0])
  print (nb_train_samples)
  nb_classes = len(glob.glob(args.train_dir[0] + "/*"))
  print (nb_classes)
  nb_val_samples = get_nb_files(args.val_dir[0])
  nb_epoch = int(args.epoch[0])
  batch_size = int(args.batch[0])

  train_datagen =  ImageDataGenerator(preprocessing_function=preprocess_input,
      rotation_range=30,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True)
  
  test_aug = get_bool_fn(args.test_aug[0])  
  
  if test_aug==True:
      test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
          rotation_range=30,
          width_shift_range=0.2,
          height_shift_range=0.2,
          shear_range=0.2,
          zoom_range=0.2,
          horizontal_flip=True)
  else:
      test_datagen = ImageDataGenerator(rescale=1. / 255)

  train_generator = train_datagen.flow_from_directory(args.train_dir[0],
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size,
    class_mode='categorical')

  validation_generator = test_datagen.flow_from_directory(
    args.val_dir[0],
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size,
    class_mode='categorical')
  
  base_model = InceptionV3(weights='imagenet', include_top=False) #include_top=False excludes final FC layer
  model = add_new_last_layer(base_model, nb_classes)
  print ("Inception model loaded...")
  print (model.summary)
  
  try:
      load_model = get_bool_fn(args.load_model[0])
      train_model = get_bool_fn(args.train_model[0])
      fine_tune_model = get_bool_fn(args.fine_tune[0])
      if load_model == True:
            model = keras.models.load_model(args.model_file[0])
            print ("Loaded saved model weights...")
      else:
                print ("Tabula rasa....")
      if train_model == True:
            # transfer learning
            setup_to_transfer_learn(model, base_model, optimizer)
            print (model.summary)
            print (nb_train_samples)
            
            model_tl = model.fit_generator(
                  train_generator,
                  epochs=nb_epoch,
                  steps_per_epoch=nb_train_samples // batch_size,
                  validation_data=validation_generator,
                  validation_steps=nb_val_samples // batch_size,
                  class_weight='auto')

          # fine-tuning
            setup_to_finetune(model, optimizer)

            model_ft = model.fit_generator(
                  train_generator,
                  steps_per_epoch=nb_train_samples // batch_size,
                  epochs=nb_epoch,
                  validation_data=validation_generator,
                  validation_steps=nb_val_samples // batch_size,
                  class_weight='auto')
            
            if fine_tune_model == True:
                model_train = model_ft
            else:
                model_train = model_tl
                
            output_loc = args.output_dir[0]
            output_file = (output_loc+"//"+"trained_"+timestr+"bone_age.model")
            model.save(output_file)
      else:
          print ("Using pre-trained model for prediction...")
  except:
        print ("No pre-trained model or model weights loaded...")
  
  gen_plot = get_bool_fn(args.plot[0])
  if gen_plot==True:
      plot_training(model_train)
  else:
    model_train

def plot_training(history):
  output_loc = args.output_dir[0]
  output_file_acc = os.path.join(output_loc+"//training_plot_acc_"+timestr+".png")
  output_file_loss = os.path.join(output_loc+"//training_plot_loss_"+timestr+".png")
  
  fig = plt.figure()
  plt.plot(history.history['acc'])
  plt.plot(history.history['val_acc'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  fig.savefig(output_file_acc, dpi=fig.dpi)
  print ("Successfully created model training accuracy plot...")

  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  fig.savefig(output_file_loss, dpi=fig.dpi)
  print ("Successfully created model training loss function plot...")

def get_user_options():
    a = argparse.ArgumentParser()
    a.add_argument("--training_directory", 
                 help = "Specify folder contraining the files for training...", 
                 dest = "train_dir", 
                 required = True, 
                 type=lambda x: is_valid_dir(a, x), 
                 nargs=1)
    a.add_argument("--validation_directory", 
                   help = "Specify folder containing the files for validation...", 
                   dest = "val_dir", 
                   required = True, 
                   type=lambda x: is_valid_dir(a, x), 
                   nargs=1)
    a.add_argument("--epochs", 
                   help = "Specify epochs for training...", 
                   dest = "epoch", 
                   default=DEFAULT_EPOCHS, 
                   required=False, 
                   type = int, 
                   nargs=1)
    a.add_argument("--batches", help = "Specify batches for training...", 
                   dest = "batch", 
                   default=DEFAULT_BATCHES, 
                   required=False, 
                   type = int, 
                   nargs=1)
    a.add_argument("--model_file", 
                   help = "Specify pre-trained model file for training...", 
                   dest = "model_file", 
                   default="/home/info/model/inceptionv3-ft.model", 
                   required=False, 
                   nargs=1)
    a.add_argument("--output_directory", 
                   help = "Specify output folder...", 
                   dest = "output_dir", 
                   required = True, 
                   type=lambda x: is_valid_dir(a, x),
                   nargs=1)
    a.add_argument("--train_model", 
                   help = "Specify if the model should be trained...", 
                   dest = "train_model", 
                   required=False, 
                   default=True, 
                   nargs=1, 
                   type = bool)
    a.add_argument("--load_model", 
                   help = "Specify if a pre-trained model should be loaded...", 
                   dest = "load_model", 
                   required=False, 
                   default=False, 
                   nargs=1, 
                   type = bool)
    a.add_argument("--fine_tune", 
                   help = "Specify model should be fine tuned...", 
                   dest = "fine_tune", 
                   required=False, 
                   default=True, 
                   nargs=1, 
                   type = bool)
    a.add_argument("--test_augmentation", 
                   help = "Specify if image augmentation should be done on test dataset...", 
                   dest = "test_aug", 
                   required=False, 
                   default=False, 
                   nargs=1, 
                   type = bool)
    a.add_argument("--plot", 
                   help = "Specify if a plot should be generated...", 
                   dest = "plot", 
                   required=False, 
                   default=True, 
                   nargs=1, 
                   type = bool)
    args = a.parse_args()
    return args

if __name__=="__main__":
    args = get_user_options()

    if (not os.path.exists(args.train_dir[0])) or (not os.path.exists(args.val_dir[0])) or (not os.path.exists(args.model_file[0])) or (not os.path.exists(args.output_dir[0])):
      print("directories do not exist")
      sys.exit(1)

    train(args)
