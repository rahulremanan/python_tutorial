# Transfer learning using Keras and Tensorflow.
# Written by Rahul Remanan and MOAD (https://www.moad.computer) machine vision team.
# For more information contact: info@moad.computer
# License: MIT open source license (
# Repository: https://github.com/rahulremanan/python_tutorial/import time
import os
import time
import sys
import glob
import h5py
import json
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow
import keras
import PIL
from collections import defaultdict
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, model_from_json
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, RMSprop, Adagrad

IM_WIDTH, IM_HEIGHT = 299, 299                                                  # Fixed input image size for Inception version 3
DEFAULT_EPOCHS = 100
DEFAULT_BATCHES = 20
FC_SIZE = 1024
NB_LAYERS_TO_FREEZE = 169

sgd = SGD(lr=1e-7, decay=0.5, momentum=1, nesterov=True)
rms = RMSprop(lr=1e-7, rho=0.9, epsilon=1e-08, decay=0.0)
ada = Adagrad(lr=1e-7, epsilon=1e-08, decay=0.0)
optimizer = sgd

def generate_timestamp():
    timestring = time.strftime("%Y_%m_%d-%H_%M_%S")
    print ("Time stamp generated: " + timestring)
    return timestring

timestr = generate_timestamp()

def is_valid_file(parser, arg):
    if not os.path.isfile(arg):
        parser.error("The file %s does not exist ..." % arg)
    else:
        return arg
    
def is_valid_dir(parser, arg):
    if not os.path.isdir(arg):
        parser.error("The folder %s does not exist ..." % arg)
    else:
        return arg
 
def get_bool_fn(bool_fn):                                                       # Boolean filter
    try:
        load_value = bool(bool_fn)
        return load_value
    except:
        print (("Please check if: \
            --train_model \
            --load_model \
            --fine_tune \
            --test_aug \
            --plot \
            --summary \
            arguments are True or False statements..."))
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
  model.compile(optimizer=optimizer, 
                loss='categorical_crossentropy', metrics=['accuracy'])
  return model

def add_new_last_layer(base_model, nb_classes):                                 # Add the fully connected convolutional neural network layer
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(FC_SIZE, activation='relu')(x)                                      # New fully connected layer, random init
  predictions = Dense(nb_classes, activation='softmax')(x)                      # New softmax layer
  model = Model(inputs=base_model.input, outputs=predictions)
  return model

def setup_to_finetune(model, optimizer):                                        # Freeze the bottom NB_LAYERS and retrain the remaining top layers
  for layer in model.layers[:NB_LAYERS_TO_FREEZE]:
     layer.trainable = False
  for layer in model.layers[NB_LAYERS_TO_FREEZE:]:
     layer.trainable = True
  model.compile(optimizer=optimizer, loss='categorical_crossentropy', 
                metrics=['accuracy'])
  return model

def save_model(args, model):
    file_loc = args.output_di[0]
    file_pointer = os.path.join(file_loc+"//trained_"+ timestr)
    model.save_weights(os.path.join(file_pointer + "_weights.model"))
    # serialize model to JSON
    model_json = model.to_json()
    with open(os.path.join(file_pointer+"_model.json"), "w") as json_file:
        json_file.write(model_json)
    print ("Saved the trained model weights to: " + 
           str(os.path.join(file_pointer + ".model")))
    print ("Saved the trained model configuration as a json file to: " + 
           str(os.path.join(file_pointer+"_model.json")))

def generate_labels(args):
    file_loc = args.output_dir[0]
    data_dir = args.train_dir[0]
    file_pointer = os.path.join(file_loc+"//trained_"+ timestr)
    dt = defaultdict(list)
    for root, subdirs, files in os.walk(data_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            assert file_path.startswith(data_dir)
            suffix = file_path[len(data_dir):]
            suffix = suffix.lstrip("/")
            label = suffix.split("/")[0]
            dt[label].append(file_path)

    labels = sorted(dt.keys())

    with open(os.path.join(file_pointer+"_labels.json"), "w") as json_file:
        json.dump(labels, json_file)

    return labels

def generate_plot(args, model_train):
    gen_plot = get_bool_fn(args.plot[0])
    if gen_plot==True:
        plot_training(model_train)
    else:
        print ("No training summary plots generated ...")
        print ("Set: --plot True for creating training summary plots")

def plot_training(history):
  output_loc = args.output_dir[0]
  
  output_file_acc = os.path.join(output_loc+
                                 "//training_plot_acc_"+timestr+".png")
  output_file_loss = os.path.join(output_loc+
                                  "//training_plot_loss_"+timestr+".png")
  fig_acc = plt.figure()
  plt.plot(history.history['acc'])
  plt.plot(history.history['val_acc'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  fig_acc.savefig(output_file_acc, dpi=fig_acc.dpi)
  print ("Successfully created the training accuracy plot: " 
         + str(output_file_acc))
  plt.close()

  fig_loss = plt.figure()
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  fig_loss.savefig(output_file_loss, dpi=fig_loss.dpi)
  print ("Successfully created the loss function plot: " 
         + str(output_file_loss))
  plt.close()
        
def train(args):                                                                # Transfer learning and fine-tuning for training
  nb_train_samples = get_nb_files(args.train_dir[0])
  print ("Total number of training samples = " + str(nb_train_samples))
  nb_classes = len(glob.glob(args.train_dir[0] + "/*"))
  print ("Number of training categories = " + str(nb_classes))
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
      test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
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

  validation_generator = test_datagen.flow_from_directory(args.val_dir[0],
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size,
    class_mode='categorical')
  
  base_model = InceptionV3(weights='imagenet', include_top=False)               # Model argument: include_top=False excludes the final FC layer
  model = add_new_last_layer(base_model, nb_classes)
  print ("Base model for transfer learning: Inception version 3 ...")
  
  labels = generate_labels(args)
  
  print ((args.model_summary[0]))
  
  model_summary_ = get_bool_fn(args.model_summary[0])
  
  if model_summary_ == True:
      print (model.summary())
  else:
      print ("Successfully loaded Inception version 3 for training ...")
    
  load_model = get_bool_fn(args.load_model[0])
  
  fine_tune_model = get_bool_fn(args.fine_tune[0])
  
  if load_model == True:
      try:
          with open(args.config_file[0]) as json_file:
              model_json = json_file.read()
          model = model_from_json(model_json)
      except:
          model = model
      print ("Loading model weights from: " + str(args.weights_file[0]))
      model.load_weights(args.weights_file[0])
      print ("Successfully loaded saved model weights ...")
  else:
      model = model
      print ("Tabula rasa ...")
      
  if fine_tune_model == True:
      setup_to_finetune(model, optimizer)
  else:
      setup_to_transfer_learn(model, base_model, optimizer)
            
  print ("Initializing training with  category labels: " + 
         str(labels))
  
  model_train = model.fit_generator(train_generator,
                  epochs=nb_epoch,
                  steps_per_epoch=nb_train_samples // batch_size,
                  validation_data=validation_generator,
                  validation_steps=nb_val_samples // batch_size,
                  class_weight='auto')
  
  save_model(args, model)
  generate_plot(args, model_train)           

def get_user_options():
    a = argparse.ArgumentParser()
    
    a.add_argument("--training_directory", 
                 help = "Specify folder contraining the training files ...", 
                 dest = "train_dir", 
                 required = True, 
                 type=lambda x: is_valid_dir(a, x), 
                 nargs=1)
    
    a.add_argument("--validation_directory", 
                   help = "Specify folder containing the validation files ...", 
                   dest = "val_dir", 
                   required = True, 
                   type=lambda x: is_valid_dir(a, x), 
                   nargs=1)
    
    a.add_argument("--epochs", 
                   help = "Specify epochs for training ...", 
                   dest = "epoch", 
                   default=[DEFAULT_EPOCHS], 
                   required=False, 
                   type = int, 
                   nargs=1)
    
    a.add_argument("--batches", help = "Specify batches for training ...", 
                   dest = "batch", 
                   default=[DEFAULT_BATCHES], 
                   required=False, 
                   type = int, 
                   nargs=1)
    
    a.add_argument("--weights_file", 
                   help = "Specify pre-trained model weights file for training ...", 
                   dest = "weights_file", 
                   default=["./model/trained_weights.model"], 
                   required=False,
                   type=lambda x: is_valid_file(a, x),
                   nargs=1)
    
    a.add_argument("--config_file", 
                   help = "Specify pre-trained model configuration file ...", 
                   dest = "config_file", 
                   default=["./model/model.json"], 
                   required=False,
                   type=lambda x: is_valid_file(a, x),
                   nargs=1)
    
    a.add_argument("--output_directory", 
                   help = "Specify output folder ...", 
                   dest = "output_dir", 
                   required = True, 
                   type=lambda x: is_valid_dir(a, x),
                   nargs=1)
    
    a.add_argument("--train_model", 
                   help = "Specify if the model should be trained ...", 
                   dest = "train_model", 
                   required=True, 
                   default=[True], 
                   nargs=1, 
                   type = bool)
    
    a.add_argument("--load_model", 
                   help = "Specify if pre-trained model should be loaded ...", 
                   dest = "load_model", 
                   required=False, 
                   default=[False], 
                   nargs=1, 
                   type = bool)
    
    a.add_argument("--fine_tune", 
                   help = "Specify model should be fine tuned ...", 
                   dest = "fine_tune", 
                   required=False, 
                   default=[True], 
                   nargs=1, 
                   type = bool)
    
    a.add_argument("--test_augmentation", 
                   help = "Specify image augmentation for test dataset ...", 
                   dest = "test_aug", 
                   required=False, 
                   default=[False], 
                   nargs=1, 
                   type = bool)
    
    a.add_argument("--plot", 
                   help = "Specify if a plot should be generated ...", 
                   dest = "plot", 
                   required=False, 
                   default=[True], 
                   nargs=1, 
                   type = bool)
    
    a.add_argument("--summary", 
                   help = "Specify if a summary should be generated ...", 
                   dest = "model_summary", 
                   required=False, 
                   default=[False], 
                   type = bool,
                   nargs=1)
    
    args = a.parse_args()
    
    return args

if __name__=="__main__":
    args = get_user_options()

    if ((not os.path.exists(args.train_dir[0])) 
        or 
    (not os.path.exists(args.val_dir[0])) 
        or 
    (not os.path.exists(args.output_dir[0]))):
      print("Specified directories do not exist ...")
      sys.exit(1)
    train_model = get_bool_fn(args.train_model[0])  
    
    if train_model ==True:
        print ("Training sesssion initiated ...")
        train(args)
    else:
        print ("Nothing to do here ...")
        print ("Try setting the --train_model flag to True ...")
        print ("For more help, run with -h ...")
        sys.exit(1)