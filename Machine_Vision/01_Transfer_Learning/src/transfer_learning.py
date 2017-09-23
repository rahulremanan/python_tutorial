import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, RMSprop, Adagrad
import keras 

IM_WIDTH, IM_HEIGHT = 299, 299 #fixed size for InceptionV3
NB_EPOCHS = 100
BAT_SIZE = 10
FC_SIZE = 1024
NB_IV3_LAYERS_TO_FREEZE = 172

sgd = SGD(lr=1e-7, decay=0.5, momentum=1, nesterov=True)
rms = RMSprop(lr=1e-7, rho=0.9, epsilon=1e-08, decay=0.0)
ada = Adagrad(lr=1e-7, epsilon=1e-08, decay=0.0)
optimizer = sgd

def get_load_model(bool_fn):
    load_model = bool(bool_fn)
    return load_model

def get_train_model(bool_fn):
    train_model = bool(bool_fn)
    return train_model

def get_fine_tune(bool_fn):
    tune_model = bool(bool_fn)
    return tune_model

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
  for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
     layer.trainable = False
  for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
     layer.trainable = True
  model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
  return model

def train(args):
  """Use transfer learning and fine-tuning to train a network on a new dataset"""
  nb_train_samples = get_nb_files(args.train_dir)
  print (nb_train_samples)
  nb_classes = len(glob.glob(args.train_dir + "/*"))
  print (nb_classes)
  nb_val_samples = get_nb_files(args.val_dir)
  nb_epoch = int(args.epoch)
  batch_size = int(args.batch)

  # data prep
  train_datagen =  ImageDataGenerator(preprocessing_function=preprocess_input,
      rotation_range=30,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True)
  
  test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
      rotation_range=30,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True)

  train_generator = train_datagen.flow_from_directory(args.train_dir,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size)

  validation_generator = test_datagen.flow_from_directory(
    args.val_dir,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size)
  
  base_model = InceptionV3(weights='imagenet', include_top=False) #include_top=False excludes final FC layer
  model = add_new_last_layer(base_model, nb_classes)
  print ("Inception model loaded...")
  print (model.summary)
  
  try:
      load_model = get_load_model(args.load_model)
      train_model = get_train_model(args.train_model)
      fine_tune_model = get_fine_tune(args.fine_tune)
      if load_model == True:
            model = keras.models.load_model(args.model_file)
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

            model.save(args.model_file)
      else:
          print ("Using pre-trained model for prediction...")
  except:
        print ("No pre-trained model or model weights loaded...")


  if args.plot:
    plot_training(model_train)

def plot_training(history):
  output_loc = args.output_dir
  acc = history.history['acc']
  val_acc = history.history['val_acc']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(len(acc))

  plt.plot(epochs, acc, 'r.')
  plt.plot(epochs, val_acc, 'r')
  plt.title('Training and validation accuracy')

  plt.figure()
  plt.plot(epochs, loss, 'r.')
  plt.plot(epochs, val_loss, 'r-')
  plt.title('Training and validation loss')
  fig = plt.figure()
  fig.savefig(output_loc)

if __name__=="__main__":
  a = argparse.ArgumentParser()
  a.add_argument("--train_dir")
  a.add_argument("--val_dir")
  a.add_argument("--epoch", default=NB_EPOCHS)
  a.add_argument("--batch", default=BAT_SIZE)
  a.add_argument("--model_file", default="/home/info/model/inceptionv3-ft.model")
  a.add_argument("--plot", action="store_true")
  a.add_argument("--output_dir")
  a.add_argument("--train_model", default=True)
  a.add_argument("--load_model", default=False)
  a.add_argument("--fine_tune", default=True)

  args = a.parse_args()
  if args.train_dir is None or args.val_dir is None:
    a.print_help()
    sys.exit(1)

  if (not os.path.exists(args.train_dir)) or (not os.path.exists(args.val_dir)):
    print("directories do not exist")
    sys.exit(1)

  train(args)
# Example usage: python3 transfer_learning.py --train_dir /home/info/train --val_dir /home/info/val --batch 20 --epoch 10 --model_file /home/info/transfer_learn_epoch100.model --output_dir /home/info/model --train_model True --load_model True --fine_tune True