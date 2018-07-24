# !/usr/bin/python3.6
# -*- coding: utf-8 -*-
# Transfer learning using Keras and Tensorflow.
# Written by Rahul Remanan and MOAD (https://www.moad.computer) machine vision team.
# For more information contact: info@moad.computer
# License: MIT open source license
# Repository: https://github.com/rahulremanan/python_tutorial

import argparse
import os
import time
import sys
import glob
try:
    import h5py
except:
    print ('Package h5py needed for saving model weights ...')
    sys.exit(1)
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    import tensorflow
    import keras
except:
    print ('This code uses tensorflow deep-learning framework and keras api ...')
    print ('Install tensorflow and keras to train the classifier ...')
    sys.exit(1)
import PIL
from collections import defaultdict
from keras.applications.inception_v3 import InceptionV3,    \
                                            preprocess_input as preprocess_input_inceptionv3
from keras.applications.inception_resnet_v2 import InceptionResNetV2,    \
                                            preprocess_input as preprocess_input_inceptionv4
from keras.models import Model,                             \
                         model_from_json,                    \
                         load_model
from keras.layers import Dense,                             \
                         GlobalAveragePooling2D,            \
                         Dropout,                           \
                         BatchNormalization
from keras.layers.merge import concatenate
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD,                           \
                             RMSprop,                       \
                             Adagrad,                       \
                             Adadelta,                      \
                             Adam,                          \
                             Adamax,                        \
                             Nadam
from keras.callbacks import EarlyStopping,   \
                            ModelCheckpoint, \
                            ReduceLROnPlateau

IM_WIDTH, IM_HEIGHT = 299, 299                                                  # Default input image size for Inception v3 and v4 architecture
DEFAULT_EPOCHS = 100
DEFAULT_BATCHES = 20
FC_SIZE = 4096
DEFAULT_DROPOUT = 0.1
DEFAULT_NB_LAYERS_TO_FREEZE = 169

sgd = SGD(lr=1e-7, decay=0.5, momentum=1, nesterov=True)
rms = RMSprop(lr=1e-7, rho=0.9, epsilon=1e-08, decay=0.0)
ada = Adagrad(lr=1e-3, epsilon=1e-08, decay=0.0)
    
DEFAULT_OPTIMIZER = ada

def generate_timestamp():
    """ 
        A function to generate time-stamp information.
        Calling the function returns a string formatted current system time.
        Eg: 2018_10_10_10_10_10
    
        Example usage: generate_timestamp() 
    """    
    timestring = time.strftime("%Y_%m_%d-%H_%M_%S")
    print ("Time stamp generated: " + timestring)
    return timestring

timestr = generate_timestamp()

def is_valid_file(parser, arg):
    """
        A function that checks if a give file path contains a valid file or not.
        
        The function returns the full file path if there is a valid file persent.
        If there is no valid file present at a file path location, it returns a parser error message.
        
        Takes two positional arguments: parser and arg
        
        Example usage: 
            import argsparse
            
            a = argparse.ArgumentParser()
            a.add_argument("--file_path", 
                              help = "Check if a file exists in the specified file path ...", 
                              dest = "file_path", 
                              required=False,
                              type=lambda x: is_valid_file(a, x),
                              nargs=1)
            
            args = a.parse_args()
            
            args = get_user_options()
    """
    if not os.path.isfile(arg):
        try:
            parser.error("The file %s does not exist ..." % arg)
            return None
        except:
            if parser != None:
                print ("No valid argument parser found ...")
                print ("The file %s does not exist ..." % arg)
                return None
            else:
                print ("The file %s does not exist ..." % arg)
                return None
    else:
        return arg
    
def is_valid_dir(parser, arg):
    """
        This function checks if a directory exists or not.
        It can be used inside the argument parser.
        
        Example usage: 
            
            import argsparse
            
            a = argparse.ArgumentParser()
            a.add_argument("--dir_path", 
                              help = "Check if a file exists in the specified file path ...", 
                              dest = "file_path", 
                              required=False,
                              type=lambda x: is_valid_dir(a, x),
                              nargs=1)
            
            args = a.parse_args()
            
            args = get_user_options() 
    """
    if not os.path.isdir(arg):
        try:
            return parser.error("The folder %s does not exist ..." % arg)
        except:
            if parser != None:
                print ("No valid argument parser found")
                print ("The folder %s does not exist ..." % arg)
                return None
            else:
                print ("The folder %s does not exist ..." % arg)
                return None
    else:
        return arg
    
def string_to_bool(val):
    """
        A function that checks if an user argument is boolean or not.
        
        Example usage:
            
            
                import argsparse
            
                a = argparse.ArgumentParser()
                
                a.add_argument("--some_bool_arg", 
                   help = "Specify a boolean argument ...", 
                   dest = "some_bool_arg", 
                   required=False, 
                   default=[True], 
                   nargs=1, 
                   type = string_to_bool)
                
            args = a.parse_args()
            
            args = get_user_options()
            
    """
    if val.lower() in ('yes', 'true', 't', 'y', '1', 'yeah', 'yup'):
        return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0', 'none', 'nope'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected ...')
        
def activation_val(val):
    if val.lower() in ('hard_sigmoid',
                       'elu',
                       'linear',
                       'relu', 
                       'selu', 
                       'sigmoid',
                       'softmax',
                       'softplus',
                       'sofsign',
                       'tanh'):
        return True
    else:
        raise argparse.ArgumentTypeError('Unexpected activation function. Expected values are:  hard_sigmoid, elu, linear, relu, selu, sigmoid, softmax, softplus, sofsign, tanh ...')
        
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
                loss='categorical_crossentropy', 
                metrics=['accuracy'])
  return model

def add_top_layer(args, base_model, nb_classes):
  """
    This functions adds a fully connected convolutional neural network layer to a base model.
    
    The required input arguments for this function are: args, base_model and nb_classes.
        args: argument inputs the user arguments to be passed to the function,
        base_model: argument inputs the base model architecture to be added to the top layer,
        nb_classes: argument inputs the total number of classes for the output layer.    
  """
  try:
      dropout = float(args.dropout[0])
  except:
      dropout = DEFAULT_DROPOUT
      print ('Invalid input for dropout ...')
      
  try:
      activation = str(args.activation[0]).lower()
      print ('Building model using activation function: ' + str(activation))
  except:
      activation = 'relu'
      print ('Invalid input for activation function ...')
      print ('Choice of activation functions: hard_sigmoid, elu, linear, relu, selu, sigmoid, softmax, softplus, sofsign, tanh ...')
      print ('Building model using default activation function: relu')
      
  bm = base_model.output
  
  x = Dropout(dropout)(bm)
  x = GlobalAveragePooling2D()(x)
  x = Dropout(dropout)(x)
  x = BatchNormalization()(x)
  x = Dense(FC_SIZE, activation=activation)(x)
  x = Dropout(dropout)(x)
  
  x1 = Dense(FC_SIZE, activation=activation, name="fc_dense1")(x)
  x1 = Dropout(dropout, name = 'dropout1')(x1)
  x1 = BatchNormalization(name="fc_batch_norm1")(x1)
  x1 = Dense(FC_SIZE, activation=activation, name="fc_dense2")(x1)
  x1 = Dropout(dropout, name = 'dropout2')(x1)

  x2 = Dense(FC_SIZE, activation=activation, name="fc_dense3")(x)
  x2 = Dropout(dropout, name = 'dropout3')(x2)
  x2 = BatchNormalization(name="fc_batch_norm2")(x2)
  x2 = Dense(FC_SIZE, activation=activation, name="fc_dense4")(x2)
  x2 = Dropout(dropout, name = 'dropout4')(x2)

  x12 = concatenate([x1, x2], name = 'mixed11')
  x12 = Dropout(dropout, name = 'dropout5')(x12)
  x12 = Dense(FC_SIZE//16, activation=activation, name = 'fc_dense5')(x12)
  x12 = Dropout(dropout, name = 'dropout6')(x12)
  x12 = BatchNormalization(name="fc_batch_norm3")(x12)
  x12 = Dense(FC_SIZE//32, activation=activation, name = 'fc_dense6')(x12)
  x12 = Dropout(dropout, name = 'dropout7')(x12)
  
  x3 = GlobalAveragePooling2D( name = 'global_avg_pooling2')(bm)
  x3 = Dense(FC_SIZE//2, activation=activation, name = 'fc_dense7')(x3)
  x3 = Dropout(dropout, name = 'dropout8')(x3)
  x3 = BatchNormalization(name="fc_batch_norm4")(x3)
  x3 = Dense(FC_SIZE//2, activation=activation, name = 'fc_dense8')(x3)
  x3 = Dropout(dropout, name = 'dropout9')(x3)
  
  xout = concatenate([x12, x3], name ='mixed12')
  xout = Dense(FC_SIZE//32, activation= activation, name = 'fc_dense9')(xout)
  xout = Dropout(dropout, name = 'dropout10')(xout)
  
  predictions = Dense(nb_classes,           \
                      activation='softmax', \
                      name='prediction')(xout) # Softmax output layer
  
  model = Model(inputs=base_model.input, outputs=predictions)
  
  return model

def setup_to_finetune(model, optimizer, NB_FROZEN_LAYERS):
  """
      A function that freezes the bottom NB_LAYERS and retrain the remaining top layers.
      
      The required input arguments for this function are: model, optimizer and NB_FROZEN_LAYERS.
          model: inputs a model architecture with base layers to be frozen during training,
          optimizer: inputs an choice of optimizer values for compiling the model,
          NB_FROZEN_LAYERS: inputs a number that selects the total number of base layers to be frozen during training.
      
  """
                     
  for layer in model.layers[:NB_FROZEN_LAYERS]:
     layer.trainable = False
  for layer in model.layers[NB_FROZEN_LAYERS:]:
     layer.trainable = True
  model.compile(optimizer=optimizer, loss='categorical_crossentropy', 
                metrics=['accuracy'])
  return model

def save_model(args, name, model):
    file_loc = args.output_dir[0]
    file_pointer = os.path.join(file_loc+"//trained_"+ timestr)
    model.save_weights(os.path.join(file_pointer + "_weights"+str(name)+".model"))
    
    model_json = model.to_json()                                                # Serialize model to JSON
    with open(os.path.join(file_pointer+"_config"+str(name)+".json"), "w") as json_file:
        json_file.write(model_json)
    print ("Saved the trained model weights to: " + 
           str(os.path.join(file_pointer + "_weights"+str(name)+".model")))
    print ("Saved the trained model configuration as a json file to: " + 
           str(os.path.join(file_pointer+"_config"+str(name)+".json")))

def generate_labels(args):
    file_loc = args.output_dir[0]
    file_pointer = os.path.join(file_loc+"//trained_labels")
    
    data_dir = args.train_dir[0]
    val_dir_ = args.val_dir[0]
    
    dt = defaultdict(list)
    dv = defaultdict(list)
    
    for root, subdirs, files in os.walk(data_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            assert file_path.startswith(data_dir)
            suffix = file_path[len(data_dir):]
            suffix = suffix.lstrip("/")
            label = suffix.split("/")[0]
            dt[label].append(file_path)
            
    for root, subdirs, files in os.walk(val_dir_):
        for filename in files:
            file_path = os.path.join(root, filename)
            assert file_path.startswith(val_dir_)
            suffix = file_path[len(val_dir_):]
            suffix = suffix.lstrip("/")
            label = suffix.split("/")[0]
            dv[label].append(file_path)

    labels = sorted(dt.keys())
    val_labels = sorted(dv.keys())
    
    if set(labels) == set (val_labels):
        print("Training labels: " + str(labels))
        print("Validation labels: " + str(val_labels))
        with open(os.path.join(file_pointer+".json"), "w") as json_file:
            json.dump(labels, json_file)
    else:
      print("Training labels: " + str(labels))
      print("Validation labels: " + str(val_labels))
      print ("Mismatched training and validation data labels ...")
      print ("Sub-folder names do not match between training and validation directories ...")
      sys.exit(1)

    return labels

def generate_plot(args, name, model_train):
    gen_plot = args.plot[0]
    if gen_plot==True:
        plot_training(args, name, model_train)
    else:
        print ("No training summary plots generated ...")
        print ("Set: --plot True for creating training summary plots")

def plot_training(args, name, history):
  output_loc = args.output_dir[0]
  
  output_file_acc = os.path.join(output_loc+
                                 "//training_plot_acc_" + 
                                 timestr+str(name)+".png")
  output_file_loss = os.path.join(output_loc+
                                  "//training_plot_loss_" + 
                                  timestr+str(name)+".png")
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
        
def train(args): 
  """
    A function that takes the user arguments and initiates a training session of the neural network.
    
    This function takes only one input: args
    
    Example usage:
            
        if train_model == True:
            print ("Training sesssion initiated ...")
            train(args)
  """    

  
  if not os.path.exists(args.output_dir[0]):
    os.makedirs(args.output_dir[0])
    
  optimizer_val = args.optimizer_val[0]
  lr = args.learning_rate[0]
  decay = args.decay[0]
  epsilon = args.epsilon[0]
  
  if optimizer_val.lower() == 'sgd' :
    optimizer = SGD(lr=lr, decay=decay, momentum=1, nesterov=False)
    print ("Using SGD as the optimizer ...")
  elif optimizer_val.lower() == 'nsgd':
    optimizer = SGD(lr=lr, decay=decay, momentum=1, nesterov=True)
    print ("Using SGD as the optimizer with Nesterov momentum ...")
  elif optimizer_val.lower() == 'rms' or optimizer_val.lower() == 'rmsprop':
    optimizer = RMSprop(lr=lr, rho=0.9, epsilon=epsilon, decay=decay)
    print ("Using RMSProp as the optimizer ...")
  elif optimizer_val.lower() == 'ada' or optimizer_val.lower() == 'adagrad':
    optimizer = Adagrad(lr=lr, epsilon=epsilon, decay=decay)
    print ("Using Adagrad as the optimizer ...")
  elif optimizer_val.lower() == 'adelta' or optimizer_val.lower() == 'adadelta':
    optimizer = Adadelta(lr=lr, rho=0.95, epsilon=epsilon, decay=0.0)
    print ("Using Adadelta as the optimizer ...")
  elif optimizer_val.lower() == 'adam':
    optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=epsilon, decay=decay, amsgrad=False)
    print ("Using Adam as the optimizer ...")
  elif optimizer_val.lower() == 'amsgrad':
    optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=epsilon, decay=decay, amsgrad=True)
    print ("Using AmsGrad variant of Adam as the optimizer ...")
  elif optimizer_val.lower() == 'adamax':  
    optimizer = Adamax(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=epsilon, decay=decay)
    print ("Using Adamax variant of Adam as the optimizer ...")
  elif optimizer_val.lower() == 'nadam':  
    optimizer = Nadam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=epsilon, schedule_decay=decay)
    print ("Using Nesterov Adam optimizer ...\
           \n decay arguments is passed on to schedule_decay variable ...")
  else:
      optimizer = DEFAULT_OPTIMIZER
      print ("Options for optimizer are: 'sgd', 'nsgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'nadam', 'amsgrad', 'adamax' ...")
      
  nb_train_samples = get_nb_files(args.train_dir[0])
  nb_classes = len(glob.glob(args.train_dir[0] + "/*"))
  
  print ("Total number of training samples = " + str(nb_train_samples))
  print ("Number of training classes = " + str(nb_classes))
  
  nb_val_samples = get_nb_files(args.val_dir[0])
  nb_val_classes = len(glob.glob(args.val_dir[0] + "/*"))
  
  print ("Total number of validation samples = " + str(nb_val_samples))
  print ("Number of validation classes = " + str(nb_val_classes))
  
  if nb_val_classes == nb_classes:
      print ("Initiating training session ...")
  else:
      print ("Mismatched number of training and validation data classes ...")
      print ("Unequal number of sub-folders found between train and validation directories ...")
      print ("Each sub-folder in train and validation directroies are treated as a separate class ...")
      print ("Correct this mismatch and re-run ...")
      print ("Now exiting ...")
      sys.exit(1)
      
  nb_epoch = int(args.epoch[0])
  batch_size = int(args.batch[0])    
  train_aug = args.train_aug[0] 
  
  
  if str((args.base_model[0]).lower()) == 'inceptionv4' or  \
     str((args.base_model[0]).lower()) == 'inception_v4' or \
     str((args.base_model[0]).lower()) == 'inception_resnet':
      preprocess_input = preprocess_input_inceptionv4
  else:
      preprocess_input = preprocess_input_inceptionv3
  
  if train_aug==True:
    train_datagen =  ImageDataGenerator(preprocessing_function=preprocess_input,
      rotation_range=30,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True)
  else:
      train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    
  
  test_aug = args.test_aug[0]  
  
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
      test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
      
  print ("Generating training data: ... ")

  train_generator = train_datagen.flow_from_directory(args.train_dir[0],
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size,
    class_mode='categorical')
  
  print ("Generating validation data: ... ")

  validation_generator = test_datagen.flow_from_directory(args.val_dir[0],
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size,
    class_mode='categorical')
  
  if str((args.base_model[0]).lower()) == 'inceptionv4' or  \
     str((args.base_model[0]).lower()) == 'inception_v4' or \
     str((args.base_model[0]).lower()) == 'inception_resnet':
      base_model = InceptionResNetV2(weights='imagenet', \
                                     include_top=False)
      base_model_name = 'Inception version 4'
  else:
      base_model = InceptionV3(weights='imagenet', 
                               include_top=False)               # Model argument: include_top=False excludes the final FC layer
      base_model_name = 'Inception version 3'
  print ('Base model: ' + str(base_model_name))
  
  model = add_top_layer(args, base_model, nb_classes)
  print ("New top layer added to: " + str(base_model_name))
  
  labels = generate_labels(args)
  
  load_weights_ = args.load_weights[0]
  fine_tune_model = args.fine_tune[0]
  load_checkpoint = args.load_checkpoint[0]
  
  checkpointer_savepath = os.path.join(args.output_dir[0]     +       
                                       '/checkpoint/Transfer_learn_' +       
                                       str(IM_WIDTH)  + '_'  + 
                                       str(IM_HEIGHT) + '_'  + '.h5')
   
  if load_weights_ == True:     
      try:
          with open(args.config_file[0]) as json_file:
              model_json = json_file.read()
          model = model_from_json(model_json)
      except:
          model = model
      try:
          model.load_weights(args.weights_file[0])
          print ("Loaded model weights from: " + str(args.weights_file[0]))
      except:
          print ("Error loading model weights ...")
          print ("Tabula rasa ...")
          print ("Loaded default model weights ...")
  elif load_checkpoint == True and os.path.exists(checkpointer_savepath):     
      try:
          model = load_model(checkpointer_savepath)
          print ("Loaded model from checkpoint: " + str(checkpointer_savepath))
      except:
          if os.path.exists(args.saved_chkpnt[0]):
            model = load_model(args.saved_chkpnt[0])
            print ('Loaded saved checkpoint file ...')
          else:
            print ("Error loading model checkpoint ...")
            print ("Tabula rasa ...")
            print ("Loaded default model weights ...")
  else:
      model = model
      print ("Tabula rasa ...")
      print ("Loaded default model weights ...")
 
  try:
      NB_FROZEN_LAYERS = args.frozen_layers[0]
  except:
      NB_FROZEN_LAYERS = DEFAULT_NB_LAYERS_TO_FREEZE
      
  if fine_tune_model == True:
      print ("Fine tuning Inception architecture ...")
      print ("Frozen layers: " + str(NB_FROZEN_LAYERS))
      setup_to_finetune(model, optimizer, NB_FROZEN_LAYERS)
  else:
      print ("Transfer learning using Inception architecture ...")
      setup_to_transfer_learn(model, base_model, optimizer)
            
  print ("Initializing training with  class labels: " + 
         str(labels))
  
  model_summary_ = args.model_summary[0]
  
  if model_summary_ == True:
      print (model.summary())
  else:
      print ("Successfully loaded deep neural network classifier for training ...")
        
  if not os.path.exists(os.path.join(args.output_dir[0] + '/checkpoint/')):
    os.makedirs(os.path.join(args.output_dir[0] + '/checkpoint/'))
    
  earlystopper = EarlyStopping(patience=6, verbose=1)
  checkpointer = ModelCheckpoint(checkpointer_savepath, 
                                 verbose=1,  
                                 save_best_only=True)
  learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                              patience=2,
                                              mode = 'max',
                                              epsilon=1e-4, 
                                              cooldown=1,
                                              verbose=1, 
                                              factor=0.5, 
                                              min_lr=lr*1e-2)
  
  model_train = model.fit_generator(train_generator,
                                    epochs=nb_epoch,
                                    validation_data=validation_generator,
                                    class_weight='auto', 
                                    callbacks=[earlystopper, 
                                               learning_rate_reduction, 
                                               checkpointer])
  
  if fine_tune_model == True:
      save_model(args, "_ft_", model)
      generate_plot(args, "_ft_", model_train)
  else:
      save_model(args, "_tl_", model)
      generate_plot(args, "_tl_", model_train)
      
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
                   help = "Specify pre-trained model weights for training ...", 
                   dest = "weights_file", 
                   required=False,
                   type=lambda x: is_valid_file(a, x),
                   nargs=1)
    
    a.add_argument("--checkpoints_file", 
                   help = "Specify saved checkpoint weights for resuming training ...", 
                   dest = "saved_chkpnt", 
                   required=False,
                   type=lambda x: is_valid_file(a, x),
                   nargs=1)
    
    a.add_argument("--config_file", 
                   help = "Specify pre-trained model configuration file ...", 
                   dest = "config_file",  
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
                   type = string_to_bool)
    
    a.add_argument("--load_truncated", 
                   help = "Specify if truncated image loading should be supported ...", 
                   dest = "load_truncated", 
                   required=False, 
                   default=[False], 
                   nargs=1, 
                   type = string_to_bool)
    
    a.add_argument("--load_weights", 
                   help = "Specify if pre-trained model should be loaded ...", 
                   dest = "load_weights", 
                   required=False, 
                   default=[False], 
                   nargs=1, 
                   type = string_to_bool)
    
    
    a.add_argument("--load_checkpoint", 
                   help = "Specify if checkpointed weights are to be used ...", 
                   dest = "load_checkpoint", 
                   required=False, 
                   default=[True], 
                   nargs=1, 
                   type = string_to_bool)
    
    a.add_argument("--fine_tune", 
                   help = "Specify model should be fine tuned ...", 
                   dest = "fine_tune", 
                   required=False, 
                   default=[True], 
                   nargs=1, 
                   type = string_to_bool)
    
    a.add_argument("--test_augmentation", 
                   help = "Specify image augmentation for test dataset ...", 
                   dest = "test_aug", 
                   required=False, 
                   default=[False], 
                   nargs=1, 
                   type = string_to_bool)
    
    a.add_argument("--train_augmentation", 
                   help = "Specify image augmentation for train dataset ...", 
                   dest = "train_aug", 
                   required=False, 
                   default=[False], 
                   nargs=1, 
                   type = string_to_bool)
    
    a.add_argument("--plot", 
                   help = "Specify if a plot should be generated ...", 
                   dest = "plot", 
                   required=False, 
                   default=[True], 
                   nargs=1, 
                   type = string_to_bool)
    
    a.add_argument("--summary", 
                   help = "Specify if a summary should be generated ...", 
                   dest = "model_summary", 
                   required=False, 
                   default=[False], 
                   type = string_to_bool,
                   nargs=1)
    
    a.add_argument("--dropout", 
                   help = "Specify values for dropout function ...", 
                   dest = "dropout", 
                   required=False, 
                   default=[0.4], 
                   type = float,
                   nargs=1)
    
    a.add_argument("--epsilon", 
                   help = "Specify values for epsilon function ...", 
                   dest = "epsilon", 
                   required=False, 
                   default=[1e-8], 
                   type = float,
                   nargs=1)
    
    a.add_argument("--activation", 
                   help = "Specify values for activation function.\
                           Available activation functions are: hard_sigmoid, elu, linear, relu, selu, sigmoid, softmax, softplus, sofsign, tanh ...", 
                   dest = "activation", 
                   required=False, 
                   default=['relu'], 
                   type = activation_val,
                   nargs=1)
    
    a.add_argument("--learning_rate", 
                   help = "Specify values for learning rate ...", 
                   dest = "learning_rate", 
                   required=False, 
                   default=[1e-07], 
                   type = float,
                   nargs=1)
    
    a.add_argument("--decay", 
                   help = "Specify values for decay function ...", 
                   dest = "decay", 
                   required=False, 
                   default=[0.0], 
                   type = float,
                   nargs=1)
    
    a.add_argument("--optimizer", 
                   help = "Specify the type of optimizer to choose from. \
                           \n Options for optimizer are: 'sgd', 'nsgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'nadam', 'amsgrad', 'adamax' ...", 
                   dest = "optimizer_val", 
                   required=False, 
                   default=['rms'], 
                   nargs=1)
    
    a.add_argument("--base_model", 
                   help = "Specify the type of base model classifier to build the neural network. Options are: Inception_v4 or Inception_v3", 
                   dest = "base_model", 
                   required=False, 
                   default=['Inception_V4'], 
                   nargs=1)
    
    a.add_argument("--frozen_layers", 
                   help = "Specify the number of frozen bottom layers during fine-tuning ...", 
                   dest = "frozen_layers", 
                   required=False, 
                   default=[DEFAULT_NB_LAYERS_TO_FREEZE],
                   type = int,
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
    train_model = args.train_model[0]
    if args.load_truncated[0]:
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
    if train_model ==True:
        print ("Training sesssion initiated ...")
        train(args)
    else:
        print ("Nothing to do here ...")
        print ("Try setting the --train_model flag to True ...")
        print ("For more help, run with -h flag ...")
        sys.exit(1)