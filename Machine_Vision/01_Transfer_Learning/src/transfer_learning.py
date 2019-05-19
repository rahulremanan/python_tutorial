# !/usr/bin/python3.6
# -*- coding: utf-8 -*-
# Transfer learning using Keras and Tensorflow.
# Written by Rahul Remanan and MOAD (https://www.moad.computer) machine vision team.
# For more information contact: info@moad.computer
# License: MIT open source license
# Repository: https://github.com/rahulremanan/python_tutorial

import argparse
import os
import random
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
from keras.regularizers import l2
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
                            
from multiprocessing import Process
from execute_in_shell import execute_in_shell

IM_WIDTH, IM_HEIGHT = 299, 299                                                  # Default input image size for Inception v3 and v4 architecture
DEFAULT_EPOCHS = 100
DEFAULT_BATCHES = 20
FC_SIZE = 4096
DEFAULT_DROPOUT = 0.1
DEFAULT_NB_LAYERS_TO_FREEZE = 169

verbose = False

sgd = SGD(lr=1e-7, decay=0.5, momentum=1, nesterov=True)
rms = RMSprop(lr=1e-7, rho=0.9, epsilon=1e-08, decay=0.0)
ada = Adagrad(lr=1e-3, epsilon=1e-08, decay=0.0)
    
DEFAULT_OPTIMIZER = sgd

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
    activation_function_options = ('hard_sigmoid',
                                   'elu',
                                   'linear',
                                   'relu', 
                                   'selu', 
                                   'sigmoid',
                                   'softmax',
                                   'softplus',
                                   'sofsign',
                                   'tanh')
    if val.lower() in activation_function_options:
        return val
    else:
        raise argparse.ArgumentTypeError('Unexpected activation function. \
                                         \nExpected values are:  {} ...'.format(activation_function_options))

def loss_val(val):
    loss_function_options = ('mean_squared_error',
                             'mean_absolute_error',
                             'mean_absolute_percentage_error',
                             'mean_squared_logarithmic_error', 
                             'squared_hinge', 
                             'hinge',
                             'categorical_hinge',
                             'logcosh',
                             'categorical_crossentropy',
                             'sparse_categorical_crossentropy',
                             'binary_crossentropy',
                             'kullback_leibler_divergence',
                             'poisson',
                             'cosine_proximity')
    if val.lower() in loss_function_options:
        return val
    else:
        raise argparse.ArgumentTypeError('Unexpected loss function. \
                                         \nExpected values are:  {} ...'.format(loss_function_options))
        
def get_nb_files(directory):
  if not os.path.exists(directory):
    return 0
  cnt = 0
  for r, dirs, files in os.walk(directory):
    for dr in dirs:
      cnt += len(glob.glob(os.path.join(r, dr + "/*")))
  return cnt

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
      weight_decay = float(args.decay[0])
      enable_dropout = args.enable_dropout[0]
  except:
      dropout = DEFAULT_DROPOUT
      weight_decay = 0.01
      enable_dropout = True
      print ('Invalid user input ...')
      
  try:
      activation = str(args.activation[0]).lower()
      print ('Building model using activation function: ' + str(activation))
  except:
      activation = 'relu'
      print ('Invalid user input for activation function ...')
      print ('Choice of activation functions: hard_sigmoid, elu, linear, relu, selu, sigmoid, softmax, softplus, sofsign, tanh ...')
      print ('Building model using default activation function: relu')
      
  base_model.trainable = False
  bm = base_model.output
  
  x = Dropout(dropout,
              name='gloablDropout')(bm,
                                      training=enable_dropout)
  gap = GlobalAveragePooling2D(name='gloablAveragePooling2D')(x)
  bn = BatchNormalization(name='gloabl_batchNormalization')(x)
  
  enable_attention = args.enable_attention[0]
  enable_multilayerDense = args.enable_multilayerDense[0]
  ATTN_UNIT_SIZE = 256
  ATTN_CONV_LAYER_DEPTH = 2
  
  if enable_attention:
    """
    Covolutional attention layers 
    """
    preTrained_featureSize = base_model.get_output_shape_at(0)[-1]
    x = bn
    for i in range(ATTN_CONV_LAYER_DEPTH):
        x = Conv2D(ATTN_UNIT_SIZE, 
                   kernel_size=(1,1),
                   padding='same',
                   activation=activation,
                   name='convAttentionLayer_{}'.format(i))(x)
        x = Dropout(dropout,
                    name='attentionDropout_{}'.format(i))(x,
                                               training=enable_dropout)
    
    
    x = Conv2D(1, 
               kernel_size=(1,1),
               padding='valid',
               activation=activation,
               name='convAttentionLayer_1D')(x)
    x = Dropout(dropout,
                name='attentionDropout_1D')(x,
                                           training=enable_dropout)
    
    upConv2d_weights = np.ones((1, 1, 1, 1, preTrained_featureSize))
    
    upConv2d = Conv2D(preTrained_featureSize,
                      kernel_size = (1,1), 
                      padding = 'same', 
                      activation = 'linear', 
                      use_bias = False, 
                      weights = upConv2d_weights,
                      name='upConv2d')
    upConv2d.trainable = False
    
    x = upConv2d(x)
    maskFeatures = multiply([x,
                             bn],
                            name='multiply_maskFeature')
    
    gapFeatures = GlobalAveragePooling2D(name='attentionGlobalAveragePooling_features')(maskFeatures)
    gapMask = GlobalAveragePooling2D(name='attentionGlobalAveragePooling_mask')(x)
  
    gap = Lambda(lambda x: x[0]/x[1], 
                 name = 'rescaleGlobalAeragePooling')([gapFeatures,
                                                       gapMask])
  if enable_multilayerDense:
    x = Dropout(dropout,
                name='dropout_fc1')(gap,
                                    training=enable_dropout)
    x = BatchNormalization(name='batchNormalization_fc1')(x)
    x = Dense(FC_SIZE, 
              activation=activation,
              kernel_regularizer=l2(weight_decay),
              name='dense_fc1')(x)
    x = Dropout(dropout,
                name='dropout_fc2')(x,
                                    training=enable_dropout)
  
    x1 = Dense(FC_SIZE, 
               activation=activation,
               kernel_regularizer=l2(weight_decay),
               name="dense_fc2")(x)
    x1 = Dropout(dropout,
                 name = 'dropout_fc3')(x1, 
                                       training=enable_dropout)
    x1 = BatchNormalization(name="batchNormalization_fc2")(x1)
    x1 = Dense(FC_SIZE, 
               activation=activation, 
               kernel_regularizer=l2(weight_decay),
               name="dense_fc3")(x1)
    x1 = Dropout(dropout,
                 name = 'dropout_fc4')(x1, 
                                  training=enable_dropout)

    x2 = Dense(FC_SIZE, 
               activation=activation, 
               kernel_regularizer=l2(weight_decay),
               name="dense_fc4")(x)
    x2 = Dropout(dropout,
                 name = 'dropout_fc5')(x2, 
                                  training=enable_dropout)
    x2 = BatchNormalization(name="batchNormalization_fc3")(x2)
    x2 = Dense(FC_SIZE, 
               activation=activation, 
               kernel_regularizer=l2(weight_decay),
               name="dense_fc5")(x2)
    x2 = Dropout(dropout,
                 name = 'dropout_fc6')(x2, 
                                       training=enable_dropout)

    x12 = concatenate([x1, x2], name = 'mixed11')
    x12 = Dropout(dropout,
                  name = 'dropout_fc7')(x12, 
                                        training=enable_dropout)
    x12 = Dense(FC_SIZE//16, 
                activation=activation, 
                kernel_regularizer=l2(weight_decay),
                name = 'dense_fc6')(x12)
    x12 = Dropout(dropout,
                  name = 'dropout_fc8')(x12, 
                                        training=enable_dropout)
    x12 = BatchNormalization(name="batchNormalization_fc4")(x12)
    x12 = Dense(FC_SIZE//32, 
                activation=activation, 
                kernel_regularizer=l2(weight_decay),
                name = 'dense_fc7')(x12)
    x12 = Dropout(dropout,
                  name = 'dropout_fc9')(x12, 
                                         training=enable_dropout)
  
    x3 = Dense(FC_SIZE//2, 
               activation=activation, 
               kernel_regularizer=l2(weight_decay),
               name = 'dense_fc8')(gap)
    x3 = Dropout(dropout,
                 name = 'dropout_fc11')(x3, 
                                  training=enable_dropout)
    x3 = BatchNormalization(name="batchNormalization_fc5")(x3)
    x3 = Dense(FC_SIZE//2, 
               activation=activation, 
               kernel_regularizer=l2(weight_decay),
               name = 'dense_fc9')(x3)
    x3 = Dropout(dropout,
                 name = 'dropout_fc12')(x3, 
                                        training=enable_dropout)
  
    xout = concatenate([x12, x3], name ='mixed12')
    xout = Dense(FC_SIZE//32, 
                 activation= activation, 
                 kernel_regularizer=l2(weight_decay),
                 name = 'dense_fc10')(xout)
    xout = Dropout(dropout,
                   name = 'dropout_fc13')(xout, 
                                     training=enable_dropout)
    
  else:
    x = BatchNormalization(name='batchNormalization_fc1')(gap)
    xout = Dense(FC_SIZE, 
                 activation=activation,
                 kernel_regularizer=l2(weight_decay),
                 name='dense_fc1')(x)
    xout = Dropout(dropout,
                   name = 'dropout_fc13')(xout, 
                                          training=enable_dropout)
    
  predictions = Dense(nb_classes,           \
                      activation='softmax', \
                      kernel_regularizer=l2(weight_decay),
                      name='prediction')(xout) # Softmax output layer
  model = Model(inputs=base_model.input, 
                outputs=predictions)
  
  return model



def finetune_model(model, base_model, optimizer, loss, NB_FROZEN_LAYERS):
  """
      A function that freezes the bottom NB_LAYERS and retrain the remaining top layers.
      
      The required input arguments for this function are: model, optimizer and NB_FROZEN_LAYERS.
          model: inputs a model architecture with base layers to be frozen during training,
          optimizer: inputs a choice of optimizer value for compiling the model,
          loss: inputs a choice for loss function used for compiling the model,
          NB_FROZEN_LAYERS: inputs a number that selects the total number of base layers to be frozen during training.
      
  """
                     
  for layer in base_model.layers[:NB_FROZEN_LAYERS]:
     layer.trainable = False
  for layer in base_model.layers[NB_FROZEN_LAYERS:]:
     layer.trainable = True
  model.compile(optimizer=optimizer, 
                loss=loss, 
                metrics=['accuracy'])
  return model

def transferlearn_model(model, base_model, optimizer, loss):
  """
     Function that freezes the base layers to train just the top layer.
     
     This function takes three positional arguments:
         model: specifies the input model,
         base_model: specifies the base model architecture,
         optimizer: optimizer function for training the model,
         loss: loss function for compiling the model
     
     Example usage:
         transferlearn_model(model, base_model, optimizer)
  """
  for layer in base_model.layers:
    layer.trainable = False
  model.compile(optimizer=optimizer, 
                loss=loss, 
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
        print("\nTraining labels: " + str(labels))
        print("\nValidation labels: " + str(val_labels))
        with open(os.path.join(file_pointer+".json"), "w") as json_file:
            json.dump(labels, json_file)
    else:
      print("\nTraining labels: " + str(labels))
      print("\nValidation labels: " + str(val_labels))
      print ("Mismatched training and validation data labels ...")
      print ("Sub-folder names do not match between training and validation directories ...")
      sys.exit(1)

    return labels

def normalize(args, 
              labels, 
              move = False, 
              sub_sample = False):
    if args.normalize[0] and os.path.exists(args.root_dir[0]):      
        commands = ["rm -r {}/.tmp_train/".format(args.root_dir[0]),
                    "rm -r {}/.tmp_validation/".format(args.root_dir[0]),
                    "mkdir {}/.tmp_train/".format(args.root_dir[0]),
                    "mkdir {}/.tmp_validation/".format(args.root_dir[0])]
        execute_in_shell(command=commands,
                         verbose=verbose)
        del commands
        
        mk_train_folder = "mkdir -p {}/.tmp_train/".format(args.root_dir[0]) + "{}"
        mk_val_folder = "mkdir -p {}/.tmp_validation/".format(args.root_dir[0]) + "{}"
        
        train_class_sizes = []
        val_class_sizes = []
        
        for label in labels:
            train_class_sizes.append(len(glob.glob(args.train_dir[0] + "/{}/*".format(label))))
            val_class_sizes.append(len(glob.glob(args.val_dir[0] + "/{}/*".format(label))))
        
        train_size = min(train_class_sizes)
        val_size = min(val_class_sizes)
        
        try:
          if sub_sample and 0 <= args.train_sub_sample[0] <=1 and 0 <= args.val_sub_sample[0] <=1 :
              train_size = int(train_size * args.train_sub_sample[0])
              val_size = int(val_size * args.val_sub_sample[0])
        except:
          print ('Sub sample mode disabled ...')
        
        print ("Normalized training class size {}".format(train_size))
        print ("Normalized validation class size {}".format(val_size))
        
        for label in labels:
            commands = [mk_train_folder.format(label),
                        mk_val_folder.format(label)]
        
            execute_in_shell(command=commands,
                             verbose=verbose)
            del commands
        
        commands = []
        
        for label in labels:
            train_images = (glob.glob('{}/{}/*.*'.format(args.train_dir[0], label), recursive=True))
            val_images = (glob.glob('{}/{}/*.*'.format(args.val_dir[0], label), recursive=True))
            
            sys_rnd = random.SystemRandom()
            
            if move:
              cmd = 'mv'
            else:
              cmd = 'cp'
            
            for file in sys_rnd.sample(train_images, train_size):
                if os.path.exists(file):
                    commands.append('{} {} {}/.tmp_train/{}/'.format(cmd, file, args.root_dir[0], label))
            
            for file in sys_rnd.sample(val_images, val_size):
                if os.path.exists(file):
                    commands.append('{} {} {}/.tmp_validation/{}/'.format(cmd, file, args.root_dir[0], label))
                
            p = Process(target=execute_in_shell, args=([commands]))
            p.start()
            p.join()
        print ("\nData normalization pipeline completed successfully ...")
    else:
        print ("\nFailed to initiate data normalization pipeline ...")
        return False, None, None
    return True, train_size, val_size

def generate_plot(args, name, model_train):
    gen_plot = args.plot[0]
    if gen_plot==True:
        plot_training(args, name, model_train)
    else:
        print ("\nNo training summary plots generated ...")
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
        
def select_optimizer(args):
  optimizer_val = args.optimizer_val[0]
  lr = args.learning_rate[0]
  decay = args.decay[0]
  epsilon = args.epsilon[0]
  rho = args.rho[0]
  beta_1 = args.beta_1[0]
  beta_2 = args.beta_2[0]
  
  if optimizer_val.lower() == 'sgd' :
    optimizer = SGD(lr=lr,       \
                    decay=decay, \
                    momentum=1,  \
                    nesterov=False)
    print ("Using SGD as the optimizer ...")
  elif optimizer_val.lower() == 'nsgd':
    optimizer = SGD(lr=lr,      \
                    decay=decay,\
                    momentum=1, \
                    nesterov=True)
    print ("Using SGD as the optimizer with Nesterov momentum ...")
  elif optimizer_val.lower() == 'rms' \
       or \
       optimizer_val.lower() == 'rmsprop':
    optimizer = RMSprop(lr=lr,          \
                        rho=rho,        \
                        epsilon=epsilon,\
                        decay=decay)
    print ("Using RMSProp as the optimizer ...")
  elif optimizer_val.lower() == 'ada' \
       or \
       optimizer_val.lower() == 'adagrad':
    optimizer = Adagrad(lr=lr,           \
                        epsilon=epsilon, \
                        decay=decay)
    print ("Using Adagrad as the optimizer ...")
  elif optimizer_val.lower() == 'adelta' \
       or \
       optimizer_val.lower() == 'adadelta':
    optimizer = Adadelta(lr=lr,           \
                         rho=rho,         \
                         epsilon=epsilon, \
                         decay=decay)
    print ("Using Adadelta as the optimizer ...")
  elif optimizer_val.lower() == 'adam':
    optimizer = Adam(lr=lr,           \
                     beta_1=beta_1,   \
                     beta_2=beta_2,    \
                     epsilon=epsilon, \
                     decay=decay,     \
                     amsgrad=False)
    print ("Using Adam as the optimizer ...")
    print ("Optimizer parameters (recommended default): ")
    print ("\n lr={} (0.001),     \
            \n beta_1={} (0.9),   \
            \n beta_2={} (0.999), \
            \n epsilon={} (1e-08), \
            \n decay={} (0.0)".format(lr, 
                                      beta_1, 
                                      beta_2, 
                                      epsilon, 
                                      decay))
  elif optimizer_val.lower() == 'amsgrad':
    optimizer = Adam(lr=lr,           \
                     beta_1=beta_1,   \
                     beta_2=beta_2,    \
                     epsilon=epsilon, \
                     decay=decay,     \
                     amsgrad=True)
    print ("Using AmsGrad variant of Adam as the optimizer ...")
    print ("Optimizer parameters (recommended default): ")
    print ("\n lr={} (0.001),     \
            \n beta_1={} (0.9),   \
            \n beta_2={} (0.999), \
            \n epsilon={} (1e-08), \
            \n decay={} (0.0)".format(lr, 
                                      beta_1, 
                                      beta_2, 
                                      epsilon, 
                                      decay))
  elif optimizer_val.lower() == 'adamax':  
    optimizer = Adamax(lr=lr,           \
                       beta_1=beta_1,   \
                       beta_2=beta_2,    \
                       epsilon=epsilon, \
                       decay=decay)
    print ("Using Adamax variant of Adam as the optimizer ...")
    print ("Optimizer parameters (recommended default): ")
    print ("\n lr={} (0.002),     \
            \n beta_1={} (0.9),   \
            \n beta_2={} (0.999), \
            \n epsilon={} (1e-08), \
            \n schedule_decay={} (0.0)".format(lr, 
                                               beta_1, 
                                               beta_2, 
                                               epsilon, 
                                               decay))
  elif optimizer_val.lower() == 'nadam':  
    optimizer = Nadam(lr=lr,            \
                      beta_1=beta_1,    \
                      beta_2=beta_2,     \
                      epsilon=epsilon,  \
                      schedule_decay=decay)
    print ("Using Nesterov Adam optimizer ...\
           \n decay arguments is passed on to schedule_decay variable ...")
    print ("Optimizer parameters (recommended default): ")
    print ("\n lr={} (0.002),     \
            \n beta_1={} (0.9),   \
            \n beta_2={} (0.999), \
            \n epsilon={} (1e-08), \
            \n schedule_decay={} (0.004)".format(lr, 
                                                 beta_1, 
                                                 beta_2, 
                                                 epsilon, 
                                                 decay))
  else:
      optimizer = DEFAULT_OPTIMIZER
      print ("Using stochastic gradient descent with Nesterov momentum ('nsgd') as the default optimizer ...")
      print ("Options for optimizer are: 'sgd',        \
                                         \n'nsgd',     \
                                         \n'rmsprop',  \
                                         \n'adagrad',  \
                                         \n'adadelta', \
                                         \n'adam',     \
                                         \n'nadam',    \
                                         \n'amsgrad',  \
                                         \n'adamax' ...")
  return optimizer

def process_model(args, 
                  model, 
                  base_model, 
                  optimizer, 
                  loss, 
                  checkpointer_savepath):
  load_weights_ = args.load_weights[0]
  fine_tune_model = args.fine_tune[0]
  load_checkpoint = args.load_checkpoint[0]
   
  if load_weights_ == True:     
      try:
          with open(args.config_file[0]) as json_file:
              model_json = json_file.read()
          model = model_from_json(model_json)
      except:
          model = model
      try:
          model.load_weights(args.weights_file[0])
          print ("\nLoaded model weights from: " + str(args.weights_file[0]))
      except:
          print ("\nError loading model weights ...")
          print ("Tabula rasa ...")
          print ("Loaded default model weights ...")
  elif load_checkpoint == True and os.path.exists(checkpointer_savepath):     
      try:
          model = load_model(checkpointer_savepath)
          print ("\nLoaded model from checkpoint: " + str(checkpointer_savepath))
      except:
          if os.path.exists(args.saved_chkpnt[0]):
            model = load_model(args.saved_chkpnt[0])
            print ('\nLoaded saved checkpoint file ...')
          else:
            print ("\nError loading model checkpoint ...")
            print ("Tabula rasa ...")
            print ("Loaded default model weights ...")
  else:
      model = model
      print ("\nTabula rasa ...")
      print ("Loaded default model weights ...")
 
  try:
      NB_FROZEN_LAYERS = args.frozen_layers[0]
  except:
      NB_FROZEN_LAYERS = DEFAULT_NB_LAYERS_TO_FREEZE
      
  if fine_tune_model == True:
      print ("\nFine tuning Inception architecture ...")
      print ("Frozen layers: " + str(NB_FROZEN_LAYERS))
      model = finetune_model(model, base_model, optimizer, loss, NB_FROZEN_LAYERS)
  else:
      print ("\nTransfer learning using Inception architecture ...")
      model = transferlearn_model(model, base_model, optimizer, loss)
      
  return model

def process_images(args):  
  train_aug = args.train_aug[0] 
  test_aug = args.test_aug[0] 
   
  if str((args.base_model[0]).lower()) == 'inceptionv4' or  \
     str((args.base_model[0]).lower()) == 'inception_v4' or \
     str((args.base_model[0]).lower()) == 'inception_resnet':
      preprocess_input = preprocess_input_inceptionv4
  else:
      preprocess_input = preprocess_input_inceptionv3
  
  if train_aug==True:
    try:
        train_rotation_range = args.train_rot[0]
        train_width_shift_range = args.train_w_shift[0]
        train_height_shift_range = args.train_ht_shift[0]
        train_shear_range = args.train_shear[0]
        train_zoom_range = args.train_zoom[0]
        train_vertical_flip = args.train_vflip[0]
        train_horizontal_flip = args.train_hflip[0]
    except:
        train_rotation_range = 30
        train_width_shift_range = 0.2
        train_height_shift_range = 0.2
        train_shear_range = 0.2
        train_zoom_range = 0.2
        train_vertical_flip = True
        train_horizontal_flip = True
        print ("\nFailed to load custom training image augmentation parameters ...")
        print ("Loaded pre-set defaults ...")
        print ("To switch off image augmentation during training, set --train_augmentation flag to False")
        
    train_datagen =  ImageDataGenerator(preprocessing_function=preprocess_input,
                                        rotation_range=train_rotation_range,
                                        width_shift_range=train_width_shift_range,
                                        height_shift_range=train_height_shift_range,
                                        shear_range=train_shear_range,
                                        zoom_range=train_zoom_range,
                                        vertical_flip=train_vertical_flip,                                  
                                        horizontal_flip=train_horizontal_flip)
    print ("\nCreated image augmentation pipeline for training images ...")     
    print ("Image augmentation parameters for training images: \
          \n image rotation range = {},\
          \n width shift range = {},\
          \n height shift range = {}, \
          \n shear range = {} ,\
          \n zoom range = {}, \
          \n enable vertical flip = {}, \
          \n enable horizontal flip = {}".format(train_rotation_range,
                                                   train_width_shift_range,
                                                   train_height_shift_range,
                                                   train_shear_range,
                                                   train_zoom_range,
                                                   train_vertical_flip,
                                                   train_horizontal_flip))
  else:
      train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
  
  if test_aug==True:
      try:
        test_rotation_range = args.test_rot[0]
        test_width_shift_range = args.test_w_shift[0]
        test_height_shift_range = args.test_ht_shift[0]
        test_shear_range = args.test_shear[0]
        test_zoom_range = args.test_zoom[0]
        test_vertical_flip = args.test_vflip[0]
        test_horizontal_flip = args.test_hflip[0]
      except:
        test_rotation_range = 30
        test_width_shift_range = 0.2
        test_height_shift_range = 0.2
        test_shear_range = 0.2
        test_zoom_range = 0.2
        test_vertical_flip = True
        test_horizontal_flip = True
        print ("\nFailed to load custom training image augmentation parameters ...")
        print ("Loaded pre-set defaults ...")
        print ("To switch off image augmentation during training, set --train_augmentation flag to False")
      test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                        rotation_range=test_rotation_range,
                                        width_shift_range=test_width_shift_range,
                                        height_shift_range=test_height_shift_range,
                                        shear_range=test_shear_range,
                                        zoom_range=test_zoom_range,
                                        vertical_flip=test_vertical_flip,
                                        horizontal_flip=test_horizontal_flip)
      print ("\nCreated image augmentation pipeline for training images ...")     
      print ("\nImage augmentation parameters for training images:")
      print( "\n image rotation range = {},\
              \n width shift range = {},\
              \n height shift range = {}, \
              \n shear range = {} ,\
              \n zoom range = {}, \
              \n enable vertical flip = {}, \
              \n enable horizontal flip = {}".format(test_rotation_range,
                                                     test_width_shift_range,
                                                     test_height_shift_range,
                                                     test_shear_range,
                                                     test_zoom_range,
                                                     test_vertical_flip,
                                                     test_horizontal_flip))
  else:
      test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

  return [train_datagen, test_datagen]

def gen_model(args, enable_dropout):
  if str((args.base_model[0]).lower()) == 'inceptionv4' or  \
     str((args.base_model[0]).lower()) == 'inception_v4' or \
     str((args.base_model[0]).lower()) == 'inception_resnet':
      base_model = InceptionResNetV2(weights='imagenet', \
                                     include_top=False)
      base_model_name = 'Inception version 4'
  else:
      base_model = InceptionV3(weights='imagenet', 
                               include_top=False)
      base_model_name = 'Inception version 3'
  print ('\nBase model: ' + str(base_model_name))
  nb_classes = len(glob.glob(args.train_dir[0] + "/*"))
  model = add_top_layer(args,
                        base_model, 
                        nb_classes)
  print ("New top layer added to: " + str(base_model_name))
  return [model, base_model]

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
    
  optimizer  = select_optimizer(args)
  loss = args.loss[0]
  checkpointer_savepath = os.path.join(args.output_dir[0]     +       
                                       '/checkpoint/Transfer_learn_' +       
                                       str(IM_WIDTH)  + '_'  + 
                                       str(IM_HEIGHT) + '_'  + '.h5')
  
  nb_train_samples = get_nb_files(args.train_dir[0])
  nb_classes = len(glob.glob(args.train_dir[0] + "/*"))
  
  print ("\nTotal number of training samples = " + str(nb_train_samples))
  print ("Number of training classes = " + str(nb_classes))
  
  nb_val_samples = get_nb_files(args.val_dir[0])
  nb_val_classes = len(glob.glob(args.val_dir[0] + "/*"))
  
  print ("\nTotal number of validation samples = " + str(nb_val_samples))
  print ("Number of validation classes = " + str(nb_val_classes))
  
  if nb_val_classes == nb_classes:
      print ("\nInitiating training session ...")
  else:
      print ("\nMismatched number of training and validation data classes ...")
      print ("Unequal number of sub-folders found between train and validation directories ...")
      print ("Each sub-folder in train and validation directroies are treated as a separate class ...")
      print ("Correct this mismatch and re-run ...")
      print ("\nNow exiting ...")
      sys.exit(1)
      
  nb_epoch = int(args.epoch[0])
  batch_size = int(args.batch[0])    
  
  [train_datagen, validation_datagen] = process_images(args)
  
  labels = generate_labels(args)
  
  train_dir = args.train_dir[0]
  val_dir = args.val_dir[0]
  
  if args.normalize[0] and os.path.exists(args.root_dir[0]):
      _, train_size, val_size = normalize(args, 
                                          labels, 
                                          move = False,
                                          sub_sample = args.sub_sample[0])
      train_dir = os.path.join(args.root_dir[0] + 
                               str ('/.tmp_train/'))
      val_dir = os.path.join(args.root_dir[0] + 
                             str ('/.tmp_validation/'))
      
  print ("\nGenerating training data: ... ")
  train_generator = train_datagen.flow_from_directory(train_dir,
                                                      target_size=(IM_WIDTH, IM_HEIGHT),
                                                      batch_size=batch_size,
                                                      class_mode='categorical')
  
  print ("\nGenerating validation data: ... ")
  validation_generator = validation_datagen.flow_from_directory(val_dir,
                                                          target_size=(IM_WIDTH, IM_HEIGHT),
                                                          batch_size=batch_size,
                                                          class_mode='categorical')
  
  enable_dropout = args.enable_dropout[0]
  [model, base_model] = gen_model(args, enable_dropout)
    
  model = process_model(args, 
                        model, 
                        base_model, 
                        optimizer, 
                        loss, 
                        checkpointer_savepath)
            
  print ("\nInitializing training with  class labels: " + 
         str(labels))
  
  model_summary_ = args.model_summary[0]
  
  if model_summary_ == True:
      print (model.summary())
  else:
      print ("\nSuccessfully loaded deep neural network classifier for training ...")
      print ("\nReady, Steady, Go ...")
      print ("\n")
        
  if not os.path.exists(os.path.join(args.output_dir[0] + '/checkpoint/')):
    os.makedirs(os.path.join(args.output_dir[0] + '/checkpoint/'))
    
  lr = args.learning_rate[0]
    
  earlystopper = EarlyStopping(patience=6, 
                               verbose=1)
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
  
  steps_pre_epoch = nb_train_samples//batch_size
  validation_steps = nb_val_samples//batch_size
  
  if args.normalize[0]:
      steps_pre_epoch = (train_size*len(labels))//batch_size
      validation_steps = (train_size*len(labels))//batch_size
      
  model_train = model.fit_generator(train_generator,
                                    epochs=nb_epoch,
                                    steps_per_epoch=steps_pre_epoch,
                                    validation_data=validation_generator,
                                    validation_steps=validation_steps,
                                    class_weight='auto', 
                                    callbacks=[earlystopper, 
                                               learning_rate_reduction, 
                                               checkpointer])
  
  if args.fine_tune[0] == True:
      save_model(args, "_ft_", model)
      generate_plot(args, "_ft_", model_train)
  else:
      save_model(args, "_tl_", model)
      generate_plot(args, "_tl_", model_train)
      
def get_user_options():
    """
        A function that uses argument parser to pass user options from command line.
        
        Example usage:
                args = get_user_options()
                if ((not os.path.exists(args.train_dir[0])) 
                or 
                (not os.path.exists(args.val_dir[0])) 
                or 
                (not os.path.exists(args.output_dir[0]))):
                    print("Specified directories do not exist ...")
                    sys.exit(1)
    """
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
    
    a.add_argument("--root_directory", 
                   help = "Specify the root folder for sub-sampling and normalization ...", 
                   dest = "root_dir", 
                   required = False, 
                   type=lambda x: is_valid_dir(a, x), 
                   default = ['./'],
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
    
    a.add_argument("--enable_dropout", 
                   help = "Specify if the dropout layer should be enabled during inference ...", 
                   dest = "enable_dropout", 
                   required=False, 
                   default=[True], 
                   nargs=1, 
                   type = string_to_bool)
    
    a.add_argument("--enable_attention", 
                   help = "Specify if the dropout layer should be enabled during inference ...", 
                   dest = "enable_attention", 
                   required=False, 
                   default=[True], 
                   nargs=1, 
                   type = string_to_bool)
    
    a.add_argument("--enable_multilayerDense", 
                   help = "Specify if the dropout layer should be enabled during inference ...", 
                   dest = "enable_multilayerDense", 
                   required=False, 
                   default=[False], 
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
    
    a.add_argument("--normalize", 
                   help = "Specify if a training and validation data should be normalized ...", 
                   dest = "normalize", 
                   required=False, 
                   default=[False], 
                   nargs=1, 
                   type = string_to_bool)
    
    a.add_argument("--sub_sample", 
                   help = "Specify if a training and validation data should be should be sub sampled ...", 
                   dest = "normalize", 
                   required=False, 
                   default=[False], 
                   nargs=1, 
                   type = string_to_bool)
    
    a.add_argument("--train_sub_sample", 
                   help = "Specify the sub sampling fraction for training data ...", 
                   dest = "train_sub_sample", 
                   required=False, 
                   default=[0.8], 
                   type = float,
                   nargs=1)
    
    a.add_argument("--validation_sub_sample", 
                   help = "Specify the sub sampling fraction for validation data ...", 
                   dest = "val_sub_sample", 
                   required=False, 
                   default=[0.8], 
                   type = float,
                   nargs=1)
    
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
    
    a.add_argument("--train_image_rotation", 
                   help = "Specify values for rotation range to be applied to training images during pre-processing ...", 
                   dest = "train_rot", 
                   required=False, 
                   default=[30], 
                   type = float,
                   nargs=1)
    
    a.add_argument("--train_image_width_shift", 
                   help = "Specify values for width shift range to be applied to training images during pre-processing ...", 
                   dest = "train_w_shift", 
                   required=False, 
                   default=[0.2], 
                   type = float,
                   nargs=1)
    
    a.add_argument("--train_image_height_shift", 
                   help = "Specify values for height shift range to be applied to training images during pre-processing ...", 
                   dest = "train_ht_shift", 
                   required=False, 
                   default=[0.2], 
                   type = float,
                   nargs=1)
    
    a.add_argument("--train_image_shear", 
                   help = "Specify values for shear transformation range to be applied to training images during pre-processing ...", 
                   dest = "train_shear", 
                   required=False, 
                   default=[0.2], 
                   type = float,
                   nargs=1)
    
    a.add_argument("--train_image_zoom", 
                   help = "Specify values for zooming transformation range to be applied to training images during pre-processing ...", 
                   dest = "train_zoom", 
                   required=False, 
                   default=[0.2], 
                   type = float,
                   nargs=1)
    
    a.add_argument("--train_image_vertical_flip", 
                   help = "Specify if training image should be randomly flipped vertical during pre-processing ...", 
                   dest = "train_vflip", 
                   required=False, 
                   default=[False], 
                   type = string_to_bool,
                   nargs=1)
    
    a.add_argument("--train_image_horizontal_flip", 
                   help = "Specify if training image should be randomly flipped horizontal during pre-processing ...", 
                   dest = "train_hflip", 
                   required=False, 
                   default=[False], 
                   type = string_to_bool,
                   nargs=1)

    a.add_argument("--test_image_rotation", 
                   help = "Specify values for rotation range to be applied to training images during pre-processing ...", 
                   dest = "test_rot", 
                   required=False, 
                   default=[30], 
                   type = float,
                   nargs=1)
    
    a.add_argument("--test_image_width_shift", 
                   help = "Specify values for width shift range to be applied to training images during pre-processing ...", 
                   dest = "test_w_shift", 
                   required=False, 
                   default=[0.2], 
                   type = float,
                   nargs=1)
    
    a.add_argument("--test_image_height_shift", 
                   help = "Specify values for height shift range to be applied to training images during pre-processing ...", 
                   dest = "test_ht_shift", 
                   required=False, 
                   default=[0.2], 
                   type = float,
                   nargs=1)
    
    a.add_argument("--test_image_shear", 
                   help = "Specify values for shear transformation range to be applied to training images during pre-processing ...", 
                   dest = "test_shear", 
                   required=False, 
                   default=[0.2], 
                   type = float,
                   nargs=1)
    
    a.add_argument("--test_image_zoom", 
                   help = "Specify values for zooming transformation range to be applied to training images during pre-processing ...", 
                   dest = "test_zoom", 
                   required=False, 
                   default=[0.2], 
                   type = float,
                   nargs=1)
    
    a.add_argument("--test_image_vertical_flip", 
                   help = "Specify if training image should be randomly flipped vertical during pre-processing ...", 
                   dest = "test_vflip", 
                   required=False, 
                   default=[False], 
                   type = string_to_bool,
                   nargs=1)
    
    a.add_argument("--test_image_horizontal_flip", 
                   help = "Specify if training image should be randomly flipped horizontal during pre-processing ...", 
                   dest = "test_hflip", 
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
                   help = "Specify activation function.\
                           \nAvailable activation functions are: 'hard_sigmoid', \
                                                                 'elu',          \
                                                                 'linear',       \
                                                                 'relu',         \
                                                                 'selu',         \
                                                                 'sigmoid',      \
                                                                 'softmax',      \
                                                                 'softplus',     \
                                                                 'sofsign',      \
                                                                 'tanh' ...", 
                   dest = "activation", 
                   required=False, 
                   default=['relu'], 
                   type = activation_val,
                   nargs=1)
    
    a.add_argument("--loss", 
                   help = "Specify loss function.\
                           \nAvailable loss functions are:  'mean_squared_error', \
                                                            'mean_absolute_error' \
                                                            'mean_absolute_percentage_error' \
                                                            'mean_squared_logarithmic_error' \
                                                            'squared_hinge' \
                                                            'hinge',          \
                                                            'categorical_hinge',       \
                                                            'logcosh',         \
                                                            'categorical_crossentropy',         \
                                                            'sparse_categorical_crossentropy',      \
                                                            'binary_crossentropy',      \
                                                            'kullback_leibler_divergence',     \
                                                            'poisson',      \
                                                            'cosine_proximity' ...", 
                   dest = "loss", 
                   required=False, 
                   default=['categorical_crossentropy'], 
                   type = loss_val,
                   nargs=1)
    
    a.add_argument("--learning_rate", 
                   help = "Specify values for learning rate ...", 
                   dest = "learning_rate", 
                   required=False, 
                   default=[1e-07], 
                   type = float,
                   nargs=1)
    
    a.add_argument("--rho", 
                   help = "Specify values for rho\
                           \n Applied to RMSprop and Adadelta ...", 
                   dest = "rho", 
                   required=False, 
                   default=[0.9], 
                   type = float,
                   nargs=1)
 
    a.add_argument("--beta_1", 
                   help = "Specify values for beta_1\
                           \n Applied to Adam, AmsGrad, Adadelta and Nadam ...", 
                   dest = "beta_1", 
                   required=False, 
                   default=[0.9], 
                   type = float,
                   nargs=1)    
    
    a.add_argument("--beta_2", 
                   help = "Specify values for beta_2\
                           \n Applied to Adam, AmsGrad, Adadelta and Nadam ...", 
                   dest = "beta_2", 
                   required=False, 
                   default=[0.999], 
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
                           \nOptions for optimizer are:  'sgd',     \
                                                         'nsgd',    \
                                                         'rmsprop', \
                                                         'adagrad', \
                                                         'adadelta',\
                                                         'adam',    \
                                                         'nadam',   \
                                                         'amsgrad', \
                                                         'adamax' ...", 
                   dest = "optimizer_val", 
                   required=False, 
                   default=['adam'], 
                   nargs=1)
    
    a.add_argument("--base_model", 
                   help = "Specify the type of base model classifier to build the neural network. \
                           \nOptions are: Inception_v4 or Inception_v3 ...", 
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
