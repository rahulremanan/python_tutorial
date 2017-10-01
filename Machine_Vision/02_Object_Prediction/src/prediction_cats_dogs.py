#!/usr/bin/python3.5
import sys
import argparse
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
import json
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
from keras.models import model_from_json
from keras.optimizers import SGD, RMSprop, Adagrad

sgd = SGD(lr=1e-7, decay=0.5, momentum=1, nesterov=True)
rms = RMSprop(lr=1e-7, rho=0.9, epsilon=1e-08, decay=0.0)
ada = Adagrad(lr=1e-7, epsilon=1e-08, decay=0.0)
optimizer = sgd

target_size = (229, 229)                                                        # Fixed size for InceptionV3 architecture

def generate_timestamp():
    timestring = time.strftime("%Y_%m_%d-%H_%M_%S")
    print ("Time stamp generated: "+timestring)
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

def compile_model(model):
    model.compile(optimizer=optimizer, 
                loss='categorical_crossentropy', metrics=['accuracy'])
    
def load_prediction_model(args):
    try:
        print ("Loaded prediction model configuration from: " + 
               str(args.config_file[0]))
        with open(args.config_file[0]) as json_file:
              model_json = json_file.read()
        model = model_from_json(model_json)
    except:
          print ("Please specify a model configuration file ...")
          sys.exit(1)
    try:
          model.load_weights(args.weights_file[0])
          print ("Loaded model weights from: " + 
                 str(args.weights_file[0]))
    except:
          print ("Error loading model weights ...")
          sys.exit(1)
    try:
        print (args.labels_file[0])
        with open(args.labels_file[0]) as json_file:
            labels = json.load(json_file)
        print ("Loaded labels from: " + str(args.labels_file[0]))
    except:
        print ("No labels loaded ...")
        sys.exit(1)
    return model, labels

def predict(model, img, target_size):
  print ("Running prediction model on the image file ...")
  if img.size != target_size:
    img = img.resize(target_size)

  _x_ = image.img_to_array(img)
  _x_ = np.expand_dims(_x_, axis=0)
  _x_ = preprocess_input(_x_)
  preds = model.predict(_x_)
  probabilities = model.predict(_x_, batch_size=1).flatten()
  prediction = labels[np.argmax(probabilities)]
  return preds[0], prediction

def plot_preds(image, preds, labels):
  output_loc = args.output_dir[0]
  output_file_preds = os.path.join(output_loc+"//preds_out_"+timestr+".png")
  fig = plt.figure()
  plt.axis('on')
  labels = labels
  plt.barh([0, 1], preds, alpha=0.5)
  plt.yticks([0, 1], labels)
  plt.xlabel('Probability')
  plt.xlim(0,1.01)
  plt.tight_layout()
  fig.savefig(output_file_preds, dpi=fig.dpi)
  
def get_user_options():
    a = argparse.ArgumentParser()
    
    a.add_argument("--image",
                   dest="image",
                   help="path to image",
                   required = False,
                   type=lambda x: is_valid_file(a, x),
                   nargs=1)
    
    a.add_argument("--image_url", help="url to image")
    
    a.add_argument("--weights_file", 
                   help = "Specify pre-trained model weights for training ...", 
                   dest = "weights_file", 
                   required=False,
                   type=lambda x: is_valid_file(a, x),
                   nargs=1)
    
    a.add_argument("--config_file", 
                   help = "Specify pre-trained model configuration file ...", 
                   dest = "config_file",  
                   required=False,
                   type=lambda x: is_valid_file(a, x),
                   nargs=1)
    
    a.add_argument("--labels_file", 
                   help = "Specify class labels file ...", 
                   dest = "labels_file",  
                   required=False,
                   type=lambda x: is_valid_file(a, x),
                   nargs=1)
    
    a.add_argument("--output_directory", 
                   help = "Specify output folder ...", 
                   dest = "output_dir", 
                   required = True, 
                   type=lambda x: is_valid_dir(a, x),
                   nargs=1)
    
    args = a.parse_args()
    
    return args

if __name__=="__main__":
  args = get_user_options()
  
  if args.image is None and args.image_url is None:
    args.print_help()
    sys.exit(1)

  if ((not os.path.exists(args.config_file[0])) 
        or 
    (not os.path.exists(args.weights_file[0])) 
        or 
    (not os.path.exists(args.labels_file[0]))):
      print("Specified directories do not exist ...")
      sys.exit(1)
    
  print ("Loading neural network ...")

  try:
      model, labels = load_prediction_model(args)
      print ("Prediction model and class labels loaded ...")
  except:
      print ("Prediction model failed to load ...")
        
  if args.image is not None:
    img = Image.open(args.image[0])
    preds = predict(model, img, target_size)
    print (preds[1] + "\t" + "\t".join(map(lambda x: "%.2f" % x, preds[0])))
    print (str(preds[1]))
    plot_preds(img, preds[0], labels)

  if args.image_url is not None:
    response = requests.get(args.image_url[0])
    img = Image.open(BytesIO(response.content))
    preds = predict(model, img, target_size)
    print (preds[1] + "\t" + "\t".join(map(lambda x: "%.2f" % x, preds[0])))
    print (str(preds[1]))
    plot_preds(img, preds[0], labels)