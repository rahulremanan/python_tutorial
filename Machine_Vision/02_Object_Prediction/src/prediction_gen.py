#!/usr/bin/python3.6
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
from keras.preprocessing.image import ImageDataGenerator

target_size = (299, 299)                                                        # Fixed size for InceptionV3 architecture

def generate_timestamp():
    timestring = time.strftime("%Y_%m_%d-%H_%M_%S")
    if verbose == True:
        print ("Time stamp generated: "+timestring)
    return timestring

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
        if args.verbose == True:
            print ("Loaded prediction model configuration from: " + \
               str(args.config_file[0]))
        with open(args.config_file[0]) as json_file:
              model_json = json_file.read()
        model = model_from_json(model_json)
    except:
        if args.verbose == True:
            print ("Please specify a model configuration file ...")
        sys.exit(1)
    try:
        model.load_weights(args.weights_file[0])
        if args.verbose == True:
            print ("Loaded model weights from: " + 
                 str(args.weights_file[0]))
    except:
        if args.verbose == True:
            print ("Error loading model weights ...")
        sys.exit(1)
    try:
        if args.verbose == True:
            print (args.labels_file[0])
        with open(args.labels_file[0]) as json_file:
            labels = json.load(json_file)
        if args.verbose == True:
            print ("Loaded labels from: " + str(args.labels_file[0]))
    except:
        if args.verbose == True:
            print ("No labels loaded ...")
        sys.exit(1)
    return model, labels

def predict(model, img, target_size, verbose):
    if verbose == True:
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

def predict_gen(model, preds_dir, target_size, verbose, batch_size=1):
    preds_datagen = ImageDataGenerator(rescale=1. / 255)
    preds_generator = preds_datagen.flow_from_directory(args.preds_dir[0],
    target_size=target_size,
    batch_size=batch_size,
    shuffle=False,
    class_mode='categorical')
    if verbose == True:
        print(preds_generator.filenames)
    preds = model.predict_generator(preds_generator, steps=len(preds_generator.filenames), max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)
    p = []
    for prediction in preds:
        probabilities = prediction.flatten()
        p.append(labels[np.argmax(probabilities)])       
    return preds, p

def plot_preds(preds, labels, timestr):
  output_loc = args.output_dir[0]
  output_file_preds = os.path.join(output_loc+"//preds_out_"+timestr+".png")
  fig = plt.figure()
  plt.axis('on')
  labels = labels
  plt.barh(list(range(0, len(labels))), preds, alpha=0.5)
  plt.yticks(list(range(0, len(labels))), labels)
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
    
    a.add_argument("--image_url", 
                   dest = "image_url",
                   help="url to image",
                   nargs=1,
                   required = False)
    
    a.add_argument("--verbose", 
                   dest = "verbose",
                   help="set verbose for detailed steps",
                   nargs=1,
                   required = False,
                   type= bool,
                   default = False)
    
    a.add_argument("--weights_file", 
                   help = "Specify pre-trained model weights for training ...", 
                   dest = "weights_file", 
                   required=True,
                   type=lambda x: is_valid_file(a, x),
                   nargs=1,
                   default = ['/weight.model'])
    
    a.add_argument("--config_file", 
                   help = "Specify pre-trained model configuration file ...", 
                   dest = "config_file",  
                   required=True,
                   type=lambda x: is_valid_file(a, x),
                   nargs=1,
                   default=['/model_config.json'])
    
    a.add_argument("--labels_file", 
                   help = "Specify class labels file ...", 
                   dest = "labels_file",  
                   required=True,
                   type=lambda x: is_valid_file(a, x),
                   nargs=1,
                   default = ['/labels.json'])
    
    a.add_argument("--output_directory", 
                   help = "Specify output folder ...", 
                   dest = "output_dir", 
                   required = True, 
                   type=lambda x: is_valid_dir(a, x),
                   nargs=1,
                   default = ['/'])
    
    a.add_argument("--prediction_directory", 
                   help = "Specify prediction folder ...", 
                   dest = "preds_dir", 
                   required = False, 
                   type=lambda x: is_valid_dir(a, x),
                   nargs=1)
    
    a.add_argument("--batch_size", 
                   help = "Specify prediction sample size ...", 
                   dest = "batch_size", 
                   required = True, 
                   type=int,
                   nargs=1,
                   default = [1])
    
    args = a.parse_args()
    
    return args

if __name__=="__main__":
    args = get_user_options()
    
    try:
        verbose = args.verbose
    except:
        versbose = False
  
    if args.image is None and args.image_url and args.preds_dir is None:
        args.print_help()
        sys.exit(1)

    if ((not os.path.exists(args.config_file[0])) 
        or 
    (not os.path.exists(args.weights_file[0])) 
        or 
    (not os.path.exists(args.labels_file[0]))):
        print("Specified directories do not exist ...")
        sys.exit(1)
    
    if verbose == True:
        print ("Loading neural network ...")

    try:
        model, labels = load_prediction_model(args)
        print ("Prediction model and class labels loaded ...")
    except:
        print ("Prediction model failed to load ...")
        
    if args.image is not None:
        img = Image.open(args.image[0])
        preds = predict(model, img, target_size, verbose)
        print (str(args.image[0]))
        print (preds[1] + "\t" + "\t".join(map(lambda x: "%.2f" % x, preds[0])))
        if verbose == True:
            print ("The image is most likely :" + str(preds[1]) + " breast tissue ")
        timestr = generate_timestamp()
        plot_preds(preds[0], labels, timestr)
    
    elif args.preds_dir is not None:
        preds_dir = args.preds_dir[0]
        batches = args.batch_size[0]
        preds = predict_gen(model, preds_dir, target_size, verbose, batches)
        print (preds)

    elif args.image_url is not None:
        response = requests.get(args.image_url[0])
        img = Image.open(BytesIO(response.content))
        preds = predict(model, img, target_size, verbose)
        print (preds[1] + "\t" + "\t".join(map(lambda x: "%.2f" % x, preds[0])))
        if verbose == True:
            print ("The image is most likely :" + str(preds[1]) + " breast tissue ")
        timestr = generate_timestamp()
        plot_preds(preds[0], labels, timestr)