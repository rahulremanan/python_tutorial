#!/usr/bin/env python3
import cv2
import argparse
import sys
import os
import time
import numpy as np
import scipy.misc
import skvideo.io
import json
import dataset
import keras
from keras.models import model_from_json
from keras.optimizers import SGD, RMSprop, Adagrad

sgd = SGD(lr=1e-7, decay=0.5, momentum=1, nesterov=True)
rms = RMSprop(lr=1e-7, rho=0.9, epsilon=1e-08, decay=0.0)
ada = Adagrad(lr=1e-7, epsilon=1e-08, decay=0.0)
optimizer = sgd

n = 224

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
    
def string_to_bool(val):
    if val.lower() in ('yes', 'true', 't', 'y', '1', 'yeah'):
        return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0', 'none'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected ...')

def compile_model(model):
    model.compile(optimizer=optimizer, 
                loss='categorical_crossentropy', metrics=['accuracy'])
    
def load_prediction_model(args):
    try:
        with open(args.config_file[0]) as json_file:
              model_json = json_file.read()
        model = model_from_json(model_json)
    except:
          print ("Please specify a model configuration file ...")
          sys.exit(1)
    try:
          model.load_weights(args.weights_file[0])
          print ("Loaded model weights from: " + str(args.weights_file[0]))
    except:
          print ("Error loading model weights ...")
          sys.exit(1)
    try:
        with open(args.labels_file[0]) as json_file:
            labels = json.load(json_file)
        print ("Loaded labels from: " + str(args.labels_file[0]))
    except:
        print ("No labels loaded ...")
        sys.exit(1)
    return model, labels

def get_user_options():
    a = argparse.ArgumentParser()
    
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
    
    a.add_argument("--cascade_file", 
                   help = "Specify class labels file ...", 
                   dest = "cascade_file",  
                   required=False,
                   type=lambda x: is_valid_file(a, x),
                   nargs=1)
    
    a.add_argument("--video", 
                   help = "Specify video file ...", 
                   dest = "video_file",  
                   required=True,
                   type=lambda x: is_valid_file(a, x),
                   nargs=1)
       
    a.add_argument("--webcam", 
                   help = "Specify if local webcam should be the source ...", 
                   dest = "webcam", 
                   required=True, 
                   default=[False], 
                   nargs=1, 
                   type = string_to_bool)
    
    a.add_argument("--output_directory", 
                   help = "Specify output folder ...", 
                   dest = "output_dir", 
                   required = True, 
                   type=lambda x: is_valid_dir(a, x),
                   nargs=1)
    
    args = a.parse_args()
    
    return args
    
    
def gen_predict(model):
    try:
        compile_model(model)
        print ("Model successfully compiled ...")
    except:
        print ("Model failed to compile ...")

    print ("Compiling predictor function ...")                                          # to avoid the delay during video capture.
    _ = model.predict(np.zeros((1, n, n, 3), dtype=np.float32), batch_size=1)
    print ("Compilation completed ...")

def face_detect(model, labels, args):
    save_path = os.path.join(args.output_dir[0]+"//"+timestr+".avi")
    
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video_writer = cv2.VideoWriter(save_path, fourcc, 30.0, (853,480), True)

    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    gen_predict(model)
    cascade_filename = (args.cascade_file[0])
    assert os.path.isfile(cascade_filename), "Face detector model: haarcascade_frontalface_default.xml must be specified in the user arguments"
    faceCascade = cv2.CascadeClassifier(cascade_filename)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    video_source = args.video_file[0]
    web_cam = args.webcam[0]
    
    if web_cam == True:
        video_capture = cv2.VideoCapture(0)
    else:
        print ("Loading from: "+ str(args.video_file[0]))
        try:
            video_capture = cv2.VideoCapture(video_source)
        except:
            video_capture =  skvideo.io.vread(video_source)

    if int(major_ver)  < 3 :
        try:
            fps = video_capture.get(cv2.cv.CV_CAP_PROP_FPS)
            print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
        except:
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
            
    else :
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
    
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
                )

        frameOut = np.array(frame)
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            if w>100 and h>100:
                cv2.rectangle(frameOut, (x, y), (x+w, y+h), (0, 255, 0), 2)
                square = frame[max((y-h//2,0)):y+3*h//2, max((x-w//2,0)):x+3*w//2]
                square = scipy.misc.imresize(square.astype(np.float32), size=(n, n), interp='bilinear')
                square = np.expand_dims(square, axis=0)
                square = square.transpose((0, 3, 1, 2))
                square = dataset.preprocess_input(square)

                probabilities = model.predict(square, batch_size=1).flatten()
                prediction = labels[np.argmax(probabilities)]
                print (prediction + "\t" + "\t".join(map(lambda x: "%.2f" % x, probabilities)))
                cv2.putText(frameOut, prediction, (x, y-2), font, 1, (255,255,255), 2)

        if video_writer is None:
            (height, width)= frameOut.shape[:2]
            zeros = np.zeros((height, width), dtype="uint8")
            (B, G, R) = cv2.split(frame)
            R = cv2.merge([zeros, zeros, R])
            G = cv2.merge([zeros, G, zeros])
            B = cv2.merge([B, zeros, zeros])
            video_output = np.zeros((height * 2, width * 2, 3), dtype="uint8")
            video_output[0:height, 0:width] = frame
            video_output[0:height, width:width * 2] = R
            video_output[height:height * 2, width:width * 2] = G
            video_output[height:height * 2, 0:width] = B
 
            # write the output frame to file
            video_writer.write(video_output)

    video_capture.release()
    video_writer.release()

if __name__=="__main__":
    args = get_user_options()

    if ((not os.path.exists(args.config_file[0])) 
        or 
    (not os.path.exists(args.weights_file[0])) 
        or 
    (not os.path.exists(args.labels_file[0]))):
      print("Specified directories do not exist ...")
      sys.exit(1)
    
    print ("Loading neural network")
    try:
        model, labels = load_prediction_model(args)
        print ("Prediction model and class labels loaded ...")
    except:
        print ("Prediction model failed to load ...")
        
    face_detect(model, labels, args) 