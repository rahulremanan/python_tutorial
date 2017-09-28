#!/usr/bin/python3.5
import face_recognition
import cv2
import os
from os.path import basename
import skvideo.io
import glob
import numpy as np
#import subprocess

source = '/home/info/Drag_Me_Down_LowRes.mp4'
try:
    video_capture = cv2.VideoCapture(source)
except:
    video_capture =  skvideo.io.vread(source)

reference_image_path = "/home/info/ref_img/"
file_list = glob.glob(reference_image_path + '/*.jpg')
save_path = "/home/info/proc_vid.avi"
# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
mf = 0.5

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video_writer = cv2.VideoWriter(save_path, fourcc, 30.0, (853,480), True)

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

if int(major_ver)  < 3 :
    fps = video_capture.get(cv2.cv.CV_CAP_PROP_FPS)
    print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
else :
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
     
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()    
    # Resize frame of video to 1/4 size for faster face recognition processing
    #small_frame = cv2.resize(frame, (0, 0), fx=mf, fy=mf)
    small_frame = frame
    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            for file_path in file_list:
                reference_image = face_recognition.load_image_file(file_path)
                reference_face_encoding = face_recognition.face_encodings(reference_image)[0]
                name_ID = (os.path.splitext(basename(file_path))[0])
                name_ID = name_ID.replace("_", " ")
            # See if the face is a match for the known face(s)
                match = face_recognition.compare_faces([reference_face_encoding], face_encoding)
                name = "Unknown"

                if match[0]:
                    name = name_ID

                face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= int(1/mf)
        right *= int(1/mf)
        bottom *= int(1/mf)
        left *= int(1/mf)
        # Draw a box around the face
        #cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # Draw an ellipse around the face
        ex = left
        ey = top
        ew = int(abs(right - ex))
        eh = int(abs(bottom - ey))
        p1 = int(ew/2 + ex)
        p2 = int(eh/2 + ey)
        h1 = int(ew/2)
        h2 = int(eh/2)
        cv2.ellipse(frame, (p1, p2), (h1,h2), 0,0,360, (0,255,0), 2)
        # Draw a label with a name below the face
        cv2.rectangle(frame, (p1 - 100, bottom - 2), (p1 + 100, bottom + 33), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (p1  - 94, bottom + 23 ), font, 0.75, (255, 255, 255), 1)
        
        if video_writer is None:
            (height, width)= frame.shape[:2]
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

# Release handle to read the video file or webcam
#subprocess.call(["ffmpeg", "-y", "-i", "a.wav",  "-r", "30", "-i", "v.h264",  "-filter:a", "aresample=async=1", "-c:a", "flac", "-c:v", "copy", "av.mkv"])
video_capture.release()
video_writer.release()