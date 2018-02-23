#!/usr/bin/python3.5
import face_recognition
import cv2
import os
from os.path import basename
import skvideo.io
import glob
import sys
import subprocess

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

if int(major_ver)  < 3 :
    print ("This version of OpenCV is unsupported ...")
    print ("Please update OpenCV ...")
    sys.exit(1)

source = '/home/info/Drag_Me_Down.mp4'

try:
    video_capture = cv2.VideoCapture(source)
    print ("Imported video using Open-CV ...")
except:
    video_capture =  skvideo.io.vread(source)
    print ("Imported video using sci-kit video ...")
    
length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

#save_path = "/home/info/proc_vid.ogv"
save_path = "/home/info/proc_vid.mp4"

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
frame_number = 0
process_this_frame = True
inverse_scale_factor = 2

w, h = int(video_capture.get(3)),int(video_capture.get(4))

print ("Source image width: "+ str(w))
print ("Source image height: "+ str(h))

fps = video_capture.get(cv2.CAP_PROP_FPS)
print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

#fourcc = cv2.VideoWriter_fourcc(*'THEO')
#fourcc = cv2.VideoWriter_fourcc(*'MJPG')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(save_path,fourcc, fps, (w,h), True)

reference_image_path = "/home/info/ref_img/"
file_list = glob.glob(reference_image_path + '/*.jpg')

n_proc_frames = length
resize_img = False
verbose = True
   
while (video_capture.isOpened()):
    # Grab a single frame of video
    ret, frame = video_capture.read()
    
    frame_number += 1
       
    if resize_img ==True:
        # Resize frame of video to 1/4 size for faster face recognition processing
        isf = inverse_scale_factor
        small_frame = cv2.resize(frame, (0, 0), fx=(1/isf), fy=(1/isf))
    else:
        isf = 1
        small_frame = frame
    # Only process every other frame of video to save time
    if frame_number <=n_proc_frames:
        if ret ==True:
            if process_this_frame:
                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(small_frame)
                face_encodings = face_recognition.face_encodings(small_frame, face_locations)

                face_names = []
                for face_encoding in face_encodings:
                    for file_path in file_list:
                        reference_image = face_recognition.load_image_file(file_path)
                        try:
                            reference_face_encoding = face_recognition.face_encodings(reference_image)[0]
                            if verbose == True:
                                print ("Processed face encodings ...")
                            else:
                                pass
                        except:
                            if verbose == True:
                                print("Failed processing face encodings ...")
                            else:
                                pass
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
                top *= int(isf)
                right *= int(isf)
                bottom *= int(isf)
                left *= int(isf)
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
        
            try:
                video_writer.write(frame)
                print("Processed frame {} / {}".format(frame_number, length))
            except:
                print("Failed writing frame {} / {}".format(frame_number, length))
    else:
        print ("Processed "+ str(n_proc_frames) + " frames")
        break

# Release handle to read the video file or webcam
video_capture.release()
video_writer.release()

cmd = 'ffmpeg -i %s -i audio.wav -shortest -c:v copy -c:a aac -b:a 256k  %s' % (source, save_path)
subprocess.call(cmd, shell=True)
print('Muxing completed ...')
print('Saved output file to: %s')(save_path)