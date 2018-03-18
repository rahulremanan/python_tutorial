# -*- coding: utf-8 -*-
#!/usr/bin/python3.5
import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier('D:\\haarcascade_frontalface.xml')
eye_cascade = cv2.CascadeClassifier('D:\\haarcascade_eye.xml')
cap = cv2.VideoCapture(0)
scale_factor = 1.1
min_neighbors = 3
min_size_face = (50, 50)
min_size_eye = (30, 30)
flags = cv2.CASCADE_SCALE_IMAGE
while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor = scale_factor, minNeighbors = min_neighbors,
    minSize = min_size_face, flags = flags)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.2, 5, minSize = min_size_eye, flags=flags)        
        for (ex,ey,ew,eh) in eyes:
            #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            p1 = int(ew + ex)
            p2 = int(eh + ey)
            h1 = int(ew)
            h2 = int(eh)
            cv2.ellipse(roi_color, (p1, p2), (h1,h2), 0,0,360, (0,255,0), 2)
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:                                                                 # Press Esc to quit
        break
cap.release()
cv2.destroyAllWindows()