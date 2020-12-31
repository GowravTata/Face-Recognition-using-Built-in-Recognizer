# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 15:47:50 2020

@author: Gowrav Tata
"""

import cv2 as cv
import numpy as np

haar_cascade = cv.CascadeClassifier('haar_face.xml')

import os
os.chdir(r'C:\Users\hp\Data\Open CV')
features = np.load('features.npy',allow_pickle=True)
labels = np.load('labels.npy',allow_pickle=True)

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img = cv.imread(r'E:\ley.jpg')

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# Detect the faces in the image    
faces_rect = haar_cascade.detectMultiScale(gray,1.1,4)
for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h,x:x+h]
    
    label,confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with a confidence of {confidence}')
    cv.putText(img,str(people[label]),(20,20),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),2)
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)    
    
cv.imshow('Deteced Face',img)
    
cv.waitKey(0)