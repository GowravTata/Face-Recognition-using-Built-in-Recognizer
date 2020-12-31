# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 09:18:29 2020

@author: Gowrav Tata
"""

import cv2 as cv
import numpy as np
import os

Stars = ['Emma Roberts','Keira Knightley','Rosie Huntington','Scarlett Johansson','Shailene Woodley']
DIR = r'E:\Train'
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# Function to loop every folder in base folder
#Features are the arrays of the image
features = []
labels = []

def loop_data():
    for i in Stars:
        path = os.path.join(DIR, i)
        label = Stars.index(i)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            faces_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)
            
            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

loop_data()
# Convert image to numpy arrays

features = np.array(features,dtype='object')
labels = np.array(labels)
print(f'Length of the features is = {len(features)}')
print(f'Length of the labels is = {len(labels)}')

np.save('features.npy',features)
np.save('labels.npy',labels)

# Instantiate the OpenCV face recognizer

face_recognize = cv.face.LBPHFaceRecognizer_create()
face_recognize.train(features,labels)

face_recognize.save('face_trained.yml')
# Train the recognizer on features and label list'''


            
            