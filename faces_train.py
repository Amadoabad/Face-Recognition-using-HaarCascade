import os
import cv2 as cv
import numpy as np
from scipy.misc import face

# Path to persons folder
DIR = os.path.join('Persons')

# Getting the names of all people we have records for
people = []
for person in os.listdir(DIR):
    people.append(person)

# Creating the haar cascade classifier
haar_cascade = cv.CascadeClassifier('haar_frontalface_default.xml')

features = []
labels = []

def CreateTrainSet():
    # Getting all the images for every person
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            
            # Reading images and turning it into gray images    
            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            
            # Detecting the face of the image
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, 
            minNeighbors=4)
        
            # Getting the face into features with the label
            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

CreateTrainSet()

# Instantiate the face recognizer
face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Turning lists into arrays
features = np.array(features, dtype='object')
labels = np.array(labels)

# Train the Recognizer on the features list and the labels list
face_recognizer.train(features, labels)

# Saving the the Recognizer
face_recognizer.save('face_trained.yml')
print('face_recognizer saved!')

# Saving names of persons trained
with open('persons_trained.txt', 'w') as f:
    for person in people:
        f.writelines(person)

# Saving features and labels arrays
np.save('features.npy', features)
print('features saved!')
np.save('labels.npy', labels)
print('lables saved!')

