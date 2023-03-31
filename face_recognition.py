import numpy as np
import cv2 as cv

# Creating the haar cascade classifier
haar_cascade = cv.CascadeClassifier('haar_frontalface_default.xml')

# Reading names of persons trained
people =[]
with open('persons_trained.txt', 'r') as f:
    for person in f:
        people.append(person)

# Reading features and labels arrays
# features = np.load('features.npy', allow_pickle=True)
# labels = np.load('labels.npy')

# Reading the recognizer
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

# Reading and showing image to test
imagePath = r"E:\Amado\Career\Python\Course\Build a Python Facial Recognition App with Tensorflow and Kivy\data\negative\Agbani_Darego_0001.jpg"
img = cv.imread(imagePath)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person', gray)

# Detect the face in the image
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)
for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+h]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with a confidence of {confidence}')

    cv.putText(img=img, text=str(people[label]), org=(20,20),
                fontFace=cv.FONT_HERSHEY_COMPLEX, fontScale=1.0,
                color=(0, 255, 0), thickness=2)
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Detected Face', img)

cv.waitKey(0)