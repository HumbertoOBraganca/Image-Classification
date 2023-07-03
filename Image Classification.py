import cv2
import numpy as np
import matplotlib.pyplot as plt

#Dependendo da data que tu baixar tu pode identificar casas, placas, carros, etc
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#cv2 voce chama o opencv, classifier detecta e 'classifica'

#img = cv2.imread('pessoas3.jpg')
#se tu colocar o valor de 0 ele usa o video da webcam
webcam = cv2.VideoCapture('video2.MP4')

while True:
    #le o frame atual
    successful_frame_read, frame = webcam.read()

    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    for [x, y, w, h] in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

    cv2.imshow('Isso aqui é um teste: ', frame)
    key = cv2.waitKey(1)

    if key == 80 or key == 112:
        break
webcam.release()

print('Codigo completo')
'''

grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#detecta o objeto, nesse caso sao rostos, mas depende de qual biblioteca xml tu importou e passa as coordenadas
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

for [x, y, w, h] in face_coordinates:
    cv2.rectangle(grayscaled_img, (x, y), (x+w, y+h), (0, 255, 0), 1)

print(face_coordinates, 'cordenadas')
cv2.imshow('Isso aqui é um teste: ', grayscaled_img)
cv2.waitKey()

print('Codigo completo')

'''