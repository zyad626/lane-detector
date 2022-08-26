import cv2
import numpy as np

cap = cv2.VideoCapture("Traffic.mp4")
carcas = cv2.CascadeClassifier("cars.xml")

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = carcas.detectMultiScale(gray, 1.1, 1)
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.imshow('Output', frame )
    key=cv2.waitKey(1)
    #press escape key to turn off
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows() 