import numpy as np
import cv2
import pickle
import time

cap = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./recognizers/face-trainner.yml")

labels = {"person_name": 1}
with open("pickles/face-labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}

while(True):
    # Capture frame-by-frame
    frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5)
    for(x,y,w,h) in faces:
        roiGray = gray[y:y+h, x:x+w]
        roiColor = gray[y:y+h, x:x+w]
        id_, conf = recognizer.predict(roiGray)
        if conf >= 45 and conf <= 85:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 1
            cv2.putText(frame, name, (x,y-8), font, 1, color, stroke, cv2.LINE_AA)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
