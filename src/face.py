import sys
import cv2
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from keras.models import load_model
from keras.preprocessing.image import img_to_array

sys.path.append(".")

class_labels = ['angry', 'scared', 'happy', 'sad', 'surprised', 'neutral', 'neutral']
classifier = load_model('src/emotion_detector.h5')
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')
recon = cv2.face.LBPHFaceRecognizer_create()
recon.read("trained.yml")

labels = {}

#getting labels
with open("labels.pickle",'rb') as f:
    tmp = pickle.load(f)
    labels = {v:k for k,v in tmp.items()}

cap = cv2.VideoCapture(0)

print(labels)
while(True):
    #reading frame by frame
    ret, frame = cap.read()             

    #converting to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detecting the face based on cascade
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors =5)
    
    #getting the face specifically
    for(x, y, w, h) in faces:
        
        #roi gray 
        roi_gray = gray[y:y+h, x:x+w]
    
        #roi color 
        roi_color = frame[y:y+h, x:x+w]
        
        id_, conf = recon.predict(roi_gray)
        
        #recognizing the person
        if (conf >=45 and conf <= 85):
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 100, 50)
            stroke = 2
            cv2.putText(frame, name, (x,y-60), font ,1, color, stroke, cv2.LINE_AA)
            #for recognizing the emotion
            if np.sum([roi_gray])!=0:
                roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)

                prediction = classifier.predict(roi)[0]
                label=class_labels[prediction.argmax()]
                label_position = (x+50,y-30)
                cv2.putText(frame,"is",label_position,font,1,color, stroke, cv2.LINE_AA)
                cv2.putText(frame,label,(x+13,y),font,1,color, stroke, cv2.LINE_AA)
            else:
                cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 100, 50),2)     
        
        #bgr scale
        color = (255, 0, 0)

        #how wide the box will be
        stroke = 2

        #draw on the frame
        #cv2.rectangle(frame, (x,y), (x+w, y+h), color, stroke)

    cv2.imshow('frame', frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows