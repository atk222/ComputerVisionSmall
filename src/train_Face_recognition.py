from image import Image_process
import cv2
import pickle
import numpy as np

face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_alt2.xml')
recon = cv2.face.LBPHFaceRecognizer_create()

class Train_face_recogntion:

    #initializing an array to which we will add the ROI's
    train_roi = []

    #label ids
    label_id = {}

    #labels
    labels = []

    def __init__(self):
        
        current_id = 0

        img_obj = Image_process()

        #iterating through the gray scale images and their labels
        for label, img in img_obj.label_image:

            if not label in self.label_id:

                self.label_id[label] = current_id

                current_id = current_id + 1

            id_ = self.label_id[label]

            #determining the face with face cascades
            faces = face_cascade.detectMultiScale(img, scaleFactor = 1.5, minNeighbors =5)

            #grabbing the roi and the ids
            for (x,y,w,h) in faces:

                roi = img[y:y+h, x:x+w]

                self.train_roi.append(roi)
                self.labels.append(id_)
            
            #saving for later use
            with open("labels.pickle",'wb') as f:

                pickle.dump(self.label_id,f)

    #training the recognizer
    def train(self):
        
        recon.train(self.train_roi, np.array(self.labels))
        recon.save("trained.yml")


