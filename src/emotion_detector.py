import pandas as pd
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


X = []
y = []

#reading in the data
df = pd.read_csv('fer2013.csv')
print(df.info())

#removing all entries where the emotion is disgust, and then re-doing the order of all the emotions
df_new = df[(df.emotion != 1)]
df_new['emotion'] = np.where(df_new['emotion']==2, 1, df_new['emotion'])
df_new['emotion'] = np.where(df_new['emotion']==3, 2, df_new['emotion'])
df_new['emotion'] = np.where(df_new['emotion']==4, 3, df_new['emotion'])
df_new['emotion'] = np.where(df_new['emotion']==5, 4, df_new['emotion']) 
print(df_new.info())

#matching the X to the images and y to the corresponding labels
def load_data():
    for index, row in df_new.iterrows():
        try:
            pixels=np.asarray(list(row['pixels'].split(' ')), dtype=np.uint8)
            img = pixels.reshape((48,48))
            X.append(img)
            y.append(row['emotion'])
        except Exception as e:
            pass
        
load_data()

#reshaping the photos
X = np.array(X, dtype='float32').reshape(-1, 48, 48, 1)
X=X/255.
y = np.asarray(y)

#splitting our test and training sets
(X_train, X_val, y_train, y_val) = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=42,
                                                    shuffle=True,
                                                    stratify=y)
#just checking to see if our data looks ok
print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)

#transforming our images and generating tensor image data using ImageDataGenerator()
aug_train = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

val_generator = ImageDataGenerator()

aug_train.fit(X_train)

val_generator.fit(X_val)

#building our model
def build_model(hp):
    model = Sequential()
    
    #first layer
    model.add(Conv2D(32, (3, 3), input_shape=X_train.shape[1:]))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    #second layer
    model.add(Conv2D(128, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    #third layer
    model.add(Conv2D(128, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    #fourth layer
    model.add(Conv2D(256, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    model.add(Flatten()) 

    model.add(Dense(6))
    model.add(Activation("softmax"))

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    
    return model

#initializing callbacks to make training the model easier + save our model for use in an h5 file
checkpoint = ModelCheckpoint('emotion_detector.h5',
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)

earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=20,
                          verbose=1,
                          restore_best_weights=True
                          )

lr_reduction = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=3,
                              verbose=1,
                              min_delta=0.0001)


model.compile(loss='sparse_categorical_crossentropy',
              optimizer = Adam(learning_rate=0.001),
              metrics=['accuracy'])

history=model.fit(aug_train.flow(X_train, y_train, batch_size=128),
                  validation_data=val_generator.flow(X_val, y_val, batch_size=128),
                  steps_per_epoch=len(y_train)//batch_size,
                  epochs=100,
                  callbacks=[earlystop,checkpoint,lr_reduction],
                  validation_steps=len(y_val)//batch_size)

#ended with 68 epochs, train acc = ~68% and val acc = ~64%
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)

plt.plot(history.history['loss'], color="green")
plt.plot(history.history['val_loss'], color="red")
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.figure(figsize=(10,5))
plt.subplot(1, 2, 2)
plt.ylabel('accuracy')
plt.title('model accuracy')
plt.plot(history.history['accuracy'], color="green")
plt.plot(history.history['val_accuracy'], color="red")
plt.legend(['train accuracy', 'test accuracy'], loc='lower right')
plt.show()