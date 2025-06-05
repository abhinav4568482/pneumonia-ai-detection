import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense , Conv2D , MaxPooling2D , Dropout , BatchNormalization , Flatten , MaxPool2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2

model = Sequential()

model.add(Conv2D(32 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (256,256,3)))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

model.add(Conv2D(128 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(256 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

model.add(Flatten())

model.add(Dense(units = 128 , activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(units = 1 , activation = 'sigmoid'))

model.summary()

from tensorflow.keras.optimizers import  Adam

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics = ['accuracy'])

train_dir = 'dataset//train'
test_dir = 'dataset//test'
val_dir = 'dataset//val'

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(256,256),
        batch_size=32,
        class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(256,256),
        batch_size=32,
        class_mode='binary')

val_datagen = ImageDataGenerator(rescale=1./255)

val_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=(256,256),
        batch_size=32,
        class_mode='binary')


history = model.fit(train_generator,
                         steps_per_epoch = 163,
                         epochs = 50,
                         validation_data = val_generator,
                         validation_steps = 624)


test_accuracy = model.evaluate(test_generator,steps=624)
train_accuracy = model.evaluate(train_generator,steps=163)
print('The testing accuracy is :',test_accuracy[1]*100, '%')
print('The training accuracy is :',train_accuracy[1]*100, '%')


model.save('models//Pneumonia_Model.h5')