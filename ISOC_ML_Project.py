#Convolutional Neural Network

#Installing the libraries
import numpy as np
import pandas as pd

#Part 1-Building CNN

#Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Intializing the CNN
seq=Sequential()

#Step 1-Convolution
seq.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))

#Step 2-Pooling
seq.add(MaxPooling2D(pool_size=(2, 2)))

#Step 3-Flattening
seq.add(Flatten())

#Step 4-Full Connection
seq.add(Dense(units=128, activation='relu'))
seq.add(Dense(units=9, activation='softmax'))

#Compiling the CNN
seq.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Part 2-Fitting the CNN
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

seq.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)