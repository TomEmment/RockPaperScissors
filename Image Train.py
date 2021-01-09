import pandas as pd
import numpy as np
from numpy import argmax
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display
import random
import warnings
import os
from PIL import Image
import pathlib
import csv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
import cv2
from PIL import Image
#from VideoCapture import Device

import time
import random

warnings.filterwarnings('ignore')


Names = ["Paper","Rock","Scissors"]
cwd = os.getcwd()

cam = cv2.VideoCapture(0)
ret, frame = cam.read()

def CropImages():
    picture = Image.open("Test.png")


    CutStarty = 9999
    CutStartx = 9999
    CutEndx = 0
    CutEndy = 0
    width, height = picture.size
    for x in range(0,width):
         for y in range(0,height):
            current_color = picture.getpixel( (x,y) )
            if current_color[0] >= 160 and current_color[1] >= 160 and current_color[2] >= 160:
                if x >= CutEndx:
                    CutEndx = x
                if y >= CutEndy:
                    CutEndy = y
                if y <= CutStarty:
                    CutStarty = y                        
                if x <= CutStartx:
                    CutStartx = x
    picture = picture.crop((CutStartx,CutStarty,CutEndx,CutEndy))
    picture = picture.resize((400,250))
    picture.save("Test.png","png")

def Test():
    image = tf.keras.preprocessing.image.load_img("Test.png")
    
    input_arr = keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) 
    predictions = model.predict(input_arr)
    if predictions[0][0] > predictions[0][1] and predictions[0][0] > predictions[0][2]:
        print("Paper")
        return 1
    if predictions[0][1] > predictions[0][0] and predictions[0][1] > predictions[0][2]:
        print("Rock")
        return 0
    if predictions[0][2] > predictions[0][1] and predictions[0][2] > predictions[0][0]:
        print("Scissors")
        return 2


                                
def GetInput():
    print("3")
    time.sleep(1)
    print("2")
    time.sleep(1)
    print("1")
    time.sleep(1)
    ret, frame = cam.read()
    print ("Taking Photo")
    img_name = "Test.png"
    cv2.imwrite(img_name, frame)
    print("Image Saved!")
    CropImages()


def Train():
    data_dir = pathlib.Path(f'img_data')
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
      data_dir,
      validation_split=0.2,
      subset="training",
      seed=123,
      image_size=(250,400),
      batch_size=24)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
      data_dir,
      validation_split=0.2,
      subset="validation",
      seed=123,
      image_size=(250,400),
      batch_size=24)

    model = tf.keras.Sequential([
        layers.experimental.preprocessing.Rescaling(1./127.5, offset=-1),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(3, activation='sigmoid')
    ])

    model.compile(
      optimizer='adam',
      loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'])

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5,
        #callbacks=tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=3)
    )

    model.save("Final.keras")




Choice = ""
Options = ["Rock","Paper","Scissors"]

#Train()

model = keras.models.load_model("Final.keras")

while Choice != "exit":
    Choice = input("Would you like to play a game of rock, paper scissors: ")
    if Choice == "yes":
        print("Present choice to camera in:")
        GetInput()
        RandomNumber = random.randint(0,2)
        print("I choose ", Options[RandomNumber])
        print("You chose: ")
        Number = Test()
        if Number == RandomNumber:
            print("Draw")
        elif (Number == 0 and RandomNumber == 2) or (Number == 1 and RandomNumber == 0) or (Number == 2 and RandomNumber == 1):
            print("You win!")
        else:
            print("I win!")
        
        


