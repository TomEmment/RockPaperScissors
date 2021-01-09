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
import sounddevice as sd
import soundfile as sf
import scipy.io.wavfile as wav
import time
import random

warnings.filterwarnings('ignore')


Names = ["Paper","Rock","Scissors"]
cwd = os.getcwd()



def ConvertDataAll():
    genres = 'Rock Paper Scissors'.split()
    for g in genres:
        pathlib.Path(f'img_data/{g}').mkdir(parents=True, exist_ok=True)
        Check = os.path.join(cwd, g)
        for filename in os.listdir(Check):
            if filename.endswith(".wav"):
                Audioname = os.path.join(Check, filename)
                y, sr = librosa.load(Audioname, mono=True, duration=2)
                print(y.shape)
                plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, sides='default', mode='default', scale='dB');
                plt.axis('off');
                plt.savefig(f'img_data/{g}/{filename[:-3].replace(".", "")}.png')
                plt.clf()

def ConvertData1():
    y, sr = librosa.load("Test.wav", mono=True, duration=2)
    plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, sides='default', mode='default', scale='dB');
    plt.axis('off');
    plt.savefig("Test.png")
    plt.clf()

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
    print ("Recording Audio")
    #time.sleep(0.3)
    myrecording = sd.rec(2 * 44100, samplerate=44100, channels=2,dtype='float64')
    sd.wait()
    Name = "Test.wav"
    sf.write(Name, myrecording, 44100)
    sd.wait()


def Train():
    data_dir = pathlib.Path(f'img_data')
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
      data_dir,
      validation_split=0.2,
      subset="training",
      seed=123,
      image_size=(480, 640),
      batch_size=32)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
      data_dir,
      validation_split=0.2,
      subset="validation",
      seed=123,
      image_size=(480, 640),
      batch_size=32)

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
        layers.Dense(3)
    ])

    model.compile(
      optimizer='adam',
      loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'])

    model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=3
    )

    model.save("model.keras")




Choice = ""
Options = ["Rock","Paper","Scissors"]



model = keras.models.load_model("model.keras")

while Choice != "exit":
    Choice = input("Would you like to play a game of rock, paper scissors: ")
    if Choice == "yes":
        print("Say your choice in:")
        GetInput()
        ConvertData1()
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
        
        


