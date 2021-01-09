import random
import aiml
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
import csv
import math
import re
import tweepy
from collections import Counter
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
import cv2
stopwords = stopwords.words('english')
warnings.filterwarnings('ignore')

consumer_key = 'X4GdKJsPC5FwUypV4R4yQWlWk'
consumer_secret = 'zF9m8Sra9lDnXC7iNkLG8AzT9ZlN7OblRqR64GdSuiPFiOg4TE'
access_token = '1318227698762874884-FPwVWzuBVEPZVRujJ0PQBoYPygWlt2'
access_token_secret = '8ERA3WilxXFPK0TtheA66Y54OPmah97klkuKg1rzXmKCk'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)

WORD = re.compile(r"\w+")

#Get bag of words

BagOfWords = []
###Cosine Similairity###
with open('CourseworkPart2Data.csv', newline='') as csvfile:
    Sentances = csv.reader(csvfile)
    for row in Sentances:
        TempWords = row[0].lower()
        Words = TempWords.split()
        for w in Words:
            if w in BagOfWords:
                Position = BagOfWords.index(w) +1
                BagOfWords[Position] = BagOfWords[Position] +1
            else:
                BagOfWords.append(w)
                BagOfWords.append(1)
    csvfile.close()


def ImplementFrequency(Vector):
    for x in BagOfWords:
        if BagOfWords.index(x)%2 == 0:
            if Vector[x] != 0:
                Vector[x] = Vector[x]/BagOfWords[BagOfWords.index(x) +1]
    return Vector
    
def CleanString(text):
    text = text.lower()
    temp = word_tokenize(text)
    temp1 = temp
    #temp = {w for w in temp if not w in stopwords}
    #if temp == "":
        #temp = temp1
    text = repr(temp)
    words = WORD.findall(text)
    return Counter(words)

def CreateVectors(Message,Orginal = False):
    vectors = CleanString(Message)
    if not Orginal:
        vectors = ImplementFrequency(vectors)
    return vectors

def CalculateCosine(vec1,vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

def FindBestResponseInCSV(Message):
    MessageVector = CreateVectors(Message,True)
    Response = ""
    Highest = 0
    Choice = random.randint(1,3)
    with open('CourseworkPart2Data.csv', newline='') as csvfile:
        Sentances = csv.reader(csvfile)
        for row in Sentances:
            MessageVector1 = CreateVectors(row[0].lower())
            Result = CalculateCosine(MessageVector,MessageVector1)
            if Result>Highest:
                Highest = Result
                Response = row[Choice]
        csvfile.close()
    return Response

def FindBestUserInCSV(Message):
    MessageVector = CreateVectors(Message,True)
    Response = ""
    Highest = 0
    with open('TwitterCheck.csv', newline='') as csvfile:
        Sentances = csv.reader(csvfile)
        for row in Sentances:
            MessageVector1 = CreateVectors(row[0].lower())
            Result = CalculateCosine(MessageVector,MessageVector1)
            if Result>Highest:
                Highest = Result
                Response = row[1]
        csvfile.close()
    return Response



modelAudio = keras.models.load_model("AudioModel.keras")
modelImage = keras.models.load_model("ImageModel.keras")

cam = cv2.VideoCapture(0)
ret, frame = cam.read()

###CNN###
def ConvertData1Audio():
    y, sr = librosa.load("Test.wav", mono=True, duration=2)
    plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, sides='default', mode='default', scale='dB');
    plt.axis('off');
    plt.savefig("Test.png")
    plt.clf()

def TestAudio():
    image = tf.keras.preprocessing.image.load_img("Test.png")
    input_arr = keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) 
    predictions = modelAudio.predict(input_arr)
    if predictions[0][0] > predictions[0][1] and predictions[0][0] > predictions[0][2]:
        print("Paper")
        return 1
    if predictions[0][1] > predictions[0][0] and predictions[0][1] > predictions[0][2]:
        print("Rock")
        return 0
    if predictions[0][2] > predictions[0][1] and predictions[0][2] > predictions[0][0]:
        print("Scissors")
        return 2
                                
def GetInputAudio():
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

def CropImage():
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

def TestImage():
    image = tf.keras.preprocessing.image.load_img("Test.png")
    
    input_arr = keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) 
    predictions = modelImage.predict(input_arr)
    if predictions[0][0] > predictions[0][1] and predictions[0][0] > predictions[0][2]:
        print("Paper")
        return 1
    if predictions[0][1] > predictions[0][0] and predictions[0][1] > predictions[0][2]:
        print("Rock")
        return 0
    if predictions[0][2] > predictions[0][1] and predictions[0][2] > predictions[0][0]:
        print("Scissors")
        return 2


                                
def GetInputImage():
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
    CropImage()



###Main###

kern = aiml.Kernel()
kern.setTextEncoding(None)

kern.bootstrap(learnFiles="CourseworkPart2.xml")

print("Welcome to this Twitter chat bot. Please feel free to ask for me to search for tweets or people for you, if your bored I can also play a game with you")

while True:
    try:
        Message = input("Enter your Message: ")
    except (KeyboardInterrupt, EOFError) as e:
        print("Bye!")
        break


    answer = kern.respond(Message)

    if answer[0] == '#':
        params = answer[1:].split('$')
        cmd = int(params[0])
        if cmd == 0:
            print(params[1])
            break
        elif cmd ==1:
            Search = params[1]
            Number = 0
            for tweet in tweepy.Cursor(api.search,q=Search,count=1,lang="en").items():
                if Number == 0:
                    Reply = WORD.findall(tweet.text)
                    Reply = " ".join(Reply)
                    print("My search for ", Search ," found the following most recent tweet: ", Reply)
                Number = Number +1
        ##Twitter search##
        elif cmd == 2:
            TwitterName = params[1]
            Return = FindBestUserInCSV(TwitterName)
            if Return == "":
                print("I couldnt find that user")
                Users = api.search_users(TwitterName,15)
                print("Is it any of the following?")
                Number = 0
                with open('TwitterCheck.csv','a', newline='') as csvfile:
                    NewNames = csv.writer(csvfile)
                    while Number <5:
                        try:
                            print(Users[Number].name)
                            NewNames.writerow([Users[Number].name,Users[Number].id])
                            Number = Number +1
                        except:
                            Number = Number +1
                    csvfile.close()
                Choice = input("->")
                Choice = Choice.lower()
                Return = FindBestUserInCSV(Choice)
            Return = api.user_timeline(Return,count=1)
            if Return == "":
                print("Unable to pull tweet, user could be on priavate or tweet contains unique text (1)")
            for status in Return:
                Reply = WORD.findall(status.text)
                Reply = " ".join(Reply)
                print("Most recent tweet by ", TwitterName ,": ", Reply)

        ##Game##
        elif cmd == 3:
            Choice = ""
            Options = ["Rock","Paper","Scissors"]

            print("We are about to play a game of rock, paper, scissors I can take your choice via audio(0) or image(1) input")

            GameType = input("")
            if GameType == "0":
                print("Say your choice in:")
                GetInputAudio()
                ConvertData1Audio()
                RandomNumber = random.randint(0,2)
                print("I choose ", Options[RandomNumber])
                print("You chose: ")
                Number = TestAudio()
                if Number == RandomNumber:
                    print("Draw")
                elif (Number == 0 and RandomNumber == 2) or (Number == 1 and RandomNumber == 0) or (Number == 2 and RandomNumber == 1):
                    print("You win!")
                else:
                    print("I win!")
            elif GameType == "1":
                print("Present choice to camera in:")
                GetInputImage()
                RandomNumber = random.randint(0,2)
                print("I choose ", Options[RandomNumber])
                print("You chose: ")
                Number = TestImage()
                if Number == RandomNumber:
                    print("Draw")
                elif (Number == 0 and RandomNumber == 2) or (Number == 1 and RandomNumber == 0) or (Number == 2 and RandomNumber == 1):
                    print("You win!")
                else:
                    print("I win!")
        ##Consine Similairty##                
        elif cmd == 99:
            Message = Message.lower()
            Return = FindBestResponseInCSV(Message)
            if Return == "TWEETSEARCH":
                Return = FindBestUserInCSV(Message)
                if Return == "":
                    print("I couldnt find that user")
                else:
                    Name = Return
                    Return = api.user_timeline(Return,count=1)
                    if Return == "":
                        print("Unable to pull tweet, user could be on priavate or tweet contains unique text (2)")
                    for status in Return:
                        Reply = WORD.findall(status.text)
                        Reply = " ".join(Reply)
                        print("Most recent tweet by ", Name ,": ", status.text.encode())
            elif Return == "GAME":
                Choice = ""
                Options = ["Rock","Paper","Scissors"]

                print("We are about to play a game of rock, paper, scissors I can take your choice via audio(0) or image(1) input")

                GameType = input("")
                if GameType == "0":
                    print("Say your choice in:")
                    GetInputAudio()
                    ConvertData1Audio()
                    RandomNumber = random.randint(0,2)
                    print("I choose ", Options[RandomNumber])
                    print("You chose: ")
                    Number = TestAudio()
                    if Number == RandomNumber:
                        print("Draw")
                    elif (Number == 0 and RandomNumber == 2) or (Number == 1 and RandomNumber == 0) or (Number == 2 and RandomNumber == 1):
                        print("You win!")
                    else:
                        print("I win!")
                elif GameType == "1":
                    print("Present choice to camera in:")
                    GetInputImage()
                    RandomNumber = random.randint(0,2)
                    print("I choose ", Options[RandomNumber])
                    print("You chose: ")
                    Number = TestImage()
                    if Number == RandomNumber:
                        print("Draw")
                    elif (Number == 0 and RandomNumber == 2) or (Number == 1 and RandomNumber == 0) or (Number == 2 and RandomNumber == 1):
                        print("You win!")
                    else:
                        print("I win!")
                    
            elif Return != "":
                print(Return)
            else:
                print("I dont understand, could you try rephrasing the question?")
    else:
        print(answer)
