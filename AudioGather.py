import sounddevice as sd
import soundfile as sf
#import numpy as np
import scipy.io.wavfile as wav
import time

fs=44100
duration = 2


for x in range(5):
    print("3")
    time.sleep(1)
    print("2")
    time.sleep(1)
    print("1")
    time.sleep(1)
    print ("Recording Audio")
    #time.sleep(0.3)
    myrecording = sd.rec(duration * fs, samplerate=fs, channels=2,dtype='float64')
    sd.wait()
    print ("Audio recording complete , Saving Audio")
    y=x+1
    Name = "Scissors Farmer " + str(y) +".wav"
    sf.write(Name, myrecording, fs)
    sd.wait()
    print ("Saving Audio Complete")
    time.sleep(1)
