import cv2
import time
from PIL import Image
import matplotlib.pyplot as plt
import pathlib
import os


cam = cv2.VideoCapture(0)
for x in range(0,200):
    
    print("1")
    print ("Taking Photo")
    ret, frame = cam.read()
    img_name = "Scissors New " + str(x+2000) + ".png"
    cv2.imwrite(img_name, frame)
    print("Image Saved!")
    picture = Image.open(img_name)
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
    picture.save(img_name,"png")    
