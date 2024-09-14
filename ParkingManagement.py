#SimpleParkingManagement : Simple parking management wich provide hability to detect if a parking place is free or busy by a truck
#Copyright (C) 2024 Mathieu DABERT
#
#This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any #later version.
#
#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more #details.
#
#You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

import datetime
from ultralytics import YOLO
import cv2
import io
import imutils
import math
import time
import socket
import struct
import numpy as np
from collections import defaultdict
from ParkingArea import ParkingArea
from ParkingObject import ParkingObject
import configparser
import sys
from hikvisionapi import Client
import io

#------------------------------------------
# Global
#------------------------------------------  
CONFIDENCE_THRESHOLD = 0.25
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)
mousecoordinates = "N/A"
parkingareas = list()
detectedobjects = list()

param_displayAll = False


#-------------------------------------------------------------------------------------------------------
# Helper
#-------------------------------------------------------------------------------------------------------
def getMouseCoordinates(event,x,y,flags,param):
    if event == cv2.EVENT_MOUSEMOVE:
        global mousecoordinates
        mousecoordinates = str(x) + ";" + str(y)


#------------------------------------------
# Load Config
#------------------------------------------  
config = configparser.ConfigParser()
config.read("main.ini")

nnmodelname = config['nn']['modelname']
analysepausetime = int(config['nn']['analysepausetime'])

settings_mindisplaydist = int(config['settings']['mindisplaydist'])
settings_minparkingdist = int(config['settings']['minparkingdist'])

configparkingarealist = list(filter(lambda k: 'ParkingArea' in k, config.sections()))

parkingareas.clear()

for aconfigparkingarea in configparkingarealist:
    tmp_coord = config[aconfigparkingarea]['coord']
    tmp_name = config[aconfigparkingarea]['name']
    tmp = ParkingArea()
    tmp.SetId(aconfigparkingarea)
    tmp.SetName(tmp_name)
    tmp.SetCoordinates(tmp_coord)
    tmp.SetDrawBoundingBox(False)
    parkingareas.append(tmp)
#------------------------------------------  


#------------------------------------------
# Init display
#------------------------------------------  
cv2.namedWindow('Image')
cv2.setMouseCallback('Image',getMouseCoordinates)
#------------------------------------------  


#------------------------------------------
# NN
#------------------------------------------  
model = YOLO(nnmodelname, verbose=False)
#------------------------------------------

print("Ready")


#------------------------------------------
# Ouverture flux video
#------------------------------------------  
#cap = cv2.VideoCapture("rtsp://x:x@x/ISAPI/Streaming/channels/101")

#still : http://x:x@x/ISAPI/Streaming/channels/101/picture

#if not cap.isOpened():
#    print("Cannot open camera")
#    exit()
#------------------------------------------

    
#------------------------------------------
# Initialisation image
#------------------------------------------      
#ret, frame = cap.read()
#if not ret:
#    print("Cannot get first frame")
#    exit()

cam = Client('http://x', 'x', 'x')
file_variable = io.BytesIO()   
   
response = cam.Streaming.channels[101].picture(method='get', type='opaque_data')
file_variable.seek(0)
for chunk in response.iter_content(chunk_size=1024):
    if chunk:
        file_variable.write(chunk)    
            
file_variable.seek(0)
img_array = np.asarray(bytearray(file_variable.read()), dtype=np.uint8)
frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

blackframe = frame.copy() 
blackframe[0:, 0:] = (0, 0, 0)
#------------------------------------------

#------------------------------------------
# Main loop
#------------------------------------------   
while True:

    start = datetime.datetime.now()
        
    # Grab frame
    #ret, frame = cap.read()
    response = cam.Streaming.channels[101].picture(method='get', type='opaque_data')
    file_variable.seek(0)
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            file_variable.write(chunk)    
                
    file_variable.seek(0)
    img_array = np.asarray(bytearray(file_variable.read()), dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    ret = 1

    # if frame is read correctly ret is True
    if ret:

        cpterror = 0
        
        #-------------------------------------------------------------------------------------------------------
        # run the YOLO model on the frame
        detections = model(frame)[0]
        
        detectedobjects.clear()
        # # loop over the detections
        for data in detections.boxes.data.tolist():
            
            confidence = data[4]
            name = model.names[int(data[5])]

            # Drawing
            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            
            # Blur People
            if name == "person":
                frame[ymin:ymax, xmin:xmax] = cv2.blur(frame[ymin:ymax, xmin:xmax] ,(23,23))  

            # filter out weak detections by ensuring the 
            # confidence is greater than the minimum confidence
            if float(confidence) < CONFIDENCE_THRESHOLD:
                continue

            if param_displayAll == False:
                if name != "truck":
                    continue
                
            tmp = ParkingObject()
            tmp.SetName(name)
            tmp.SetConfidence(confidence)
            tmp.SetBoundingBox(xmin, ymin, xmax, ymax)
            detectedobjects.append(tmp)
        #-------------------------------------------------------------------------------------------------------
        
    else:
        cpterror = cpterror+1
        frame = blackframe
     
    #-------------------------------------------------------------------------------------------------------
    for aparkingarea in parkingareas:
        aparkingarea.Calc()        
        frame = aparkingarea.Draw(frame)
        for adetectedobject in detectedobjects:
            #isin = aparkingarea.GetIsInBoundingBox(adetectedobject.GetCenter()) #ancienne méthode par boundingbox, pas terrible avec déformation objectif 2.8mm
            
            dist = aparkingarea.GetEuclideanDist(adetectedobject.GetCenter())

            if dist < settings_mindisplaydist:
                cv2.line(frame, (aparkingarea.GetCenter()), (adetectedobject.GetCenter()), GREEN, 1)
     
                xmin = min(aparkingarea.GetCenter()[0], adetectedobject.GetCenter()[0])
                ymin = min(aparkingarea.GetCenter()[1], adetectedobject.GetCenter()[1]) 
                xmax = max(aparkingarea.GetCenter()[0], adetectedobject.GetCenter()[0])
                ymax = max(aparkingarea.GetCenter()[1], adetectedobject.GetCenter()[1])
                x = xmin+int((xmax-xmin)/2)
                y = ymin+int((ymax-ymin)/2)                    
                
                cv2.putText(frame, str(dist), [x,y], cv2.FONT_HERSHEY_SIMPLEX, 1.5, GREEN, 1)

            if dist < settings_minparkingdist:                    
                isin = 1
            else:
                isin = 0
            
            if isin:
                adetectedobject.SetColor(RED)
                aparkingarea.SetLastSeen(datetime.datetime.now())
    print(len(detectedobjects))
    for adetectedobject in detectedobjects:
        frame = adetectedobject.Draw(frame)
    #-------------------------------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------------------------------
    end = datetime.datetime.now()
    total = (end - start).total_seconds()
    if total > 0:
        fps = f"FPS: {1 / total:.2f}"
    else:
        fps = f"FPS: ERROR"
    cv2.putText(frame, fps, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
    cv2.putText(frame, mousecoordinates, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
    cv2.putText(frame, f"Frame error: {cpterror}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
    cv2.putText(frame, f"Display All: {param_displayAll}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)    
    #-------------------------------------------------------------------------------------------------------
        
    #------------------------------------------
    # Display Stuff
    #------------------------------------------
    (h, w) = frame.shape[:2]
    new_width = 1920
    new_height = int(new_width * (h / w))
    resized_image = cv2.resize(frame, (new_width, new_height))
    cv2.imshow('Image', frame)
	
    if cv2.waitKey(1) == ord('q'):
        break
        
    if cv2.waitKey(1) == ord('d'):
        param_displayAll = not param_displayAll      
        	
#------------------------------------------
# End
#------------------------------------------ 
#cap.release() 
cv2.destroyAllWindows()



# Package            Version
# ------------------ ------------
# av                 12.3.0
# certifi            2024.7.4
# chardet            5.1.0
# charset-normalizer 3.3.2
# colorzero          2.0
# contourpy          1.2.1
# cycler             0.12.1
# distro             1.8.0
# filelock           3.15.4
# fonttools          4.53.1
# fsspec             2024.6.1
# gpiozero           2.0
# idna               3.7
# imutils            0.5.4
# Jinja2             3.1.4
# kiwisolver         1.4.5
# lgpio              0.2.2.0
# MarkupSafe         2.1.5
# matplotlib         3.9.2
# mpmath             1.3.0
# ncnn               1.0.20240410
# networkx           3.3
# numpy              1.26.4
# opencv-python      4.10.0.84
# packaging          24.1
# pandas             2.2.2
# picamera2          0.3.18
# pidng              4.0.9
# piexif             1.1.3
# pigpio             1.78
# pillow             10.4.0
# pip                24.2
# portalocker        2.10.1
# psutil             6.0.0
# py-cpuinfo         9.0.0
# pycryptodomex      3.11.0
# pyparsing          3.1.2
# python-apt         2.6.0
# python-dateutil    2.9.0.post0
# python-prctl       1.8.1
# pytz               2024.1
# PyYAML             6.0.2
# requests           2.32.3
# RPi.GPIO           0.7.1a4
# rpi-lgpio          0.6
# scipy              1.14.0
# seaborn            0.13.2
# setuptools         72.2.0
# simplejpeg         1.7.4
# six                1.16.0
# smbus2             0.4.2
# spidev             3.5
# ssh-import-id      5.10
# sympy              1.13.2
# toml               0.10.2
# torch              2.2.2
# torchvision        0.17.2
# tqdm               4.66.5
# typing_extensions  4.12.2
# tzdata             2024.1
# ultralytics        8.2.78
# ultralytics-thop   2.0.0
# urllib3            2.2.2
# v4l2-python3       0.3.4
# wheel              0.38.4

