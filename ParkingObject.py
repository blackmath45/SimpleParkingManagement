#SimpleParkingManagement : Simple parking management wich provide hability to detect if a parking place is free or busy by a truck
#Copyright (C) 2024 Mathieu DABERT
#
#This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any #later version.
#
#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more #details.
#
#You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import cv2

GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)
ORANGE = (0, 128, 255)

class ParkingObject:
    def __init__(self):
        #print("init")
        self.name = "None"
        self.confidence = -1
        self.boundingbox = []
        self.center = []
        self.color = BLUE
		
    def SetName(self, name):
        self.name = name
        
    def SetConfidence(self, confidence):
        self.confidence = confidence        
		
    def SetBoundingBox(self, min_x, min_y, max_x, max_y):
        #boundingbox
        self.boundingbox = [min_x, min_y],[max_x, max_y]
        self.center = [min_x + int((max_x-min_x)/2), min_y + int((max_y-min_y)/2)]
        
    def GetCenter(self):
        return self.center
        
    def SetColor(self, color):
        self.color = color
        
    def Draw(self, image):
        cv2.rectangle(image, self.boundingbox[0], self.boundingbox[1], self.color, 2)
        cv2.line(image, (self.center[0]-50, self.center[1]), (self.center[0]+50, self.center[1]), self.color, 5) 
        cv2.line(image, (self.center[0], self.center[1]-50), (self.center[0], self.center[1]+50), self.color, 5)
        title = self.name + " (" + str(int(self.confidence*100)) + "%)"
        cv2.putText(image, title, (self.boundingbox[0][0], self.boundingbox[0][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.color, 2)
        return image