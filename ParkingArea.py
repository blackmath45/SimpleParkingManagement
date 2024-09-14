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
import math
import datetime

GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)
ORANGE = (0, 128, 255)

class ParkingArea:
    def __init__(self):
        #print("init")
        self.name = "None"
        self.coord = []
        self.id = -1
        self.boundingbox = []
        self.drawBoundingBox = False
        self.npshape = []
        self.center = []
        self.status = 0
        self.lastseen = datetime.datetime(2000,1,1)
        self.tempoBeforeOccupied = 20
        self.tempoAfterFree = 30
        
    def SetTempo(self, beforeoccupied, afterfree):
        self.tempoBeforeOccupied = beforeoccupied
        self.tempoAfterFree = afterfree

    def SetId(self, id):
        self.id = id
		
    def SetName(self, name):
        self.name = name
        
    def SetDrawBoundingBox(self, enable):
        self.drawBoundingBox = enable
		
    def SetCoordinates(self, coord):
        self.coord = eval(coord)
        self.npshape = np.array(self.coord, np.int32)
        #boundingbox
        min_x, min_y = np.min(self.npshape, axis=0)
        max_x, max_y = np.max(self.npshape, axis=0)   
        self.boundingbox = [min_x, min_y],[max_x, max_y]
        self.center = [min_x + int((max_x-min_x)/2), min_y + int((max_y-min_y)/2)]
        
    def SetLastSeen(self, lastseen):
        self.lastseen = lastseen
        
    def GetBoundingBox(self):
        return self.boundingbox
        
    def GetCenter(self):
        return self.center
        
    def GetIsInBoundingBox(self, objectcoord):
        objectcoord_x = objectcoord[0]
        objectcoord_y = objectcoord[1]
        boundingbox_xmin = self.boundingbox[0][0]
        boundingbox_xmax = self.boundingbox[1][0]
        boundingbox_ymin = self.boundingbox[0][1]
        boundingbox_ymax = self.boundingbox[1][1]
        
        if (objectcoord_x >= boundingbox_xmin) and (objectcoord_x <= boundingbox_xmax) and (objectcoord_y >= boundingbox_ymin) and (objectcoord_y <= boundingbox_ymax):
            return True
        else:
            return False

    def GetEuclideanDist(self, center):
        centercoord_x = center[0]
        centercoord_y = center[1]
        mycenter_x = self.center[0]
        mycenter_y = self.center[1]

        dist = math.sqrt(pow((centercoord_x - mycenter_x), 2) + pow((centercoord_y - mycenter_y), 2))
        
        return int(dist)
        
    def Calc(self):
        if (datetime.datetime.now()-self.lastseen).total_seconds() > self.tempoAfterFree:
            self.status = 0
        else:
            self.status = 1
        
    def Draw(self, image):
        cv2.polylines(image, [self.npshape], True, WHITE, 2)
        if self.drawBoundingBox == True:
            cv2.rectangle(image, self.boundingbox[0], self.boundingbox[1], ORANGE, 2)
        cv2.line(image, (self.center[0]-50, self.center[1]), (self.center[0]+50, self.center[1]), WHITE, 4) 
        cv2.line(image, (self.center[0], self.center[1]-50), (self.center[0], self.center[1]+50), WHITE, 4)  
        cv2.putText(image, self.name, (self.center[0]+4, self.center[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
        if self.status == 1:
            cv2.putText(image, "Busy", (self.center[0]+4, self.center[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2)
        else:
            cv2.putText(image, "Free", (self.center[0]+4, self.center[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN, 2)            
        return image