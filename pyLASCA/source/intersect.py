#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 16:41:11 2020

@author: malkusch
"""
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

class Intersect(object):
    
    def __init__(self):
        self._roi_1 = np.zeros(0)
        self._roi_2 = np.zeros(0)
        self._roi_intersect = np.zeros(0)
        self._roi_1_area = float
        self._roi_2_area = float
        self._roi_intersect_area = float
    
    def __str__(self):
        message = str("the deteced shape is identified as a %s" %(self.shape))
        return(message)
        
    def __del__(self):
        message = str("Instance of Intersect removed form heap")
        print(message)
        
    @property
    def roi_1(self) -> np.ndarray:
        return(np.copy(self._roi_1))
    
    @roi_1.setter
    def roi_1(self, obj: np.ndarray):
        self._roi_1 = np.copy(obj)
    
    @property
    def roi_2(self) -> np.ndarray:
        return(np.copy(self._roi_2))
    
    @roi_2.setter
    def roi_2(self, obj: np.ndarray):
        self._roi_2 = np.copy(obj)
    
    @property
    def roi_intersect(self) -> np.ndarray:
        return(np.copy(self._roi_intersect))
    
    @property
    def roi_1_area(self) -> float:
        return(self._roi_1_area)
    
    @property
    def roi_2_area(self) -> float:
        return(self._roi_2_area)
    
    @property
    def roi_intersect_area(self) -> float:
        return(self._roi_intersect_area)
    
    
    def calculateIntresect(self):
        l1 =list()
        for i in range(np.shape(self.roi_1)[0]):
            l1.append(tuple((self.roi_1[i,0,0], self.roi_1[i,0,1])))
            
        l2 =list()
        for i in range(np.shape(self.roi_2)[0]):
            l2.append(tuple((self.roi_2[i,0,0], self.roi_2[i,0,1])))
        
        p1 = Polygon(l1)
        p2 = Polygon(l2)
        
        self._roi_1_area = p1.area
        self._roi_2_area = p2.area
        
        if(p1.intersects(p2)):
            p3 = p1.intersection(p2)
            x3,y3 = p3.exterior.xy
            roi = np.zeros([np.shape(x3)[0],1,2], dtype = int)
            roi[:,0,0] = x3[:]
            roi[:,0,1] = y3[:]
            self._roi_intersect = roi
            self._roi_intersect_area = p3.area
        else:
            self._roi_intersect_area = 0.0

        
    def plotIntersect(self):
        x1 = np.append(self.roi_1[:,0,0], self.roi_1[0,0,0])
        y1 = np.append(self.roi_1[:,0,1], self.roi_1[0,0,1])       
        plt.plot(x1, y1, color='magenta', alpha=0.7,
                 linewidth=3, solid_capstyle='round', zorder=2)
        
        x2 = np.append(self.roi_2[:,0,0], self.roi_2[0,0,0])
        y2 = np.append(self.roi_2[:,0,1], self.roi_2[0,0,1]) 
        plt.plot(x2, y2, color='cyan', alpha=0.7,
                 linewidth=3, solid_capstyle='round', zorder=2)
        
        if(self.roi_intersect_area > 0.0):
            plt.fill(self.roi_intersect[:,0,0], self.roi_intersect[:,0,1], facecolor='lightgreen', edgecolor='darkgreen', linewidth=3)
        
        plt.show()
        
        
def main():
    roi_1 = np.zeros([4,1,2], dtype = int)
    roi_1[:,0,0] = np.array([0,0,2,2])
    roi_1[:,0,1] = np.array([0,2,2,0])
    
    roi_2 = np.zeros([4,1,2], dtype = int)
    roi_2[:,0,0] = np.array([0,0,3,3])
    roi_2[:,0,1] = np.array([1,3,3,1])
    
    its = Intersect()
    its.roi_1 = roi_1
    its.roi_2 = roi_2
    its.calculateIntresect()
    print(its.roi_1_area,
          its.roi_2_area,
          its.roi_intersect_area)
    its.plotIntersect()    

if __name__ == '__main__':
    main()