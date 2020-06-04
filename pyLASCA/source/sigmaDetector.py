#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 10:39:48 2020

@author: malkusch
"""

import numpy as np
import cv2 as cv
from skimage.morphology import skeletonize
import random as rng

class SigmaDetector(object):
    
    def __init__(self):
        self._image = np.ndarray
        self._sigma = int
        self._method = 3
        self._kernel = (3,3)
        self._seed = 42
        
    def __str__(self):
        message = str("the deteced shape is identified as a %s" %(self.shape))
        return(message)
        
    def __del__(self):
        message = str("Instance of SigmaDetector removed form heap")
        print(message)
        
    @property
    def image(self) -> np.ndarray:
        return(np.copy(self._image))
    
    @image.setter
    def image(self, obj: np.ndarray):
        self._image = np.copy(obj)
        
    @property
    def sigma(self) -> int:
        return(self._sigma)
    
    @sigma.setter
    def sigma(self, value: int):
        self._sigma = value
        
    @property
    def seed(self) -> int:
        return(self._seed)
    
    @seed.setter
    def seed(self, value: int):
        self._seed = value
        
    @property
    def method(self):
        return self._method
    
    @method.setter
    def method(self, value: int):
        if(type(value) != int):
            self._method = 0
            errorMessage = str('Landmarks instance variable method should be of type int, was of type %s.' % (type(value)))
            raise Exception(errorMessage)
        elif(value < 0):
            self._method = 0
            errorMessage = str('Landmarks instance variable method must be positive, was %i.' % (value))
            raise Exception(errorMessage)
        elif(value > 3):
            self._method = 0
            errorMessage = str('Landmarks instance variable method must be smaller 4, was %i.' % (value))
        else:
            self._method = value
            
    @property
    def kernel(self):
        return self._kernel
    
    def loadImage(self, fileName: str):
        self.image = cv.imread(fileName, cv.IMREAD_COLOR)
    
    def updateSeed(self):
        rng.seed(self.seed)
            
    def grayScale(self, image: np.ndarray):
        if (self.method == 0):
            return(cv.cvtColor(image, cv.COLOR_BGR2GRAY))
        elif (self.method == 1):
            return(image[:,:,0])
        elif (self.method == 2):
            return(image[:,:,1])
        elif (self.method == 3):
            return(image[:,:,2])
        else:
            errorMessage = str('Landmarks instance variable method should be within interval [0, 3], was %i.' % (self.method))
            raise Exception(errorMessage)
        
    def cannyEdge(self, img: np.ndarray):
        # =============================================================================
        # apply automatic Canny edge detection using the computed median
        # =============================================================================
        image_gray = img.copy()
        v = np.median(image_gray)
        lower = int(max(0, (1.0 - self.sigma) * v))
        upper = int(min(255, (1.0 + self.sigma) * v))
        return(cv.Canny(image_gray, lower, upper))
    
    def thresh_callback(self, value: int):
        self.sigma = value * 0.1
        src_gray = self.grayScale(self.image)
        src_gray = cv.medianBlur(src_gray, 3)
        src_gray = cv.GaussianBlur(src_gray,self.kernel, 1.0)
        # Detect edges using Canny
        canny_output = self.cannyEdge(src_gray)
        # Find contours
        _, contours, hierarchy = cv.findContours(canny_output, cv.RETR_EXTERNAL, 2)
        # Draw contours
        drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
        for i in range(len(contours)):
            color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
            cv.drawContours(drawing, contours, i, color, 2, cv.LINE_8, hierarchy, 0)
        # Show in a window
        cv.imshow('Contours', drawing)
        
        binary =  np.zeros((canny_output.shape[0], canny_output.shape[1], 1), dtype=np.uint8)
        cv.drawContours(binary, contours, -1, 255, 1)
        
        img = cv.dilate(binary.copy(), None, iterations = 1)
        img = cv.erode(img, None, iterations = 1)
        
        img_floodfill = img.copy()
        h, w = img.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        cv.floodFill(img_floodfill, mask, (0,0), 255)
        img_floodfill= cv.bitwise_not(img_floodfill)
        
        img_bin = (img | img_floodfill)
        
        img, contours2, hierarchy = cv.findContours(img_bin, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
        binary_hull =  np.zeros((canny_output.shape[0], canny_output.shape[1], 1), dtype=np.uint8)
        cv.drawContours(binary_hull, contours2, -1, 255, 1)
            
        binary_hull = cv.dilate(binary_hull, None, iterations = 3)
        
        cv.imshow('Binary', binary_hull)
        
        out = skeletonize(binary_hull > 0)
        out = 255*(out.astype(np.uint8))
        minLineLength = 25
        maxLineGap = 0
        
        lines = cv.HoughLines(out,1,np.pi/180,minLineLength,maxLineGap)
        
        image_lines = self.image
        for rho,theta in lines[:,0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv.line(image_lines,(x1,y1),(x2,y2),(255,0,0),2)
            
        cv.imshow('Binary', image_lines)
        
    def gui(self):
        max_thresh = 200
        thresh = 50 # initial threshold
        source_window = 'Source'
        cv.namedWindow(source_window)
        cv.imshow(source_window, self.image)
        cv.createTrackbar('Canny Thresh:', source_window, thresh, max_thresh, self.thresh_callback)
        self.thresh_callback(thresh)
        while(True):
            k = cv.waitKey(33)
            if k == -1:  # if no key was pressed, -1 is returned
                continue
            else:
                break
        cv.destroyAllWindows()
        cv.waitKey(1)

        
        
        
def main():
    fileName = "/Users/malkusch/PowerFolders/LaSCA/Probanden Arm Bilder mit Markierung/SCTCAPS 05/Tag 2/SCTCAPS 05 #2 45min (Colour).jpg"
    #fileName = "/Users/malkusch/PowerFolders/LaSCA/Probanden Arm Bilder mit Markierung/SCTCAPS 01/Tag 2/SCTCAPS 01 #2  -15min (Colour).jpg"
    sd = SigmaDetector()
    sd.updateSeed()
    sd.loadImage(fileName)
    sd.gui()
    
    print(sd.sigma)
    
if __name__ == '__main__':
    main()
    