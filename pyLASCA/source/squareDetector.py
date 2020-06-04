#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 10:03:19 2020

@author: malkusch
"""

import numpy as np
import pandas as pd
import cv2 as cv
from skimage.morphology import skeletonize
from matplotlib import pyplot as plt

class SquareDetector(object):
    
    def __init__(self):
        self._shape = str
        self._image_raw = np.zeros(0)
        self._image_binary = np.zeros(0)
        self._image_skeleton = np.zeros(0)
        self._method = 3
        self._kernel = (3,3)
        self._sigma = 0.35 
        self._cnts = []
        self._lines = []
        self._vertexes = []
        self._landmarks = pd.DataFrame()
        self._pxlSize = 1.0
        
    def __str__(self):
        message = str("the deteced shape is identified as a %s" %(self.shape))
        return(message)
        
    def __del__(self):
        message = str("Instance of SquareDetector removed form heap")
        print(message)
        
    @property
    def shape(self) -> str:
        return(self._shape)
    
    @property
    def image_raw(self) -> np.ndarray:
        return(np.copy(self._image_raw))
    
    @image_raw.setter
    def  rawImage(self, obj: np.ndarray):
        self._image_raw = np.copy(obj)
    
    @property
    def image_binary(self) -> np.ndarray:
        return(np.copy(self._image_binary))
    
    @property
    def image_skeleton(self) -> np.ndarray:
        return(np.copy(self._image_skeleton))
    
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
    
    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value: float):
        if(type(value) != float):
            errorMessage = str('Landmarks instance variavle sigma should be of type float, was of type %s' % (type(value)))
            raise Exception(errorMessage)
        else:
            self._sigma = value
    
    @property
    def cnts(self):
        return(self._cnts)
    
    @property
    def lines(self):
        return(self._lines)
    
    @property
    def vertexes(self):
        return(self._vertexes)
        
    @property
    def landmarks(self) -> pd.DataFrame:
        return(self._landmarks.copy())
    
    @property
    def pxlSize(self) -> float:
        return(self._pxlSize)
    
    @pxlSize.setter
    def pxlSize(self, value: float):
        self._pxlSize = value
        
    def loadImage(self, fileName: str):
        self.rawImage = cv.imread(fileName, cv.IMREAD_COLOR)
        
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
        
    def cannyEdge(self, image: np.ndarray):
        # =============================================================================
        # apply automatic Canny edge detection using the computed median
        # =============================================================================
        v = np.median(image)
        lower = int(max(0, (1.0 - self.sigma) * v))
        upper = int(min(255, (1.0 + self.sigma) * v))
        return(cv.Canny(image, lower, upper))
        
    def sharpen(self, image: np.ndarray,):
        kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
        image_sharp = cv.filter2D(image, -1, kernel)
        return(image_sharp)
    
    def removeNoise(self, image: np.ndarray):
        image_denoised = cv.medianBlur(image, 3)
        return image_denoised
    
    def binarize(self):
        image_gray = self.grayScale(self.image_raw)
        image_clean = self.removeNoise(image_gray)
        image_smooth = cv.GaussianBlur(image_clean,self.kernel, 1.0)
        image_edge = self.cannyEdge(image_smooth)
        self._image_binary = self.contourBinary(image_edge)
    
    def skeletonize(self):
        binary_hull =  np.zeros((self.image_binary.shape[0], self.image_binary.shape[1], 1), dtype=np.uint8)
        cv.drawContours(binary_hull, self.cnts, -1, 255, 1)
        binary_hull = cv.dilate(binary_hull, None, iterations = 3)
        #self._image_skeleton = binary_hull
        img_skel = skeletonize(binary_hull > 0)
        self._image_skeleton = 255*(img_skel.astype(np.uint8))
    
    def contourBinary(self, image: np.ndarray):
        _, contours, hierarchy = cv.findContours(image, cv.RETR_EXTERNAL, 2)
        binary =  np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)
        for i in range(len(contours)):
            cv.drawContours(binary, contours, i, 255, 1, cv.LINE_8, hierarchy, 0)
            
        binary = cv.dilate(binary.copy(), None, iterations = 1)
        binary = cv.erode(binary, None, iterations = 1)
        
        image_filled = binary.copy()
        
        h, w = image.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        cv.floodFill(image_filled, mask, (0,0), 255)
        image_filled= cv.bitwise_not(image_filled)
        
        image_binary = (binary | image_filled)
        return(image_binary)
    
    def detectCnts(self):
        _, contours, hierarchy = cv.findContours(self.image_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)        
        self._cnts = sorted(contours, key=lambda x: cv.contourArea(x))
        
    
    
    def houghTransfrom(self):        
        minLineLength = 25
        maxLineGap = 0
        self._lines = cv.HoughLines(self.image_skeleton,1,np.pi/180,minLineLength,maxLineGap)
                
        
    def drawLines(self):
        image = cv.cvtColor(self.sharpen(self.image_raw), cv.COLOR_BGR2RGB)
        for rho,theta in self.lines[:,0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv.line(image,(x1,y1),(x2,y2),(255,0,0),2)
            
        caption = str("Hough Lines:")
        plt.imshow(image)
        plt.xticks([]), plt.yticks([])
        plt.title(caption)
        plt.show()
        
    def drawLine(self, idx: int):
        image = cv.cvtColor(self.sharpen(self.image_raw), cv.COLOR_BGR2RGB)
        if(idx < len(self.lines)):
            rho,theta = self.lines[idx,0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv.line(image,(x1,y1),(x2,y2),(255,0,0),2)
            
        caption = str("Hough Line: %i" %idx)
        plt.imshow(image)
        plt.xticks([]), plt.yticks([])
        plt.title(caption)
        plt.show()
        
    def deleteLine(self, idx: int):
        if(idx < len(self.lines)):
            self._lines = np.delete(self._lines, (idx), axis=0)
        
    def detectVertexes(self):
        pts = []
        for i in range(self.lines.shape[0]):
            (rho1, theta1) = self.lines[i,0]
            if (np.abs(np.tan(theta1)) <= 1e-8):
                continue
            m1 = -1/np.tan(theta1)
            if (np.abs(np.sin(theta1)) <= 1e-8):
                continue
            b1 = rho1 * np.sin(theta1) - m1 * rho1 * np.cos(theta1)
            for j in range(i+1,self.lines.shape[0]):
                (rho2, theta2) = self.lines[j,0]
                if (np.abs(np.tan(theta2)) <= 1e-8):
                    continue
                m2 = -1/np.tan(theta2)
                if (np.abs(np.sin(theta2)) <= 1e-8):
                    continue
                b2 = rho2 * np.sin(theta2) - m2 * rho2 * np.cos(theta2)
                if (np.abs(m1 - m2) <= 1e-8):
                    continue
                x = (b2 - b1) / (m1 - m2)
                y = m1*x + b1
                alpha = np.abs(theta1- theta2) * 180.0/np.pi
                if ((0 <= x < self.image_raw.shape[1]) and
                    (0 <= y < self.image_raw.shape[0]) and
                    (alpha > 90-10) and
                    (alpha < 90+10)):
                    pts.append((int(x), int(y)))
      
        pts = np.array(pts)
        
        # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        
        # Set flags (Just to avoid line break in the code)
        flags = cv.KMEANS_RANDOM_CENTERS
        
        # Apply KMeans
        # The convex hull points need to be float32
        z = pts.copy().astype(np.float32)
        compactness,labels,centers = cv.kmeans(z,4,None,criteria,10,flags)
        
        self._vertexes = centers[:,None] # We need to convert to a 3D numpy array with a singleton 2nd dimension
        
    def drawVertexes(self):
        image = cv.cvtColor(self.sharpen(self.image_raw), cv.COLOR_BGR2RGB)
        for i in range(len(self.vertexes)):
            x = self.vertexes[i,0,0]
            y = self.vertexes[i,0,1]
            cv.circle(image, center = (x,y), radius = 3, color = [255,0,0], thickness=3)
            
        caption = str("Vertexes:")
        plt.imshow(image)
        plt.xticks([]), plt.yticks([])
        plt.title(caption)
        plt.show()
        
    def detectLandmarks(self):
        if len(self.vertexes) == 4:
            x_c = np.sum(self.vertexes[:,0,0])/4.0
            y_c = np.sum(self.vertexes[:,0,1])/4.0
            dx = self.vertexes[:,0,0] - x_c
            dy = self.vertexes[:,0,1] - y_c
            alpha = np.arctan2(dx, dy) * 180 / np.pi
            alpha[alpha < 0] = alpha[alpha < 0] + 360
            lm_df = pd.DataFrame({"x_pxl": self.vertexes[:,0,0],
                                  "y_pxl": self.vertexes[:,0,1],
                                  "x_mm": self.vertexes[:,0,0] * self.pxlSize,
                                  "y_mm": self.vertexes[:,0,1] * self.pxlSize,
                                  "alpha": alpha})
            lm_df.sort_values(by="alpha", ascending = True, inplace=True)
            self._landmarks = lm_df.copy()
                
        else:
            ValueError("Number of Vertexes need to be exact 4!")
            
    def drawImage(self):
        image = cv.cvtColor(self.sharpen(self.image_raw), cv.COLOR_BGR2RGB)
        caption = str("Raw Image:")
        plt.imshow(image)
        plt.xticks([]), plt.yticks([])
        plt.title(caption)
        plt.show()
        
    def analyzeImage(self):
        self.binarize()
        self.detectCnts()
        self.skeletonize()
        self.houghTransfrom()
        
    def saveLandmarks(self, fileName = str):
        self.landmarks.to_csv(path_or_buf = fileName, index = False)
        print("landmarks written to  %s" %(fileName))
    
        
def main():
    #fileName = "/Users/malkusch/PowerFolders/LaSCA/Probanden Arm Bilder mit Markierung/SCTCAPS 05/Tag 2/SCTCAPS 05 #2 45min (Colour).jpg"
    fileName = "/Users/malkusch/PowerFolders/LaSCA/Probanden Arm Bilder mit Markierung/SCTCAPS 01/Tag 2/SCTCAPS 01 #2  -15min (Colour).jpg"
    sq = SquareDetector()
    sq.loadImage(fileName)
    sq.sigma = 5.0
    sq.binarize()
    sq.detectCnts()
    sq.skeletonize()
    sq.houghTransfrom()
    sq.drawLines()
    sq.detectVertexes()
    sq.drawVertexes()
    sq.detectLandmarks()
    print(sq.landmarks)


    
    #sq.analyzeImage()
    #sq.drawContours()
# =============================================================================
#     sq.analyzeImage()
#     sq.drawContours()
# =============================================================================
# =============================================================================
#     img_bin = sq.binary()
#     img_lap = cv.Laplacian(img_bin,cv.CV_64F)
#     plt.subplot(2,2,1),plt.imshow(img_bin,cmap = 'gray')
#     plt.title('Original'), plt.xticks([]), plt.yticks([])
#     plt.subplot(2,2,2),plt.imshow(255 - img_lap,cmap = 'gray')
#     plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
#     plt.show()
# =============================================================================
    
    #sq.houghTransfrom()
    #laplacian = cv.Laplacian(img,cv.CV_64F)
    
if __name__ == '__main__':
    main()