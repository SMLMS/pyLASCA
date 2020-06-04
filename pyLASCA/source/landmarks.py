#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 11:13:21 2020

@author: malkusch
"""

import numpy as np
import pandas as pd
import cv2 as cv
from matplotlib import pyplot as plt


class Landmarks:
    def __init__(self):
        self._rawImage = np.zeros(0)
        self._method = 3
        self._kernel = (9,9)
        self._sigma = 0.35
        self._thr = 254
        self._cnts = []
        self._landmarks = pd.DataFrame()
        self._pxlSize = 1.0
            
    @property
    def rawImage(self):
        return np.copy(self._rawImage)

    @rawImage.setter
    def rawImage(self, image: np.ndarray):
        if(type(image) != np.ndarray):
            errorMessage = str('Landmarks instance variable rawImage should be of type numpy.ndarray, was of type %s.' % (type(image)))
            raise Exception(errorMessage)
        else:
            self._rawImage = image.astype(dtype = np.uint8)
            
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
    
    @kernel.setter
    def kernel(self, value: tuple):
        if(type(value) != tuple):
            errorMessage = str('Landmarks instance variable kernel should be of type tuple, was of type %s.' % (type(value)))
            raise Exception(errorMessage)
        elif(len(value) != 2):
            errorMessage = str('Landmarks instance variable kernel should be of of length 2, was of length %i.' % (len(value)))
            raise Exception(errorMessage) 
        else:
            self._kernel = value
        
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
    def thr(self):
        return self._thr
    
    @thr.setter
    def thr(self, value: int):
        if(type(value) != int):
            errorMessage = str('Landmarks instance variable thr should be of type int, was of type %s.' % (type(value)))
            raise Exception(errorMessage)
        else:
            self._thr = value
            
    @property
    def cnts(self):
        return self._cnts
    
    @property
    def landmarks(self):
        return(self._landmarks.copy())
    
    @property
    def pxlSize(self):
        return self._pxlSize
    
    @pxlSize.setter
    def pxlSize(self, value: float):
        if(type(value) != float):
            errorMessage = str('Landmarks instance variable pxlSize should be of type float, was of type %s.' % (type(value)))
            raise Exception(errorMessage)
        else:
            self._pxlSize = value
    
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
        
    def contourBinary(self, image: np.ndarray, contours):
        # =============================================================================
        # smooth by convolution with gaussian low pass
        # =============================================================================
        img1 = cv.GaussianBlur(image,self.kernel, 0.35)
        # =============================================================================
        # fill contours with white
        # =============================================================================
        img2 = cv.fillPoly(img1, pts = contours, color=(255,255,255))
        # =============================================================================
        # convert to gray scale
        # =============================================================================
        img3 = self.grayScale(img2)
        # =============================================================================
        # create binary image, erode and dilate
        # =============================================================================
        ret, img4 = cv.threshold(img3, self.thr, 255, cv.THRESH_BINARY)
        img5 = cv.dilate(img4, None, iterations = 10)
        img6 = cv.erode(img5, None, iterations = 10)
        return(img6)
        
    def sharpen(self, image: np.ndarray,):
        kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
        image_sharp = cv.filter2D(image, -1, kernel)
        return(image_sharp)
        
    def detectContours(self):
        image = self.rawImage
        sharpImage = self.sharpen(image)
        grayImage = 255 - self.grayScale(sharpImage)
        denoisedImage = cv.medianBlur(grayImage, 3)
        edgeImage = self.cannyEdge(denoisedImage)
        cImage1, contours1, hierarchy2 = cv.findContours(edgeImage, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        binaryImage = self.contourBinary(image, contours1)
        cImage2, contours2, hierarchy2 = cv.findContours(binaryImage, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        self._cnts = sorted(contours2, key=lambda x: cv.contourArea(x))
        
    def deleteContours(self, index: int):
        if(index < self.countContours()):
            del(self._cnts[index])
        
    
    def detectLandmarks(self):
        x = []
        y = []
        for i in np.arange(0, self.countContours(), 1):
            M = cv.moments(self.cnts[i])
            x = np.append(x, M["m10"] / M["m00"])
            y = np.append(y,  M["m01"] / M["m00"])
        lm_df = pd.DataFrame({"x_pxl": x, "y_pxl": y})
        lm_df["x_mm"] = lm_df["x_pxl"] * self.pxlSize
        lm_df["y_mm"] = lm_df["y_pxl"] * self.pxlSize
        self._landmarks = lm_df
    
    def sortLandmarks(self):
        idx = np.array([self.landmarks["y_pxl"].argmax(),
                          self.landmarks["x_pxl"].argmax(),
                          self.landmarks["y_pxl"].argmin(),
                          self.landmarks["x_pxl"].argmin()])
        if ((np.unique(idx).size == len(idx)) & (len(idx) == 4)):
            self._landmarks["rank"] = idx
            self._landmarks.sort_values(by="rank", ascending = True, inplace=True)
            self._landmarks["alpha"] = np.array([0,90,180,270])
            self._landmarks.drop(labels = "rank", axis=1, inplace=True)
        else:
            raise ValueError('Wrong index list created during sortLandmarks procedure')
        
    
    def countContours(self):
        return len(self.cnts)
    
    def landmarkArray(self):
        la = [-1]
        for i in range(0, self.countContours(), 1):
            la = np.append(la, i)
        return la
    
    def loadImage(self, fileName: str):
        self.rawImage = cv.imread(fileName, cv.IMREAD_COLOR)
        
    def drawContours(self):
        image = cv.drawContours(self._rawImage, self.cnts, -1, (0,255,0), 3)
        return(image)
    
    def drawCompleteLandmarks(self):
        image = cv.cvtColor(self.sharpen(self.rawImage), cv.COLOR_BGR2RGB)
        for i in np.arange(0, self.countContours(), 1):
            cv.circle(image, (int(self.landmarks["x_pxl"][i]), int(self.landmarks["y_pxl"][i])), 7, (255, 255, 255), -1)
        plt.imshow(image)
        plt.xticks([]), plt.yticks([])
        plt.title("Landmark Iamge")
        plt.show()
        
    def drawSingleLandmark(self, index: int):
        image = cv.cvtColor(self.sharpen(self.rawImage), cv.COLOR_BGR2RGB)
        cv.circle(image, (int(self.landmarks["x_pxl"][index]), int(self.landmarks["y_pxl"][index])), 7, (255, 255, 255), -1)
        caption = str("Landmark index: %i" %(index))
        plt.imshow(image)
        plt.xticks([]), plt.yticks([])
        plt.title(caption)
        plt.show()
        
    def drawLandmarks(self, index = -1):
        if (index == -1):
            self.drawCompleteLandmarks()
        else:
            self.drawSingleLandmark(index)
    
    def drawImage(self):
        image = cv.cvtColor(self.sharpen(self.rawImage), cv.COLOR_BGR2RGB)
        plt.imshow(image)
        plt.xticks([]), plt.yticks([])
        plt.title("Raw Iamge")
        plt.show()
        
    def saveLandmarks(self, fileName = str):
        self.landmarks.to_csv(path_or_buf = fileName, index = False)
        print("landmarks written to  %s" %(fileName))
        
        
def main():
    fileName = "../../Arm_photo.bmp"
    lm = Landmarks()
    lm.loadImage(fileName)
    lm.drawImage()
    lm.detectContours()
# =============================================================================
#     lm.detectContours()
#     lm.detectLandmarks()
#     lm.sortLandmarks()
#     i = 0
#     lm.drawLandmarks(i)
#     print(lm.landmarkArray())
#     print(lm.landmarks)
# =============================================================================
     
if __name__ == '__main__':
    main()