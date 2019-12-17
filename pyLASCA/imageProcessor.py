#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 16:13:18 2019

@author: malkusch
"""

import numpy as np
import pandas as pd
import cv2 as cv

class ImageProcessor:
    def __init__(self):
        self._folderName = str()
        self._baseName = str()
        self._rawImage = np.zeros(0)
        self._binImage = np.zeros(0)
        self._kernel = (9,9)
        self._thr = 1
        self._cnts = []
        self._hll = []
        self._pxlSize = 1.0

    @property
    def folderName(self):
        return self._folderName
    
    @folderName.setter
    def folderName(self, name: str):
        if(type(name) != str):
            errorMessage = str('ImageProcessor instance variable folderName should be of type str, was of type %s.' % (type(name)))
            raise Exception(errorMessage)
        else:
            self._folderName = name
            
    @property
    def baseName(self):
        return self._baseName
    
    @baseName.setter
    def baseName(self, name: str):
        if(type(name) != str):
            errorMessage = str('ImageProcessor instance variable baseName should be of type str, was of type %s.' % (type(name)))
            raise Exception(errorMessage)
        else:
            self._baseName = name
    
    @property
    def rawImage(self):
        return np.copy(self._rawImage)
    
    @rawImage.setter
    def rawImage(self, image: np.ndarray):
        if(type(image) != np.ndarray):
            errorMessage = str('ImageProcessor instance variable rawImage should be of type numpy.ndarray, was of type %s.' % (type(image)))
            raise Exception(errorMessage)
        else:
            self._rawImage = image.astype(dtype = np.uint8)
         
    @property
    def binImage(self):
        return np.copy(self._binImage)
    
    @property
    def kernel(self):
        return self._kernel
    
    @kernel.setter
    def kernel(self, value: tuple):
        if(type(value) != tuple):
            errorMessage = str('ImageProcessor instance variable kernel should be of type tuple, was of type %s.' % (type(value)))
            raise Exception(errorMessage)
        elif(len(value) != 2):
            errorMessage = str('ImageProcessor instance variable kernel should be of of length 2, was of length %i.' % (len(value)))
            raise Exception(errorMessage) 
        else:
            self._kernel = value
        
    @property
    def thr(self):
        return self._thr
    
    @thr.setter
    def thr(self, value: int):
        if(type(value) != int):
            errorMessage = str('ImageProcessor instance variable thr should be of type int, was of type %s.' % (type(value)))
            raise Exception(errorMessage)
        else:
            self._thr = value
            
    @property
    def cnts(self):
        return self._cnts
    
    @property
    def hll(self):
        return self._hll
    
    @property
    def pxlSize(self):
        return self._pxlSize
    
    @pxlSize.setter
    def pxlSize(self, value: float):
        if(type(value) != float):
            errorMessage = str('ImageProcessor instance variable pxlSize should be of type float, was of type %s.' % (type(value)))
            raise Exception(errorMessage)
        else:
            self._pxlSize = value
    
    def detectContours(self):
        # =============================================================================
        # smooth by convolution with gaussian low pass
        # =============================================================================
        im1 = cv.GaussianBlur(self.rawImage,self.kernel, 0)
        # =============================================================================
        # create binary image
        # dilate and erode (close)
        # =============================================================================
        ret, im2 = cv.threshold(im1, self.thr, 255, cv.THRESH_BINARY)
        im3 = cv.dilate(im2, None, iterations = 1)
        self._binImage = cv.erode(im3, None, iterations = 1)
        # =============================================================================
        # detect contours
        # =============================================================================
        im5, contours, hierarchy = cv.findContours(self.binImage, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        # =============================================================================
        # sort contours by area
        # =============================================================================
        self._cnts = sorted(contours, key=lambda x: cv.contourArea(x))
    
    def calcConvexHull(self):
        # =============================================================================
        # clear array for convex hull points
        # =============================================================================
        self._hll = []
        # =============================================================================
        # calculate hull for each contour        
        # =============================================================================
        for i in range(0,len(self.cnts),1):
            self._hll.append(cv.convexHull(self.cnts[i], False))
        
        
    
    def countContours(self):
        return len(self.cnts)
    
    def contourArea(self, idx: int):
        if(len(self.cnts) > 0):
            area = cv.contourArea(self.cnts[idx]) * (self.pxlSize**2)
        else:
            area = 0.0
        return area
    
    def contourRadius(self, idx: int):
        r = np.sqrt(self.contourArea(idx)/np.pi) * self.pxlSize
        return r
    
    def contourPerimeter(self, idx: int):
        if (len(self.cnts)>0):
            p = cv.arcLength(self.cnts[idx],True) * self.pxlSize
        else:
            p = 0.0
        return p
    
    def contourCircularity(self, idx: int):
        if (len(self.cnts)>0):
            c = 2.0*np.pi*self.contourRadius(idx)/self.contourPerimeter(idx)
        else:
            c = 0.0
        return c
    
    def contourIntensity(self, idx: int):
        if(len(self.cnts)>0):
            intList = []
            # =============================================================================
            #  Create a mask image that contains the contour filled in
            # =============================================================================
            cnt = self.cnts[idx]
            cimg = np.zeros_like(self.rawImage)
            cv.drawContours(cimg, [cnt], 0, color=255, thickness=-1)
            # =============================================================================
            # Access the image pixels and create a 1D numpy array then add to list
            # =============================================================================
            pts = np.where(cimg == 255)
            intList.append(self.rawImage[pts[0], pts[1]])
            intensity = np.sum(intList)
        else:
            intensity = 0
        return intensity
    
    def contourIntensityDensity(self, idx: int):
        if(len(self.cnts)>0):
            intDen = self.contourIntensity(idx)/self.contourArea(idx)
        else:
            intDen = 0.0
        return intDen
        
    def hullArea(self, idx: int):
        if(len(self.hll) > 0):
            area = cv.contourArea(self.hll[idx]) * (self.pxlSize**2)
        else:
            area = 0
        return area
    
    def hullRadius(self, idx: int):
        r = np.sqrt(self.hullArea(idx)/np.pi) * self.pxlSize
        return r
    
    def hullPerimeter(self, idx: int):
        if(len(self.hll)>0):
            p = cv.arcLength(self.hll[idx],True) * self.pxlSize
        else:
            p = 0.0
        return p
    
    def hullCircularity(self, idx: int):
        if (len(self.hll)>0):
            c = 2.0*np.pi*self.hullRadius(idx)/self.hullPerimeter(idx)
        else:
            c = 0.0
        return c
    
    def hullIntensity(self, idx: int):
        if(len(self.hll)>0):
            intList = []
            # =============================================================================
            #  Create a mask image that contains the contour filled in
            # =============================================================================
            hull = self.hll[idx]
            cimg = np.zeros_like(self.rawImage)
            cv.drawContours(cimg, [hull], 0, color=255, thickness=-1)
            # =============================================================================
            # Access the image pixels and create a 1D numpy array then add to list
            # =============================================================================
            pts = np.where(cimg == 255)
            intList.append(self.rawImage[pts[0], pts[1]])
            intensity = np.sum(intList)
        else:
            intensity = 0
        return intensity
    
    def hullIntensityDensity(self, idx: int):
        if(len(self.hll)>0):
            intDen = self.hullIntensity(idx)/self.hullArea(idx)
        else:
            intDen = 0.0
        return intDen
    
    def solidity(self, idx: int):
        if (len(self.cnts)>0):
            s = self.contourArea(idx) / self.hullArea(idx)
        else:
            s= 0.0
        return s
    
    def area(self):
        area = 0.0
        for i in range(0, len(self.cnts), 1):
            area += cv.contourArea(self.cnts[i])
        return area * (self.pxlSize ** 2)
    
    def radius(self):
        r = np.sqrt(self.area()/np.pi) * self.pxlSize
        return r
    
    def perimeter(self):
        p = 0.0
        for i in range(0, len(self.cnts),1):
            p += cv.arcLength(self.cnts[i],True)
        return p * self.pxlSize
    
    
    def dataFrame(self):
        featureList = ["thr",
                       "area", "radius", "perimeter",
                       "cntArea", "cntRadius", "cntPerimeter", "cntCircularity", "cntIntensity", "cntIntDen",
                       "hllArea", "hllRadius", "hllPerimeter", "hllCircularity", "hllIntensity", "hllIntDen",
                       "solidity",
                       "contours"]
        df = pd.DataFrame(0, index = np.arange(255), columns = featureList)
        for i in range(1,256,1):
            self.thr = i
            self.detectContours()
            self.calcConvexHull()
            line = [self.thr,
                    self.area(),
                    self.radius(),
                    self.perimeter(),
                    self.contourArea(-1),
                    self.contourRadius(-1),
                    self.contourPerimeter(-1),
                    self.contourCircularity(-1),
                    self.contourIntensity(-1),
                    self.contourIntensityDensity(-1),
                    self.hullArea(-1),
                    self.hullRadius(-1),
                    self.hullPerimeter(-1),
                    self.hullCircularity(-1),
                    self.hullIntensity(-1),
                    self.hullIntensityDensity(-1),
                    self.solidity(-1),
                    self.countContours()]
            df.loc[i-1,:] = line
        return df        
    
    def loadImage(self, fileName: str):
        image = cv.imread(fileName)
        self.rawImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        
    def writeRawImage(self):
        fileName = str("%s/%s_raw.tiff" %(self.folderName, self.baseName))
        cv.imwrite(fileName, self.rawImage)
        
    def writeBinImage(self):
        fileName = str("%s/%s_bin_thr-%s.tiff" %(self.folderName, self.baseName, str(self.thr)))
        cv.imwrite(fileName, self.binImage)
        
    def writeContourImage(self):
        # =============================================================================
        # darw lagrgest contour
        # =============================================================================
        largestCnt = self.cnts[-1]
        image = cv.drawContours(cv.cvtColor(self.rawImage, cv.COLOR_GRAY2RGB), [largestCnt], 0, (1,255,1), 3)
        fileName = str("%s/%s_cnt_thr-%s.tiff" %(self.folderName, self.baseName, str(self.thr)))
        cv.imwrite(fileName, image)
        
    def writeHullImage(self):
        # =============================================================================
        # darw lagrgest hull
        # =============================================================================
        largestHull = self.hll[-1]
        image = cv.drawContours(cv.cvtColor(self.rawImage, cv.COLOR_GRAY2RGB), [largestHull], 0, (1,255,1), 3)
        fileName = str("%s/%s_hull_thr-%s.tiff" %(self.folderName, self.baseName, str(self.thr)))
        cv.imwrite(fileName, image)
        
def main():
    imData = np.zeros([10,10])
    ip = ImageProcessor()
    ip.fileName = "test.png"
    ip.data = imData
    ip.writeImage()
    
     
if __name__ == '__main__':
    main()