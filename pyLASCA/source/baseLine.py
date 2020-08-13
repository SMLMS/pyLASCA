#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 13:14:17 2020

@author: malkusch
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

class BaseLine(object):
    def __init__(self):
        self._image = np.zeros(0)
        self._mean_gray_value = float()
        self._median_gray_value = float()
        self._std_gray_value = float()
        self._min_gray_value = int()
        self._max_gray_value = int()
        self._int_pdf = tuple()
        
    def __str__(self):
        message = str("Instance of class BaseLine\n")
        message += str("mean: %.3f\n" %(self.mean_gray_value))
        message += str("median: %.3f\n" %(self.median_gray_value))
        message += str("std: %.3f\n" %(self.std_gray_value))
        message += str("min: %i\n" %(self.min_gray_value))
        message += str("max: %i\n" %(self.max_gray_value))
        return(message)
    
    def __del__(self):
        message = str("Instance if class BaseLine removed from heap.")
        print(message)
        
    @property
    def image(self) -> np.ndarray:
        return(np.copy(self._image))
    
    @image.setter
    def image(self, obj: np.ndarray):
        self._image = np.copy(obj)
        
    @property
    def mean_gray_value(self) -> float:
        return(self._mean_gray_value)
    
    @property
    def median_gray_value(self) -> float:
        return(self._median_gray_value)
    
    @property
    def std_gray_value(self) -> float:
        return(self._std_gray_value)
    
    @property
    def min_gray_value(self) -> int:
        return(self._min_gray_value)
    
    @property
    def max_gray_value(self) -> int:
        return(self._max_gray_value)
    
    @property
    def int_pdf(self) -> tuple:
        return(self._int_pdf)
    
    def fit(self):
        # =============================================================================
        # remove dead pxl by convolution with median filter
        # =============================================================================
        im1 = cv.medianBlur(src = self.image, ksize = 3)
        pxl_values = im1[im1>0]
        self._mean_gray_value = np.mean(pxl_values)
        self._median_gray_value = np.median(pxl_values)
        self._std_gray_value = np.std(pxl_values)
        self._min_gray_value = np.min(pxl_values)
        self._max_gray_value = np.max(pxl_values)
        self._int_pdf = np.histogram(a = pxl_values, bins = self.max_gray_value - self.min_gray_value, range = (self.min_gray_value, self.max_gray_value), density = True)
        
    def plot_int_pdf(self):
        fig = plt.figure()
        sp = fig.add_subplot(1,1,1)
        sp.bar(x = self.int_pdf[1][:-1], height=self.int_pdf[0])
        sp.set_title('PDF of image intensity')
        sp.set_xlabel('intensity [a.u.]')
        sp.set_ylabel('frequency')
        plt.show()
        
def main():
    imgName = "/Users/malkusch/PowerFolders/LaSCA/rawData/SCTCAPS01/SCTCAPS01_1.15_frame0.tiff"
    img1 = cv.imread(imgName, cv.IMREAD_ANYDEPTH)
    
    bl = BaseLine()
    bl.image = img1
    bl.fit()
    print(bl)
    bl.plot_int_pdf()
    
if __name__ == '__main__':
    main()
        
        