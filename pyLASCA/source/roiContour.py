#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 11:00:08 2020

@author: malkusch
"""

import numpy as np
import cv2 as cv
import pandas as pd
import copy as cp
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy.optimize import brute
from PIL import Image, ImageDraw


class RoiContour(object):
    
    def __init__(self):
        self._thr_abs = int()
        self._image_raw = np.zeros(0)
        self._image_bin = np.zeros(0)
        self._image_roi = np.zeros(0)
        self._contour = np.ndarray
        self._centroid_bin = np.zeros(0)
        self._centroid_roi = np.zeros(0)
        self._area_r = float()
        
    def __str__(self):
        message = str("the radius of the roi area is:  %.3f" %(self.area_r))
        return(message)
        
    def __del__(self):
        message = str("Instance of RoiContour removed form heap")
        print(message)
        
    @property
    def thr_abs(self) -> int:
        return(self._thr_abs)
    
    @thr_abs.setter
    def thr_abs(self, value: float):
        self._thr_abs = value      
    
    @property
    def image_raw(self) -> np.ndarray:
        return(np.copy(self._image_raw))
    
    @image_raw.setter
    def image_raw(self, obj: np.ndarray):
        self._image_raw = np.copy(obj)
    
    @property
    def image_bin(self) -> np.ndarray:
        return(np.copy(self._image_bin))

    @property
    def image_roi(self) -> np.ndarray:
        return(np.copy(self._image_roi))    
    
    @property
    def contour(self) -> np.ndarray:
        return(np.copy(self._contour))
    
    @contour.setter
    def contour(self, obj: np.ndarray):
        self._contour = np.copy(obj)
        
    @property
    def centroid_bin(self) -> np.ndarray:
        return(np.copy(self._centroid_bin))
    
    @property
    def centroid_roi(self) -> np.ndarray:
        return(np.copy(self._centroid_roi))
    
    @property
    def area_r(self) -> float:
        return(self._area_r)
    
    def relThr(self, alpha = 0.0, mu = 0.0, sigma = 0.0):
        self.thr_abs = (alpha * sigma) + mu
    
    def detectContoursByThr(self):
        # =============================================================================
        # remove dead pxl by convolution with median filter
        # =============================================================================
        im1 = cv.medianBlur(src = self.image_raw, ksize = 3)
        # =============================================================================
        # create binary image
        # dilate and erode (close)
        # =============================================================================
        ret, im2 = cv.threshold(im1, self.thr_abs, 255, cv.THRESH_BINARY)
        im3 = im2.astype('uint8')
        im4 = cv.dilate(im3, None, iterations = 1)
        im_bin = cv.erode(im4, None, iterations = 1)
        # =============================================================================
        # detect contours
        # =============================================================================
        im5, contours, hierarchy = cv.findContours(im_bin, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        # =============================================================================
        # sort contours by area
        # =============================================================================
        contours = sorted(contours, key=lambda x: cv.contourArea(x))
        # =============================================================================
        # clear array for convex hull points
        # =============================================================================
        if not(len(contours) < 1):
            #self._contour = cv.convexHull(contours[-1], False)
            self._contour = contours[-1]
        else:
            self._contour = (np.nan,)
            
    def analyze_roi(self, iterations=3):
        height, width = self.image_raw.shape
        im_bin = Image.new('L', (width, height), 0)
        equality_sum = np.sum(np.all(self.contour == self.contour[0], axis=1))
        coord_sum = np.prod(np.shape(self.contour))
        roi = list()
        for i in range(np.shape(self.contour)[0]):
            roi.append(tuple((self.contour[i,0,0], self.contour[i,0,1])))
        if(equality_sum != coord_sum):
            ImageDraw.Draw(im_bin).polygon(roi, outline=1, fill=1)
        else:
            ImageDraw.Draw(im_bin).point(roi, fill = 1)
        self._image_bin = ndimage.binary_dilation(im_bin, iterations=iterations).astype(np.uint16)
        self._image_roi = np.multiply(self.image_raw, self.image_bin).astype(np.uint16)

        label_img_roi, roi_labels = ndimage.label(self.image_bin)
        area_roi = ndimage.sum(self.image_bin, label_img_roi, range(roi_labels + 1))
        max_roi_idx = np.argmax(area_roi)
        self._area_r = np.sqrt(area_roi[max_roi_idx] / np.pi)
        self._centroid_bin = np.array(ndimage.center_of_mass(self.image_bin,label_img_roi, range(roi_labels+1)))[[max_roi_idx]]
        self._centroid_roi = np.array(ndimage.center_of_mass(self.image_raw,label_img_roi, range(roi_labels+1)))[[max_roi_idx]]


    def contourImage(self):
        # =============================================================================
        # darw lagrgest contour
        # =============================================================================
        image = cv.drawContours(cv.cvtColor(self.image_roi, cv.COLOR_GRAY2RGB), [self.contour], 0, (65535,1,65535), 3)
        #image = cv.circle(image, (int(self.centroid_bin[0,1]), int(self.centroid_bin[0,0])), radius = 10, color = (65535,1, 65535), thickness = 3)
        image = cv.circle(image, (int(self.centroid_roi[0,1]), int(self.centroid_roi[0,0])), radius = 10, color = (65535,1, 65535), thickness = 3)

        plt.imshow(image)
        plt.xticks([]), plt.yticks([])
        plt.title("Contour image")
        plt.show()
        