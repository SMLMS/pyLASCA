#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 11:15:29 2019

@author: malkusch
"""
import numpy as np
import cv2 as cv

filename = "/home/malkusch/PowerFolders/pharmacology/Daten/ristow/test_Image_191210.png"
folderName = "/home/malkusch/PowerFolders/pharmacology/Daten/ristow"
baseName = "test_Image_191210"
suffix = ".tiff"
n = 240
oFilename = str("%s/%s_%s%s" %(folderName,
                               baseName,
                               str(n),
                               suffix))

# =============================================================================
# Import image
# convert to gray scale
# smooth slightly wit gaussian kenrel
# =============================================================================
im1 = cv.imread(filename)
im2 = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
im3 = cv.GaussianBlur(im2,(9,9), 0)

# =============================================================================
# create binary image
# dilate and erode (close)
# =============================================================================
ret, im4 = cv.threshold(im3, n, 255, cv.THRESH_BINARY)
im5 = cv.dilate(im4, None, iterations = 1)
im6 = cv.erode(im5, None, iterations = 1)

# =============================================================================
# detect contours
# =============================================================================
print(type(im6))
print(im6.shape)
im7, contours, hierarchy = cv.findContours(im6, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

# =============================================================================
# measure Area and sort contours by area
# =============================================================================
cntsSorted = sorted(contours, key=lambda x: cv.contourArea(x))

# =============================================================================
# darw lagrgest contour
# =============================================================================
cnt = cntsSorted[-1]
im8 = cv.drawContours(im1, [cnt], 0, (0,255,0), 3)

cv.imwrite(oFilename, im8)