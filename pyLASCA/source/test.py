#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 14:04:29 2020

@author: malkusch
"""

from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import random as rng
from skimage.morphology import skeletonize
rng.seed(12345)

def cannyEdge(self, sigma: float):
        # =============================================================================
        # apply automatic Canny edge detection using the computed median
        # =============================================================================
        v = np.median(src_gray)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        return(cv.Canny(src_gray, lower, upper))

def thresh_callback(val):
    threshold = val
    # Detect edges using Canny
    #canny_output = cv.Canny(src_gray, threshold, threshold * 2)
    canny_output = cannyEdge(src_gray, sigma = threshold)
    # Find contours
    _, contours, hierarchy = cv.findContours(canny_output, cv.RETR_EXTERNAL, 2)
    #_, contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # Draw contours
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv.drawContours(drawing, contours, i, color, 2, cv.LINE_8, hierarchy, 0)
    # Show in a window
    cv.imshow('Contours', drawing)
    
    binary =  np.zeros((canny_output.shape[0], canny_output.shape[1], 1), dtype=np.uint8)
    for i in range(len(contours)):
        #cv.fillConvexPoly(binary,contours[i],255,cv.LINE_8)
        cv.drawContours(binary, contours, i, 255, 1, cv.LINE_8, hierarchy, 0)
        
    img = cv.dilate(binary.copy(), None, iterations = 1)
    img = cv.erode(img, None, iterations = 1)
    
    img_floodfill = img.copy()

    h, w = img.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv.floodFill(img_floodfill, mask, (0,0), 255)
    img_floodfill= cv.bitwise_not(img_floodfill)
    
    
    img_bin = (img | img_floodfill)
# =============================================================================
#     img_bin = cv.dilate(img_bin, None, iterations = 1)
#     img_bin = cv.erode(img_bin, None, iterations = 1)
# =============================================================================
    
    
    
    img, contours2, hierarchy = cv.findContours(img_bin, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    hull = []
    for i in range(len(contours2)):
        hull.append(cv.convexHull(contours2[i], False))
    
    binary_hull =  np.zeros((canny_output.shape[0], canny_output.shape[1], 1), dtype=np.uint8)
    #binary_hull = src_gray.copy()
    for i in range(len(hull)):
        cv.drawContours(binary_hull, contours2, i, 255, 1, cv.LINE_8, hierarchy, 0)
        
    
    binary_hull = cv.dilate(binary_hull, None, iterations = 3)
   # binary_hull = cv.dilate(binary_hul, None, iterations = 1)
    #img_bin = cv.erode(img_bin, None, iterations = 1)
    
    out = skeletonize(binary_hull > 0)
    out = 255*(out.astype(np.uint8))

    
    minLineLength = 25
    maxLineGap = 0
    
    lines = cv.HoughLines(out,1,np.pi/180,minLineLength,maxLineGap)#[[1,20]]
    
    
    image_lines = src_gray.copy()
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
    
# Load source image
parser = argparse.ArgumentParser(description='Code for Finding contours in your image tutorial.')
parser.add_argument('--input', help='Path to input image.', default='HappyFish.jpg')
args = parser.parse_args()
src = cv.imread(args.input)
if src is None:
    print('Could not open or find the image:', args.input)
    exit(0)
# Convert image to gray and blur it
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
src_gray = src[:,:,2]
src_gray = cv.medianBlur(src_gray, 3)
src_gray = cv.blur(src_gray, (3,3))
# Create Window
source_window = 'Source'
cv.namedWindow(source_window)
cv.imshow(source_window, src)
max_thresh = 100
thresh = 5 # initial threshold
cv.createTrackbar('Canny Thresh:', source_window, thresh, max_thresh, thresh_callback)
thresh_callback(thresh)
cv.waitKey()