#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 14:50:32 2020

@author: malkusch
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

def auto_canny(image, sigma = 0.35):
    # compute the mediam of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    print(lower)
    print(upper)
    edged = cv2.Canny(image, lower, upper)
    # return edged image
    return(edged)

def plot_image(image):
    plt.imshow(image, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

def find_contours(image):
    img1 = 255 - image
    img2 = cv2.GaussianBlur(img1, (3, 3), 0.35)
    #img2 = cv2.bilateralFilter(img1, 3, 50, 50) 
    img3 = auto_canny(img2)
    img_out, contours, hierarchy = cv2.findContours(img3, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnts = sorted(contours, key=lambda x: cv2.contourArea(x))
    return(cnts)

def enhanceContourContrats(image, contours):
    img1 = cv2.fillPoly(image, pts = contours, color=(255,255,255))
    img = cv2.GaussianBlur(img1, (3, 3), 0.35)
    #ret, im2 = cv.threshold(im1, self.thr, 255, cv.THRESH_BINARY)
    img2 = cv2.dilate(img1, None, iterations = 10)
    img_out = cv2.erode(img2, None, iterations = 10)
    return(img_out)

def convexHull(contours):
    hll = []
    for i in range(0,len(contours),1):
        hll.append(cv2.convexHull(contours[i], False))
    return(hll)

def drawContours(image, cnt):
    image_out = cv2.drawContours(image, cnt, -1, (0,255,0), 3)
    # image_out = cv2.polylines(image, cnt, isClosed=True, color = (0,255,0))
    return(image_out)

fileName = "Arm_photo.bmp"
img = cv2.imread(fileName, cv2.IMREAD_COLOR)
plot_image(img)
#img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img2 = img[:,:,2]
contours = find_contours(img2)
print(len(contours))
img3 = enhanceContourContrats(img, contours)
contours = find_contours(img3)
print(len(contours))
hll = convexHull(contours)

img_out = drawContours(img, contours)
plot_image(img_out)
# =============================================================================
# 
# for i in range(0,len(contours),1):
#     cnt = contours[i]
#     img_out = drawContours(img, cnt)
#     plot_image(img_out)
# =============================================================================


# =============================================================================
# imgOut = cv2.Canny(img4, 100, 180, 3)
# =============================================================================


# =============================================================================
#         im3 = cv.dilate(im2, None, iterations = 1)
#         self._binImage = cv.erode(im3, None, iterations = 1)
# =============================================================================

# =============================================================================
# img6, contours, hierarchy = cv2.findContours(img5, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# print(contours)
# =============================================================================
# =============================================================================
# img = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)
# #img2 = 255-img
# 
# kernel = np.ones((10,10),np.float32)/10
# img3 = cv2.filter2D(img,-1,kernel)
# =============================================================================

# =============================================================================
# img3 = cv2.GaussianBlur(img2, (5, 5), 0)
# =============================================================================
# =============================================================================
# threshold, img3 = cv2.threshold(img2, 100, 150, cv2.THRESH_BINARY)
# img3 = np.divide(img2, 255)
# img4 = np.multiply(img, img3)
# =============================================================================

# =============================================================================
# output = img.copy()
# 
# circles = cv2.HoughCircles(img4, cv2.HOUGH_GRADIENT, 0.1, 14, param1=0.1, param2=20, minRadius=7, maxRadius=20)
# print(circles)
# 
# # ensure at least some circles were found
# if circles is not None:
# 	# convert the (x, y) coordinates and radius of the circles to integers
# 	circles = np.round(circles[0, :]).astype("int")
# 	# loop over the (x, y) coordinates and radius of the circles
# 	for (x, y, r) in circles:
# 		# draw the circle in the output image, then draw a rectangle
# 		# corresponding to the center of the circle
# 		cv2.circle(output, (x, y), r, (0, 255, 0), 4)
# 		cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
# =============================================================================


