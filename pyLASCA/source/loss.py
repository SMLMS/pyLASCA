#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 11:43:23 2020

@author: malkusch
"""

import numpy as np
import cv2 as cv
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import minimize, brute
from scipy import ndimage
from PIL import Image, ImageDraw
from shapely.geometry import Polygon
#from ..source import intersect
import intersect

class Loss(object):
    
    def __init__(self):
        self._intersect = intersect.Intersect()
        self._pxl_size = float()
        self._image_raw = np.zeros(0)
        self._contour_thr = np.ndarray
        self._contour_usr = np.ndarray
        self._centroid_thr = np.zeros(0)
        self._centroid_usr = np.zeros(0)
        self._median = int()
        self._sigma = float()
        self._thr_abs = int()
        self._alpha = float()
        self._image_bin = np.zeros(0)
        self._area_r_usr = float()
        self._area_r_thr = float()
        self._area_r_intersect= float()
        self._delta_area_radius = int()
        self._delta_dist = int()
        self._delta_overlay_radius = int()
        self._loss = float()
        
    def __str__(self):
        message = str("the deteced loss is identified as a %s" %(self.loss))
        return(message)
        
    def __del__(self):
        message = str("Instance of Loss removed form heap")
        print(message)
        
    @property
    def intersect(self) -> np.ndarray:
        return(self._intersect)
    
    @property
    def pxl_size(self) -> float:
        return(self._pxl_size)
    
    @pxl_size.setter
    def pxl_size(self, value: float):
        self._pxl_size = value
    
    @property
    def image_raw(self) -> np.ndarray:
        return(np.copy(self._image_raw))
    
    @image_raw.setter
    def image_raw(self, obj: np.ndarray):
        self._image_raw = np.copy(obj)
        
    @property
    def contour_thr(self) -> np.ndarray:
        return(np.copy(self._contour_thr))
    
    @property
    def contour_usr(self) -> np.ndarray:
        return(np.copy(self._contour_usr))
    
    @contour_usr.setter
    def contour_usr(self, obj: np.ndarray):
        self._contour_usr = np.copy(obj)

        
    @property
    def centroid_thr(self) -> np.ndarray:
        return(np.copy(self._centroid_thr))
    
    @property
    def centroid_usr(self) -> np.ndarray:
        return(np.copy(self._centroid_usr))
    
    @property
    def median(self) -> int:
        return(self._median)
    
    @median.setter
    def median(self, value: int):
        self._median = value
        
    @property
    def sigma(self) -> float:
        return(self._sigma)
    
    @sigma.setter
    def sigma(self, value: float):
        self._sigma = value
    
    @property
    def thr_abs(self) -> int:
        return(self._thr_abs)
    
    @thr_abs.setter
    def thr_abs(self, value: float):
        self._thr_abs = value
        self._alpha = (self.thr_abs - self.median)/self.sigma
    
    @property
    def alpha(self) -> float:
        return(self._alpha)
    
    @alpha.setter
    def alpha(self, value: float):
        self._alpha = value
        self._thr_abs = (self.alpha * self.sigma) + self.median
        
    @property
    def image_bin(self) -> np.ndarray:
        return(np.copy(self._image_bin))
    
    @property
    def delta_area_radius(self) -> int:
        return(self._delta_area_radius)
    
    @property
    def delta_dist(self) -> int:
        return(self._delta_dist)
    
    @property
    def delta_overlay_radius(self) -> int:
        return(self._delta_overlay_radius)
    
    @property
    def loss(self) -> float:
        return(self._loss)
    
    @property
    def area_r_usr(self) -> float:
        return(self._area_r_usr)
    
    @property
    def area_r_thr(self) -> float:
        return(self._area_r_thr)
    
    @property
    def area_r_intersect(self) -> float:
        return(self._area_r_intersect)
    
    def detectContours(self):
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
        self._image_bin = cv.erode(im4, None, iterations = 1)
        # =============================================================================
        # detect contours
        # =============================================================================
        im5, contours, hierarchy = cv.findContours(self.image_bin, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        # =============================================================================
        # sort contours by area
        # =============================================================================
        contours = sorted(contours, key=lambda x: cv.contourArea(x))
        # =============================================================================
        # clear array for convex hull points
        # =============================================================================
        if not(len(contours) < 1):
            self._contour_thr = cv.convexHull(contours[-1], False)
        else:
            self._contour_thr = (np.nan,)
            
    def analyze_contours(self):
        height, width = self.image_raw.shape
        dark_img = Image.new('L', (width, height), 0)
        
        # analyze thr based contours
        thr_bin = dark_img.copy()
        poly_thr =list()
        for i in range(np.shape(self.contour_thr)[0]):
            poly_thr.append(tuple((self.contour_thr[i,0,0], self.contour_thr[i,0,1])))
        ImageDraw.Draw(thr_bin).polygon(poly_thr, outline=1, fill=1)
        label_img_thr, thr_labels = ndimage.label(thr_bin)
        max_thr_idx = np.argmax(ndimage.sum(thr_bin, label_img_thr, range(thr_labels + 1)))
        area_thr = ndimage.sum(thr_bin, label_img_thr, range(thr_labels + 1))[max_thr_idx] * (self.pxl_size**2)
        self._area_r_thr = np.sqrt(area_thr / np.pi)
        
        mask_thr = np.array(thr_bin)
        img_thr = np.multiply(mask_thr, self.image_raw)
        coord_thr_pxl = np.array(ndimage.center_of_mass(img_thr,label_img_thr, range(thr_labels+1)))[[max_thr_idx]]
        self._centroid_thr = np.multiply(coord_thr_pxl, self.pxl_size)
        
        # analyze usr based contours
        usr_bin = dark_img.copy()
        poly_usr =list()
        for i in range(np.shape(self.contour_usr)[0]):
            poly_usr.append(tuple((self.contour_usr[i,0,0], self.contour_usr[i,0,1])))
        ImageDraw.Draw(usr_bin).polygon(poly_usr, outline=1, fill=1)
        label_img_usr, usr_labels = ndimage.label(usr_bin)
        max_usr_idx = np.argmax(ndimage.sum(usr_bin, label_img_usr, range(usr_labels + 1)))
        area_usr = ndimage.sum(usr_bin, label_img_usr, range(usr_labels + 1))[max_usr_idx] * (self.pxl_size**2)
        self._area_r_usr = np.sqrt(area_usr / np.pi)
        
        mask_usr = np.array(usr_bin)
        img_usr = np.multiply(mask_usr, self.image_raw)
        coord_usr_pxl = np.array(ndimage.center_of_mass(img_usr,label_img_usr, range(usr_labels+1)))[[max_usr_idx]]
        self._centroid_usr = np.multiply(coord_usr_pxl, self.pxl_size)
        
    def analyze_intersect(self):
        if ((np.isnan(self.centroid_usr).any()) or (np.isnan(self.contour_thr).any())):
            self._delta_overlay_radius = np.nan
            print("warnng: could not detect intersect")
            return()
        else:
            self._intersect.roi_1 = self.contour_usr
            self._intersect.roi_2 = self.contour_thr
            try:
                self._intersect.calculateIntresect()
            except:
                self._delta_overlay_radius = np.nan
                print("warnng: could not detect intersect")
                return()
            
        
        height, width = self.image_raw.shape
        dark_img = Image.new('L', (width, height), 0)
        
        # analyze intercept based contours
        inter_bin = dark_img.copy()
        poly_inter =list()
        for i in range(np.shape(self.intersect.roi_intersect)[0]):
            poly_inter.append(tuple((self.intersect.roi_intersect[i,0,0], self.intersect.roi_intersect[i,0,1])))
        ImageDraw.Draw(inter_bin).polygon(poly_inter, outline=1, fill=1)
        label_img_inter, inter_labels = ndimage.label(inter_bin)
        max_inter_idx = np.argmax(ndimage.sum(inter_bin, label_img_inter, range(inter_labels + 1)))
        area_inter = ndimage.sum(inter_bin, label_img_inter, range(inter_labels + 1))[max_inter_idx] * (self.pxl_size**2)
        self._area_r_intersect = np.sqrt(area_inter / np.pi)
        
        
    def calc_delta_dist(self):
        if ((np.isnan(self.centroid_usr).any()) or (np.isnan(self.contour_thr).any())):
            self._delta_dist = np.nan
        else:
            self._delta_dist = np.sqrt(((self.centroid_thr[0,0] - self.centroid_usr[0,0])**2) + ((self.centroid_thr[0,1] - self.centroid_usr[0,1])**2))
        
    def calc_delta_area_radius(self):
        if ((np.isnan(self.centroid_usr).any()) or (np.isnan(self.contour_thr).any())):
            self._delta_area_radius = np.nan
        else:
            self._delta_area_radius = np.sqrt(np.abs(self.area_r_usr - self.area_r_thr))
        
    def calc_delta_overlay_radius(self):
        if (np.isnan(self.area_r_intersect) or np.isnan(self.area_r_thr) or np.isnan(self.area_r_usr)):
            self._delta_overlay_radius = np.nan
        else:
            overlay_area = np.pi*self.area_r_thr**2 + np.pi*self.area_r_usr**2 - 2*(np.pi*self.area_r_intersect**2)
            self._delta_overlay_radius = np.sqrt(overlay_area / np.pi)
            #self._delta_overlay_radius = 2*(np.pi*self.area_r_intersect**2) / (np.pi*self.area_r_thr**2 + np.pi*self.area_r_usr**2)

        
    def loss_function(self, alpha: float):
        #self.thr_abs = thr
        #self.alpha = (self.thr_abs - self.median) / self.sigma
        
        self.alpha = alpha
        #self.thr_abs = (self.alpha * self.sigma) + self.median
        self.detectContours()
        self.analyze_contours()
        self.analyze_intersect()
        self.calc_delta_dist()
        self.calc_delta_area_radius()
        self.calc_delta_overlay_radius()
        if ((np.isnan(self.centroid_usr).any()) or (np.isnan(self.contour_thr).any())):
            self._loss = np.nan
        else:
            #self._loss = np.sqrt(self.delta_dist**2 + self.delta_area_radius**2 + self.delta_overlay_radius**2)
            self._loss = self.delta_dist * self.delta_overlay_radius * self.delta_area_radius
            
        return(self.loss)
        
        
    def fit(self, init_thr: int):
        result = minimize(fun = self.loss_function,
                      x0 = init_thr,
                      method='Nelder-Mead')
        self.loss_function(thr = result.x[0])
        
    def fit_brute(self):
        #min_val = self.median
        #max_val = np.abs(self.median + self.sigma)
        min_val = -1.0
        max_val = 2.0

        
        result = brute(func = self.loss_function,
                       ranges = (slice(min_val, max_val, 0.01),)
                       )
        
        self.loss_function(alpha = int(result[0]))
        return(result[0])
        
        
        
    def plot_fit_procedure(self):
# =============================================================================
#         min_val = self.median
#         max_val = np.abs(self.median + self.sigma)
#         thr_abs_array = np.arange(start = min_val, stop = max_val, step = 1, dtype = int)
#         alpha_array = (thr_abs_array - self.median)/self.sigma
# =============================================================================
        
        
        min_val  = -1.0
        max_val = 2.0
        alpha_array = np.arange(start = min_val, stop = max_val, step = 0.01, dtype = float)
        thr_abs_array = (alpha_array * self.sigma) + self.median
        
        delta_dist_array = np.zeros(len(thr_abs_array)) * np.nan
        delta_area_radius_array = np.zeros(len(thr_abs_array)) * np.nan
        delta_overlay_radius_array = np.zeros(len(thr_abs_array)) * np.nan
        loss_array = np.zeros(len(thr_abs_array)) * np.nan
        
        for i in range(len(alpha_array)):
            alpha = alpha_array[i]
            self.loss_function(alpha)
            delta_dist_array[i] = self.delta_dist
            delta_area_radius_array[i] = self.delta_area_radius
            delta_overlay_radius_array[i] = self.delta_overlay_radius
            loss_array[i] = self.loss
        
        df = pd.DataFrame({"thr_abs": thr_abs_array,
                           "alpha": alpha_array,
                           "dist_[mm]": delta_dist_array,
                           "area_r_[mm]": delta_area_radius_array,
                           "overlay_r_[mm]": delta_overlay_radius_array,
                           "loss_[mm]": loss_array
                           })
        
        df.plot.line(x='alpha', y='dist_[mm]')
        df.plot.line(x='alpha', y='area_r_[mm]')
        df.plot.line(x='alpha', y='overlay_r_[mm]')
        df.plot.line(x='alpha', y='loss_[mm]')
        


    def contourImage(self):
        # =============================================================================
        # darw lagrgest contour
        # =============================================================================
        image = cv.drawContours(cv.cvtColor(self.image_raw, cv.COLOR_GRAY2RGB), [self.contour_thr], 0, (1,65535,65535), 3)
        image = cv.circle(image, (int(self.centroid_thr[0,1] / self.pxl_size), int(self.centroid_thr[0,0] / self.pxl_size)), radius = 10, color = (1,65535,65535), thickness = 3)
        image = cv.drawContours(image, [self.contour_usr], 0, (65535,1,65535), 3)
        image = cv.circle(image, (int(self.centroid_usr[0,1] / self.pxl_size), int(self.centroid_usr[0,0] / self.pxl_size)), radius = 10, color = (65535,0,65535), thickness = 3)
        
        plt.imshow(image)
        plt.xticks([]), plt.yticks([])
        plt.title("Contour image")
        plt.show()
        
    def contourPlot(self):
        x1 = self.pxl_size * np.append(self.contour_thr[:,0,0], self.contour_thr[0,0,0])
        y1 = self.pxl_size * np.append(self.contour_thr[:,0,1], self.contour_thr[0,0,1])       
        plt.plot(x1, y1, color='cyan', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)
        
        x2 = self.pxl_size * np.append(self.contour_usr[:,0,0], self.contour_usr[0,0,0])
        y2 = self.pxl_size * np.append(self.contour_usr[:,0,1], self.contour_usr[0,0,1])       
        plt.plot(x2, y2, color='magenta', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)
        
        plt.title("Contour image")
        plt.show()
    

def main():
    imgName = "/Users/malkusch/PowerFolders/LaSCA/rawData/SCTCAPS01/SCTCAPS01_1.45_frame0.tiff"
    img1 = cv.imread(imgName, cv.IMREAD_ANYDEPTH)
    
    coordName = "/Users/malkusch/PowerFolders/LaSCA/mechanic/SCTCAPS 01/Tag 1/SCTCAPS 01 #1 45min (Colour)_transformed_200612.csv"
    df_coord = pd.read_csv(coordName)
    #df_coord = df_coord[df_coord["effect"] == "sec_hyperalgesia"]
    df_coord = df_coord[df_coord["effect"] == "allodynia"]
    contour_usr = np.zeros([len(df_coord),1,2], dtype = int)
    contour_usr[:,0,0] = np.round(df_coord["x_img_[pxl]"].values).astype(int)
    contour_usr[:,0,1] = np.round(df_coord["y_img_[pxl]"].values).astype(int)

    loss = Loss()
    loss.pxl_size = 0.24
    loss.median = 53
    loss.sigma = 14.7
    loss.image_raw = img1
    loss.contour_usr = contour_usr

    alpha = loss.fit_brute()
    #alpha = 1.03
    print(alpha)
    loss.loss_function(alpha)
    print(loss.alpha)
    print(loss.thr_abs)
    
    #loss.contourPlot()
    loss.contourImage()
    #loss.plot_fit_procedure()
    

if __name__ == '__main__':
    main()