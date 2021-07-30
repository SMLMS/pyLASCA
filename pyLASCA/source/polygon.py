#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 08:49:28 2020

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


from scipy.spatial import distance
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error, normalized_root_mse, peak_signal_noise_ratio
import math


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
            self._contour = cv.convexHull(contours[-1], False)
            #self._contour = contours[-1]
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
        self._image_bin = ndimage.binary_dilation(im_bin, iterations=iterations).astype(int)
        self._image_roi = np.multiply(self.image_raw, self.image_bin)

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
        image = cv.drawContours(cv.cvtColor(self.image_raw, cv.COLOR_GRAY2RGB), [self.contour], 0, (1,65535,65535), 3)
        image = cv.circle(image, (int(self.centroid_bin[0,1]), int(self.centroid_bin[0,0])), radius = 10, color = (1,65535,65535), thickness = 3)
        image = cv.circle(image, (int(self.centroid_roi[0,1]), int(self.centroid_roi[0,0])), radius = 10, color = (65535,1,65535), thickness = 3)

        plt.imshow(image)
        plt.xticks([]), plt.yticks([])
        plt.title("Contour image")
        plt.show()
        

class Loss(object):
    
    def __init__(self, obj: RoiContour):
        self._keep_track = False
        self._roi_usr = RoiContour
        self._roi_thr = RoiContour
        self._thr_abs = int()
        self._mse_metric = float()
        self._nrmse_metric = float()
        self._pnsr_metric = float()
        self._ssim_metric = float()
        self._agreement_metric = float()
        self._overlap_metric = float()
        self._distance_metric = float()
        self._loss_value = float()
        self._df_metrics = pd.DataFrame
        self.initLossWithRoi(obj)
        
    def __str__(self):
        message = str("the deteced loss is identified as a %s" %(self.loss))
        return(message)
        
    def __del__(self):
        message = str("Instance of Loss removed form heap")
        print(message)
    
    @property
    def roi_usr(self) -> RoiContour:
        return(cp.deepcopy(self._roi_usr))
    
    @property
    def roi_thr(self) -> RoiContour:
        return(cp.deepcopy(self._roi_thr))
    
    @property
    def thr_abs(self) -> int:
        return(self._thr_abs)
    
    @property
    def mse_metric(self)  -> float:
        return(self._mse_metric)
    
    @property
    def nrmse_metric(self)  -> float:
        return(self._nrmse_metric)
    
    @property
    def pnsr_metric(self)  -> float:
        return(self._pnsr_metric)
    
    @property
    def ssim_metric(self)  -> float:
        return(self._ssim_metric)
    
    @property
    def agreement_metric(self) -> float:
        return(self._agreement_metric)
    
    @property
    def overlap_metric(self) -> float:
        return(self._overlap_metric)
    
    @property
    def distance_metric(self) -> float:
        return(self._distance_metric)
    
    @property
    def loss_value(self)  -> float:
        return(self._loss_value)
    
    @property
    def df_metrics(self)  -> pd.DataFrame:
        return(self._df_metrics.copy())
    
    def initLossWithRoi(self, obj:RoiContour):
        self._roi_usr = cp.deepcopy(obj)
        self._roi_usr.analyze_roi()
        self._roi_thr = cp.deepcopy(obj)
        
    def loss_function(self, thr: float):
        ####################################################
        # define threshold and calculate thr based contour #
        ####################################################
        self._thr_abs = int(thr[0])
        self._roi_thr.thr_abs = thr[0]
        try:
            self._roi_thr.detectContoursByThr()
            self._roi_thr.analyze_roi()
        except:
            self._mse_metric = 0
            self._nrmse_metric = 0
            self._pnsr_metric = math.inf
            self._ssim_metric = math.inf
            self._agreement_metric = 0
            self._overlap_metric = math.inf
            self._loss_value = math.inf
            if(self._keep_track):
                self._df_metrics = self._df_metrics.append({'thr': thr[0],
                                                            'mse': self.mse_metric,
                                                            'nrmse': self.nrmse_metric,
                                                            'pnsr': self.pnsr_metric,
                                                            'ssim': self.ssim_metric,
                                                            'agreement': self.agreement_metric,
                                                            'overlap': self.overlap_metric,
                                                            'distance': self.distance_metric,
                                                            'loss': self.loss_value},
                                                           ignore_index=True)
            return(self.loss_value)
            
        
        ###################
        # compare results #
        ###################
        im_thr = self._roi_thr.image_roi
        im_usr = self._roi_usr.image_roi
        centroid_usr = self._roi_usr.centroid_roi
        centroid_thr = self._roi_thr.centroid_roi
        
        
        self._mse_metric = mean_squared_error(im_usr, im_thr)
        self._nrmse_metric = normalized_root_mse(im_usr, im_thr, normalization = 'euclidean')
        self._pnsr_metric = peak_signal_noise_ratio(im_usr, im_thr, data_range = 2**1-1)   
        self._ssim_metric = ssim(im_usr, im_thr, data_range= 2**1-1)
        self._agreement_metric = np.sum(im_usr == im_thr) / im_usr.size
        area_overlap = np.sum(np.multiply(self._roi_usr.image_bin, self._roi_thr.image_bin))
        area_usr = np.pi * self._roi_usr.area_r**2
        area_thr = np.pi * self._roi_thr.area_r**2
        self._overlap_metric = (area_usr +area_thr - 2 * area_overlap)/(area_usr +area_thr)
        self._distance_metric = 1.0 - ( 1.0 / ( 1.0 + distance.euclidean(centroid_usr, centroid_thr)))
        ##################
        # claculate loss #
        ##################
        self._loss_value = (self.nrmse_metric * self.overlap_metric) / (self.ssim_metric * self.agreement_metric)

        ##############
        # keep track #
        ##############
        if(self._keep_track):
            self._df_metrics = self._df_metrics.append({'thr': thr[0],
                                                        'mse': self.mse_metric,
                                                        'nrmse': self.nrmse_metric,
                                                        'pnsr': self.pnsr_metric,
                                                        'ssim': self.ssim_metric,
                                                        'agreement': self.agreement_metric,
                                                        'overlap': self.overlap_metric,
                                                        'distance': self.distance_metric,
                                                        'loss': self.loss_value},
                                                       ignore_index=True)
        return(self.loss_value)
    
    def minimize_brute(self):
        ################
        # define range #
        ################
        min_value = self._roi_thr.image_raw.min()
        max_value = self._roi_thr.image_raw.max()
        
        #######################
        # define fit function #
        #######################
        self._keep_track = True
        self._df_metrics = pd.DataFrame({'thr': [math.nan],
                                         'mse': [math.nan],
                                         'nrmse': [math.nan],
                                         'pnsr': [math.nan],
                                         'ssim': [math.nan],
                                         'agreement': [math.nan],
                                         'overlap': [math.nan],
                                         'distance': [math.nan],
                                         'loss': [math.nan]})
        result = brute(func = self.loss_function,
                       ranges = (slice(min_value, max_value, 1),)
                       )
        
        ###############
        # log results #
        ###############
        self._keep_track = False
        self._df_metrics = self._df_metrics.sort_values(by='thr', ignore_index = True)
        self.loss_function(thr = result)
    
    def plot_minimization_process(self):
        self._df_metrics.plot(x="thr", y="loss", kind = "line")
        plt.axvline(x= self.thr_abs, linestyle='dashed', label = 'threshold')
        plt.show()
        plt.close()
        
    def plot_result(self):
        # =============================================================================
        # darw lagrgest contour
        # =============================================================================
        image = self._roi_thr.image_raw
        image = cv.drawContours(cv.cvtColor(image, cv.COLOR_GRAY2RGB), [self._roi_usr.contour], 0, (1,65535,65535), 3)
        image = cv.circle(image, (int(self._roi_usr.centroid_roi[0,1]), int(self._roi_usr.centroid_roi[0,0])), radius = 10, color = (1,65535,65535), thickness = 3)
        
        
        image = cv.drawContours(image, [self._roi_thr.contour], 0, (65535,1,65535), 3)
        image = cv.circle(image, (int(self._roi_thr.centroid_roi[0,1]), int(self._roi_usr.centroid_roi[0,0])), radius = 10, color = (65535,0,65535), thickness = 3)
        
        plt.imshow(image)
        plt.xticks([]), plt.yticks([])
        plt.title("Contour image")
        plt.show()
        plt.close()
    

def main():
    imgName = "/Users/malkusch/PowerFolders/LaSCA/rawData/SCTCAPS01/SCTCAPS01_1.30_frame0.tiff"
    img1 = cv.imread(imgName, cv.IMREAD_ANYDEPTH)
    
    coordName = "/Users/malkusch/PowerFolders/LaSCA/mechanic/SCTCAPS 01/Tag 1/SCTCAPS 01 #1 30min (Colour)_transformed_200612.csv"
    df_coord = pd.read_csv(coordName)
    #df_coord = df_coord[df_coord["effect"] == "sec_hyperalgesia"]
    df_coord = df_coord[df_coord["effect"] == "allodynia"]
    contour_usr = np.zeros([len(df_coord),1,2], dtype = int)
    contour_usr[:,0,0] = np.round(df_coord["x_img_[pxl]"].values).astype(int)
    contour_usr[:,0,1] = np.round(df_coord["y_img_[pxl]"].values).astype(int)

    pxl_size = 0.24
    median = 53
    sigma = 14.7
    roi_thr = RoiContour()
    
    roi_usr = RoiContour()
    roi_usr.image_raw = img1
    roi_usr.contour = contour_usr
    roi_usr.analyze_roi()

    
    loss = Loss(obj = roi_usr)
    loss.minimize_brute()
    loss.plot_minimization_process()
    loss.plot_result()
    
    loss.df_metrics.plot(x="thr", y="nrmse", kind = "line")
    plt.axvline(x= loss.thr_abs, linestyle='dashed', label = 'threshold')
    plt.show()
    plt.close()
    
    loss.df_metrics.plot(x="thr", y="pnsr", kind = "line")
    plt.axvline(x= loss.thr_abs, linestyle='dashed', label = 'threshold')
    plt.show()
    plt.close()
    
    loss.df_metrics.plot(x="thr", y="ssim", kind = "line")
    plt.axvline(x= loss.thr_abs, linestyle='dashed', label = 'threshold')
    plt.show()
    plt.close()
    
    loss.df_metrics.plot(x="thr", y="agreement", kind = "line")
    plt.axvline(x= loss.thr_abs, linestyle='dashed', label = 'threshold')
    plt.show()
    plt.close()
    
    loss.df_metrics.plot(x="thr", y="overlap", kind = "line")
    plt.axvline(x= loss.thr_abs, linestyle='dashed', label = 'threshold')
    plt.show()
    plt.close()
    
    loss.df_metrics.plot(x="thr", y="distance", kind = "line")
    plt.axvline(x= loss.thr_abs, linestyle='dashed', label = 'threshold')
    plt.show()
    plt.close()


    

if __name__ == '__main__':
    main()