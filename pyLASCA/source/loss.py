#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 11:01:52 2020

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
from .roiContour import RoiContour


from scipy.spatial import distance
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error, normalized_root_mse, peak_signal_noise_ratio
import math

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