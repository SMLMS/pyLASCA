#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 14:04:11 2019

@author: malkusch
"""

import numpy as np
from scipy.signal import gaussian, fftconvolve
import cv2 as cv
import pandas as pd
from matplotlib import pyplot as plt

class FRC:
    def __init__(self):
        self._x = int()
        self._y = int()
        self._pxlSize = float()
        self._thr = 1.0/7.0
        self._image = list()
        self._dft = list()
        self._df_frc = pd.DataFrame()
        self._intercect = float()
        self._resolution = float()
        
    @property
    def x(self):
        return(self._x)
    
    @property
    def y(self):
        return(self._y)
    
    @property
    def pxlSize(self):
        return(self._pxlSize)
    
    @pxlSize.setter
    def pxlSize(self, value: float):
        self._pxlSize = value
    
    @property
    def thr(self):
        return(self._thr)
    
    @property
    def image(self):
        return(self._image)
    
    @property
    def dft(self):
        return(self._dft)
    
    @property
    def df_frc(self):
        return(self._df_frc)
    
    @property
    def intersect(self):
        return(self._intersect)
    
    @property
    def resolution(self):
        return(self._resolution)
    
    def setImages(self, img1: np.ndarray, img2: np.ndarray):
        self._image.clear()
        self._image.append(cv.normalize(src=img1, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1))
        self._image.append(cv.normalize(src=img2, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1))
        self._y , self._x = np.shape(self.image[0])
        self.createMagnitudeSpectrum()
        self.runAnalysis()
        
    def createMagnitudeSpectrum(self):
# =============================================================================
#         img_y , img_x = np.shape(self.image[0])
#         dft_x =  cv.getOptimalDFTSize(img_x)
#         dft_y =  cv.getOptimalDFTSize(img_y)
#         delta_x = dft_x - img_x
#         delta_y = dft_y - img_y
# =============================================================================
        self.dft.clear()
        for i in range(2):
# =============================================================================
#             nimg = cv.copyMakeBorder(self.image[i],
#                                      0, delta_x,
#                                      0, delta_y,
#                                      borderType = cv.BORDER_CONSTANT,
#                                      value = 0)
# =============================================================================
            
            f = np.fft.fft2(self.image[i])
# =============================================================================
#             f = np.fft.fft2(nimg)
# =============================================================================
            self._dft.append(np.fft.fftshift(f))
            #self._dft.append(20*np.log(np.abs(fshift)))
            
    def iteration(self, x: int, y: int):
        pxl1 = (self.dft[0][x,y])
        pxl2 = (self.dft[1][x,y])
        numer = np.vdot(pxl1, pxl2)
        denom1 = np.sum(np.abs(pxl1)**2)
        denom2 = np.sum(np.abs(pxl2)**2)
        return(numer.real/np.sqrt(denom1 * denom2))

    def determineResolution(self):
        delta = self.df_frc["frc_smoothed"] - self.thr
        sign = np.sign(delta)
        diff = np.diff(sign)
        diff_clean = np.nan_to_num(diff, 0)
        intercection_idx = np.argwhere(diff_clean)
        if (len(intercection_idx)>0):
            self._intersect = self.df_frc["omega"].values[intercection_idx[0]]
            self._resolution = 1.0/self.intersect
        else:
            self._intersect = np.nan
            self._resolution = np.nan
        
        

    def runAnalysis(self):
        
        center=np.zeros([1,2])
        center[0,0]=0.5*np.shape(self.dft[0])[1]
        center[0,1]=0.5*np.shape(self.dft[0])[0]
        rMax = int(np.ceil(np.min(center)))
        fs = np.fft.fftfreq(rMax, d=self.pxlSize)
        fsMax = np.fft.fftshift(fs)[-1]
        r = np.arange(rMax)
        omega = np.arange(start=0, stop=fsMax, step = fsMax/rMax)
        frc_data = np.zeros([rMax,1])
        thr_data = np.empty([rMax,1])
        thr_data.fill(self.thr)

        for i in range(rMax):
            image_blank = np.zeros(shape=[self.y, self.x], dtype=np.uint8)
            image_temp = cv.circle(img = image_blank, center = (int(center[0,0]), int(center[0,1])), radius = i, color=255, thickness = 1)
            coord = np.nonzero(image_temp)
            frc_data[i] = self.iteration(x=coord[0], y=coord[1])
             
        self._df_frc = pd.DataFrame({"r": r,
                                     "omega": omega,
                                     "thr": thr_data[:,0],
                                     "frc": frc_data[:,0]})
        
        kernel = gaussian(M=11, std=3.0, sym=True)
        kernel /= kernel.sum()
        frc_smoothed = fftconvolve(self.df_frc["frc"] , kernel, mode = 'valid')
        front = int(np.ceil((self.df_frc.shape[0] - len(frc_smoothed)) / 2.0))
        back = int(np.floor((self.df_frc.shape[0] - len(frc_smoothed)) / 2.0))
        frc_smoothed = np.pad(array = frc_smoothed, pad_width = (front, back), mode = 'constant', constant_values = np.nan)
        self._df_frc["frc_smoothed"] = frc_smoothed
        self.determineResolution()
         
    
     
    def plotImage(self):
        image1_gray = cv.normalize(src=self.image[0], dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
        image1_color = cv.applyColorMap(image1_gray, cv.COLORMAP_JET)
        image1_color = cv.cvtColor(image1_color,cv.COLOR_BGR2RGB)
        
        image2_gray = cv.normalize(src=self.image[1], dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
        image2_color = cv.applyColorMap(image2_gray, cv.COLORMAP_JET)
        image2_color = cv.cvtColor(image2_color,cv.COLOR_BGR2RGB)
        
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(image1_color)
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[0].set_title("FLUX image of frame %i" %(0))
        ax[1].imshow(image2_color)
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[1].set_title("FLUX image of frame %i" %(1))
        plt.show()
        
    def plotPowerSpectrum(self):
        image1 = 20*np.log(np.abs(self.dft[0]))
        image1_gray = cv.normalize(src=image1, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
        image1_color = cv.applyColorMap(image1_gray, cv.COLORMAP_COOL)
        image1_color = cv.cvtColor(image1_color,cv.COLOR_BGR2RGB)
        
        image2 = 20*np.log(np.abs(self.dft[1]))
        image2_gray = cv.normalize(src=image2, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
        image2_color = cv.applyColorMap(image2_gray, cv.COLORMAP_COOL)
        image2_color = cv.cvtColor(image2_color,cv.COLOR_BGR2RGB)
        
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(image1_color)
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[0].set_title("FFT of frame %i" %(0))
        ax[1].imshow(image2_color)
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[1].set_title("FFT of frame %i" %(1))
        plt.show()
        
    def plotFrcDist(self):
        vMax = 1.0
        vMin = min(self._df_frc["frc"].values)
        fig, ax = plt.subplots()
        ax.plot(self._df_frc["omega"], self._df_frc["frc"], color = 'b', label = "FRC")
        ax.plot(self._df_frc["omega"], self._df_frc["frc_smoothed"], color='r', label = "smoothed FRC")
        ax.plot(self._df_frc["omega"], self._df_frc["thr"], color='k', linestyle = '-', label = "Threshold" )
        ax.vlines(x = self.intersect, ymin = vMin, ymax = vMax, colors='k', linestyles = 'dashed', label = "Intersect")
        ax.set_title("FRC analysis")
        ax.set_xlabel("spatial frequency")
        ax.set_ylabel("correlation")
        ax.legend(title = str("Resolution = %.2f mm" %(self.resolution)), loc='upper right', shadow=True, fontsize='x-large')
        plt.show()

    def __str__(self):
        string = str("Resolution is %.2f mm" %(self.resolution))
        return string
    
    def __del__(self):
        message  = str("removed instance of FRC from heap.")
        print(message)        
        
        
def main():
    fileName1 = "/Users/malkusch/PowerFolders/pharmacology/Daten/ristow/SCTCAPS 01 #1 15min text (Flux)_frame-1_200423.tif"
    img1 = cv.imread(fileName1, cv.IMREAD_GRAYSCALE)
    fileName2 = "/Users/malkusch/PowerFolders/pharmacology/Daten/ristow/SCTCAPS 01 #1 15min text (Flux)_frame-2_200423.tif"
    img2 = cv.imread(fileName2, cv.IMREAD_GRAYSCALE)
    frc = FRC()
    frc.pxlSize = 0.24
    frc.setImages(img1, img2)
    frc.plotFrcDist()
    frc.plotPowerSpectrum()
    frc.plotImage()
    print(frc)
    
if __name__ == '__main__':
    main()
        
