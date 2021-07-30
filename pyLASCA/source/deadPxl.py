#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 08:23:55 2020

@author: malkusch
"""

import numpy as np
import pandas as pd
import cv2 as cv
import os
from itertools import repeat
from matplotlib import pyplot as plt
from scipy.ndimage import median_filter

class DeadPxl(object):
    """DeadPxl
    
    Imports Lasca image.
    Identifies dead and hot pxl within the image by using a median filter.
    
     Attributes:
         img: A numpy array. Contains the raw data.
    """
    
    def __init__(self):
        """initializes an instance of the DeadPxl class"""
        self._img = np.ndarray
        self._thr = float
        self._edgeCorrection = True
        self._df_dead = pd.DataFrame
        
    def __str__(self):
        """prints informtions about the instance variables of the RawData class
        
        Returns:
            Compound string comprising the class' instance variables'
        """
        string = str("\nDeadPxl information\n")
        return(string)
               
    def __del__(self):
        """removes an instance of DeadPxl from the heap."""
        string = str("instance of DeadPxl removed from heap.")
        print(string)
        
    @property
    def img(self) -> np.ndarray:
        return(np.copy(self._img))
    
    @img.setter
    def img(self, obj: np.ndarray):
        self._img = np.copy(obj)
        
    @property
    def thr(self) -> float:
        return(self._thr)
    
    @property
    def edgeCorrection(self) -> bool:
        return(self._edgeCorrection)
    
    @property
    def df_dead(self) -> pd.DataFrame:
        return(self._df_dead)
        
    def detectDeadPxl(self):
# =============================================================================
#         img_blurred = median_filter(input = self.img, size = 2)
# =============================================================================
        img_blurred = cv.medianBlur(src = self.img, ksize = 3)

        img_delta = self.img.astype(int) - img_blurred.astype(int)

        self._thr = 6 * np.std(img_delta)

        
        dead_pxls = np.nonzero(np.abs(img_delta[1:-1,1:-1])>self.thr)
        dead_pxls = np.array(dead_pxls) + 1
        
        if(self.edgeCorrection):
            height,width = np.shape(self._img)
            # analyze pxls on left and right edges
            for idx in range(1,height-1):
                # left edge
                edge_blurred  = np.median(self._img[idx-1:idx+2,0:2])
                edge_delta = np.abs(self.img[idx,0].astype(int) - edge_blurred)
                if edge_delta > self.thr:
                    dead_pxls = np.hstack((dead_pxls, [[idx], [0]]))
                    
                # right edge
                edge_blurred  = np.median(self._img[idx-1:idx+2,-2:])
                edge_delta = np.abs(self.img[idx,-1].astype(int) - edge_blurred)
                if edge_delta > self.thr:
                    dead_pxls = np.hstack((dead_pxls, [[idx], [width-1]]))
                    
            # analyze pxls top and bottom corners
            for idx in range(1,width-1):
                # bottom edge
                edge_blurred  = np.median(self._img[0:2, idx-1:idx+2])
                edge_delta = np.abs(self.img[0, idx].astype(int) - edge_blurred)
                if edge_delta > self.thr:
                    dead_pxls = np.hstack((dead_pxls, [[0], [idx]]))
                    
                # top edge
                edge_blurred  = np.median(self._img[-2:, idx-1:idx+2])
                edge_delta = np.abs(self.img[-1, idx].astype(int) - edge_blurred)
                if edge_delta > self.thr:
                    dead_pxls = np.hstack((dead_pxls, [[height-1], [idx]]))
                    
            # analyze pxls in corners
            # bottom-left corner
            corner_blurred  = np.median(self._img[0:2,0:2])
            corner_delta = np.abs(self.img[0,0].astype(int) - corner_blurred)
            if corner_delta > self.thr:
                dead_pxls = np.hstack(( dead_pxls, [[0],[0]]  ))
                
            # bottom-right corner
            corner_blurred  = np.median(self._img[0:2,-2:])
            corner_delta = np.abs(self.img[0,-1].astype(int) - corner_blurred)
            if corner_delta > self.thr:
                dead_pxls = np.hstack(( dead_pxls, [[0],[width-1]]  ))
                
            # upper-left corner
            corner_blurred  = np.median(self._img[-2:,0:2])
            corner_delta = np.abs(self.img[-1,0].astype(int) - corner_blurred)
            if corner_delta > self.thr:
                dead_pxls = np.hstack(( dead_pxls, [[height-1],[0]]  ))
                
            # upper-right corner
            corner_blurred  = np.median(self._img[-2:,-2:])
            corner_delta = np.abs(self.img[-1,-1].astype(int) - corner_blurred)
            if corner_delta > self.thr:
                dead_pxls = np.hstack(( dead_pxls, [[height-1],[width-1]]  ))
                
        intensity = self.img[dead_pxls[0], dead_pxls[1]]
                
        self._df_dead = pd.DataFrame({"x": dead_pxls[0],
                                      "y": dead_pxls[1],
                                      "intensity": intensity})
        
        

        
        
    def detectHotPxl(self):
        hot_pxls = np.array(np.nonzero(self.img == 255))
        print(hot_pxls)
        print(np.min(self.img))
        print(np.max(self.img))
        
    def plotImage(self):
        img_gray = cv.applyColorMap(self.img, cv.COLORMAP_JET)
        img_color = cv.cvtColor(img_gray,cv.COLOR_BGR2RGB)
        
        fig, ax = plt.subplots(1,1)
        ax.imshow(img_color)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("FLUX image of frame %i" %(0))
        plt.show()

def main():
    dp = DeadPxl()
    testPersons = np.arange(1,20,1)
# =============================================================================
#     measurements = ["baseline1", "1.15", "1.30", "1.45", "1.60", "1.120", "1.240",
#                     "baseline2", "2.15", "2.30", "2.45", "2.60", "2.120", "2.240"]
# =============================================================================
    measurements = ["baseline1","baseline2"]
    frames = np.arange(0,3,1)
    df_data = pd.DataFrame({"measurement": ["none"],
                            "x": [0],
                            "y": [0],
                            "intensity": [0]})
    
    for person in testPersons:
        pIdx = str(person).zfill(2)
        for mIdx in measurements:
            for fIdx in frames:
                fileName =str( "/Users/malkusch/PowerFolders/LaSCA/rawData/SCTCAPS%s/SCTCAPS%s_%s_frame%i.tiff" %(pIdx, pIdx, mIdx, fIdx))
                if (os.path.isfile(fileName)):
                    dp.img = cv.imread(fileName, cv.IMREAD_ANYDEPTH)
                    dp.detectDeadPxl()
                    df_temp = dp.df_dead
                    nRow = len(df_temp)
                    measurement = str("SCTCAPS%s_%s_frame_%i" %(pIdx,mIdx, fIdx))
                    df_temp["measurement"] = list(repeat(measurement, nRow))
                    df_data = df_data.append(dp.df_dead)
                    

    df_data = df_data.drop(df_data.index[0])
    df_data.to_csv(path_or_buf = "/Users/malkusch/PowerFolders/LaSCA/deadPxl/baseline_dead_pxl.csv",
                   index = False)

    
    
if __name__ == '__main__':
    main()