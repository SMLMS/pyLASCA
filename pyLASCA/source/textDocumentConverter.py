#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 14:04:11 2019

@author: malkusch
"""
import os
import numpy as np
import pandas as pd
import cv2 as cv
from matplotlib import pyplot as plt

class TextDocumentConverter:
    def __init__(self):
        self._x = int()
        self._y = int()
        self._frames = int()
        self._t = np.ndarray([0,0])
        self._data = np.ndarray([0,0,0], dtype = np.uint8)

    @property
    def x(self):
        return self._x
    
    @property
    def y(self):
        return self._y
    
    @property
    def frames(self):
        return self._frames
    
    @property
    def t(self):
        return self._t
    
    @property
    def data(self):
        return self._data
    
    def fileExists(self, fileName = str):
        return os.path.isfile(fileName)
    
    def readResolution(self, line: str):
        dim = np.zeros([1,2])
        n = 0
        for element in line.split():
            if element.isdigit():
                dim[0,n] = element
                n += 1
        self._x = int(dim[0,0])
        self._y = int(dim[0,1])
        
    def readTotalFrames(self, line: str):
        for element in line.split():
            if element.isdigit():
                self._frames = int(element)
                break
    
    def readMetaInformation(self, fileName: str):
        if(self.fileExists(fileName)):
            with open(fileName, 'r') as fp:
                for i, line in enumerate(fp):
                    if (i==25):
                        self.readResolution(line)
                    elif (i==28):
                        self.readTotalFrames(line)
                    elif (i>29):
                        break
            self._t = np.zeros([0,self.frames], dtype = float)
            self._data = np.zeros([self.x, self.y, self.frames])
        else:
            errorMessage = str('TextDocument import error, could not read from file %s.' % (fileName))
            raise Exception(errorMessage)
            
    def readFrame(self, frameNumber: int, fileName: str):
        n = 32 + frameNumber * (2 + self.y)
        p = 0
        if(self.fileExists(fileName)):
            with open(fileName, 'r') as fp:
                for i, line in enumerate(fp):
                    if (i == n):
                        pass
                    elif (i > n + self.y):
                        break
                    elif (i > n):
                        self._data[:, p, frameNumber] = [int(s) for s in line.split() if s.isdigit()][1:]
                        p += 1
            self._data = np.clip(self._data, 0, 255)
                    
        else:
            errorMessage = str('TextDocument import error, could not read from file %s.' % (fileName))
            raise Exception(errorMessage)
    
    def readFrames(self, fileName: str):
        for i in range(0, self.frames,1):
            self.readFrame(frameNumber = i, fileName = fileName)
    
    def readTextDocument(self, fileName: str):
        self.readMetaInformation(fileName)
        self.readFrames(fileName)
        
    def writeFrameToImage(self, frame: int, fileName: str):
        if (self.frames > frame):
            cv.imwrite(fileName, self.data[:,:,frame])
            print("Wrote raw channel %i to %s" %(frame, fileName))
        
    def frame(self, frame: int):
        if (self.frames > frame):
# =============================================================================
#             frame  = self.data[:,:,frame]
#             image_gray = cv.normalize(src=frame, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
# =============================================================================
            return(self.data[:,:,frame])
        else:
            return(None)
        
    def drawFrame(self, frame: int):
        if (self.frames > frame):
            frame  = 255 - self.data[:,:,frame]
            image_gray = cv.normalize(src=frame, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
            image_color = cv.applyColorMap(image_gray, cv.COLORMAP_JET)
            fig, ax = plt.subplots()
            ax.imshow(image_color)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title("raw image")
            return(image_gray)
        else:
            return(None)
            
    def histFrame(self, frame: int):
        if(self.frames > frame):
            hist,bins = np.histogram(self.data[:,:,frame],256,[0,256])
            df_hist = pd.DataFrame({"intensity": bins[1:-1],
                                    "frequency": hist[1:]})
# =============================================================================
#             fig, ax = plt.subplots()
#             ax.bar(bins[2:], hist[1:])
#             ax.set_xlabel("intensity")
#             ax.set_ylabel("frequency")
#             ax.set_title("Intensity histogram")
# =============================================================================
            return(df_hist)
        else:
            return(None)
    
    def __str__(self):
        string = str('TextDocument dimensions:\nx: %s\ny: %s\nframes: %s\n' %(self.x,
                                                                                self.y,
                                                                                self.frames))
        return string
    
    def __del__(self):
        message  = str("removed instance of TextDocument from heap.")
        print(message)


def main():
    td = TextDocumentConverter()
    td.readTextDocument(fileName = "/Users/malkusch/PowerFolders/pharmacology/Daten/ristow/SCTCAPS 01 #1 15min text (Flux).txt")
    td.drawFrame(frame = 0)
    td.histFrame(frame = 0)
     
if __name__ == '__main__':
    main()