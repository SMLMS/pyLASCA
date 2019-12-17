#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 14:04:11 2019

@author: malkusch
"""
import os
import numpy as np

class TextDocument:
    def __init__(self):
        self._fileName = str()
        self._x = int()
        self._y = int()
        self._frames = int()
        self._t = np.ndarray([0,0])
        self._data = np.ndarray([0,0,0], dtype = np.uint8)
        
    @property
    def fileName(self):
        return self._fileName
    
    @fileName.setter
    def fileName(self, name):
        if(type(name) != str):
            errorMessage = str('TextDocument instance variable fileName should be of type str, was of type %s.' % (type(name)))
            raise Exception(errorMessage)
        else:
            self._fileName = name
            
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
    
    def fileExists(self):
        return os.path.isfile(self.fileName)
    
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
    
    def readMetaInformation(self):
        if(self.fileExists()):
            with open(self.fileName, 'r') as fp:
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
            errorMessage = str('TextDocument import error, could not read from file %s.' % (self.fileName))
            raise Exception(errorMessage)
            
    def readFrame(self, frameNumber: int):
        n = 32 + frameNumber * (2 + self.y)
        p = 0
        if(self.fileExists()):
            with open(self.fileName, 'r') as fp:
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
            errorMessage = str('TextDocument import error, could not read from file %s.' % (self.fileName))
            raise Exception(errorMessage)
    
    def readFrames(self):
        for i in range(0, self.frames,1):
            self.readFrame(i)
    
    def readTextDocument(self):
        self.readMetaInformation()
        self.readFrames()
        
    
    def __str__(self):
        string = str('TextDocument\nfileName: %s\nx: %s\ny: %s\nframes: %s\n' %(self.fileName,
                                                                                self.x,
                                                                                self.y,
                                                                                self.frames))
        return string
    
    def __del__(self):
        message  = str("removed instance of TextDocument from heap.")
        print(message)


def main():
    td = TextDocument()
    td.fileName = "/home/malkusch/PowerFolders/pharmacology/Daten/ristow/test.txt"
    td.readTextDocument()
    print(td)
    print(td.data[:,:,1])
     
if __name__ == '__main__':
    main()