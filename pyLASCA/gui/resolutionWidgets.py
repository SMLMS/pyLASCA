#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 14:40:56 2020

@author: malkusch
"""

from ..gui import pyLascaWidgets
import tkinter as tk
from tkinter.filedialog import askopenfilename

class ResolutionWidgets(pyLascaWidgets.PyLascaWidgets):
    def __init__(self):
        pyLascaWidgets.PyLascaWidgets.__init__(self)
        self.fileName2 = str()
        self.pathText = self.createPathText()
        self.pathButton = self.createDataPathButton()
        self.pathText2 = self.createPathText2()
        self.pathButton2 = self.createDataPathButton2()
        self.analyzeButton = self.createButton(desc = 'convert')
        self.pxlSizeTextFloat = self.createTextFloat(val = 0.24, maxVal = 10000, desc = "Pxl Size [mm]")
        self.saveButton = self.createButton(desc = 'save')
        
    def createPathText(self):
        text = self.createTextStr(value='', desc = "path to image 1")
        text.observe(self.updateFileName)
        return text
    
    def createDataPathButton(self):
        button = self.createButton(desc = "browse image 1")
        button.on_click(self.browseImage)
        return button
    
    def createPathText2(self):
        text = self.createTextStr(value='', desc = "path to image 2")
        text.observe(self.updateFileName2)
        return text
    
    def createDataPathButton2(self):
        button = self.createButton(desc = "browse image 2")
        button.on_click(self.browseImage2)
        return button 
    
    def browseImage2(self, b):
        root = tk.Tk()
        root.withdraw()
        root.update()
        root.name = askopenfilename(title="import image", filetypes=(("image files", "*.bmp *.tif *.tiff"),
                                       ("All files", "*.*") ))
        self.fileName2 = root.name
        self.pathText2.value = self.fileName2
        root.update()
        root.destroy()
        
    def updateFileName2(self, change):
        self.fileName2 = self.pathText2.value
        
    