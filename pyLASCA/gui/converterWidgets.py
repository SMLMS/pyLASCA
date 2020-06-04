#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 14:20:30 2020

@author: malkusch
"""

from ..gui import pyLascaWidgets

class ConverterWidgets(pyLascaWidgets.PyLascaWidgets):
    def __init__(self):
        pyLascaWidgets.PyLascaWidgets.__init__(self)
        self.pathText = self.createPathText()
        self.pathButton = self.createDataPathButton()
        self.convertButton = self.createButton(desc = 'convert')
        self.frameSelector = self.createTextInt(val=0, minVal=0, maxVal=0, stepSize=1, desc="channel")
        self.saveButton = self.createButton(desc = 'save')
        
    def createPathText(self):
        text = self.createTextStr(value='', desc='path to file')
        text.observe(self.updateFileName)
        return text
    
    def createDataPathButton(self):
        button = self.createButton(desc = 'browse')
        button.on_click(self.browseData)
        return button 
        