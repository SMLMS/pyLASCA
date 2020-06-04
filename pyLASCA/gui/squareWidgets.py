#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 12:01:47 2020

@author: malkusch
"""

from ..gui import pyLascaWidgets

class SquareWidgets(pyLascaWidgets.PyLascaWidgets):
    def __init__(self):
        pyLascaWidgets.PyLascaWidgets.__init__(self)
        self.pathText = self.createPathText()
        self.pathButton = self.createImagePathButton()
        self.importButton = self.createButton(desc = 'import')
        self.sigmaAnalysisButton = self.createButton(desc = 'analysis')
        self.vertrexAnalysisButton = self.createButton(desc = 'analysis')
        self.pxlSizeText = self.createTextFloat(val=120, minVal=0, maxVal=200, stepSize=0.1, desc="pxl size [mm]")
        self.sigmaText = self.createTextFloat(val=5.0, minVal=0.1, maxVal=20.0, stepSize=0.1, desc="sigma")
        self.sigmaSlider = self.createSliderFloat(val=5.0, minVal=0.1, maxVal=20.0, stepSize=0.1, desc="sigma")
        self.sigmaLink = self.createLink(self.sigmaText, self.sigmaSlider)
        self.linekSelector = self.createSelector(opt = [-1],val = -1, rows=1, desc = 'Select')
        self.deleteButton = self.createButton(desc = 'delete')
        self.saveButton = self.createButton(desc = 'save')
    
    def createPathText(self):
        text = self.createTextStr(value='', desc='path to file')
        text.observe(self.updateFileName)
        return text
    
    def createImagePathButton(self):
        button = self.createButton(desc = 'browse')
        button.on_click(self.browseImage)
        return button 