#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 17:44:22 2019

@author: malkusch
"""

from ..gui import pyLascaWidgets

class LandmarksWidgets(pyLascaWidgets.PyLascaWidgets):
    def __init__(self):
        pyLascaWidgets.PyLascaWidgets.__init__(self)
        self.pathText = self.createPathText()
        self.pathButton = self.createImagePathButton()
        self.importButton = self.createButton(desc = 'import')
        self.analysisButton = self.createButton(desc = 'analysis')
        self.pxlSizeText = self.createTextFloat(val=120, minVal=0, maxVal=200, stepSize=0.1, desc="pxl size [mm]")
        self.sigmaText = self.createTextFloat(val=0.35, minVal=0.0, maxVal=1.0, stepSize=0.001, desc="sigma")
        self.sigmaSlider = self.createSliderFloat(val=0.35, minVal=0.0, maxVal=1.0, stepSize=0.001, desc="sigma")
        self.sigmaLink = self.createLink(self.sigmaText, self.sigmaSlider)
        self.landmarkSelector = self.createSelector(opt = [-1],val = -1, rows=1, desc = 'Select')
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
        
        
# =============================================================================
#         self.nText = self.createTextInt(val=0, minVal=0, maxVal=100, stepSize=1, desc="n")
#         self.p0Text = self.createTextInt(val=1, minVal=0, maxVal=100, stepSize=1, desc="p0")
#         self.mText = self.createTextInt(val=1, minVal=0, maxVal=100, stepSize=1, desc="m")
#         self.pText = self.createTextFloat(val=0.3, minVal=0.0, maxVal=1.0, stepSize=0.001, desc="p")
#         self.initDText = self.createTextFloat(val=0.3, minVal=0.0, maxVal=1.0, stepSize=0.001, desc="initD")
#         self.initDSlider = self.createSliderFloat(val=0.3, minVal=0.0, maxVal=1.0, stepSize=0.001, desc="initD")
#         self.initDLink = self.createLink(self.initDText, self.initDSlider)
#         self.analysisButton = self.createButton(desc = 'run Analysis')
#         self.saveButton = self.createButton(desc = 'save results')
# =============================================================================
