#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 09:46:46 2020

@author: malkusch
"""

from ..gui import pyLascaWidgets

class FileNamesWidgets(pyLascaWidgets.PyLascaWidgets):
    def __init__(self):
        pyLascaWidgets.PyLascaWidgets.__init__(self)
        self.pathText = self.createPathText()
        self.pathButton = self.createImagePathButton()        
    
    def createPathText(self):
        text = self.createTextStr(value='', desc='path to file')
        text.observe(self.updateFileName)
        return text
    
    def createImagePathButton(self):
        button = self.createButton(desc = 'browse')
        button.on_click(self.browseImage)
        return button            