#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 17:44:22 2019

@author: malkusch
"""

from ipywidgets import widgets
from IPython.display import clear_output
import tkinter as tk
from tkinter.filedialog import askopenfilename

class PyLascaWidgets:
    def __init__(self):
        print("PyLascaWidgets initialized")
        self.style = {'description_width': 'initial'}
        self.fileName = ""
        
        
    def createTextStr(self, value='', desc='path to file', placeHolder = 'enter a string'):
        text = widgets.Text(
                value=str(value),
                placeholder=str(placeHolder),
                description=str(desc),
                disabled=False,
                style = self.style
                )
        return text
        
    def createButton(self, desc = 'browse'):
        button=widgets.Button(
                description=str(desc),
                disabled=False,
                button_style='', # 'success', 'info', 'warning', 'danger' or ''
                tooltip='Click me',
                icon='check',
                style = self.style
                )
        return button
        
    def createTextInt(self, val=0.0, minVal=0, maxVal=100, stepSize=1, desc="d"):
        text = widgets.BoundedIntText(
                value=int(val),
                min=int(minVal),
                max=int(maxVal),
                step=int(stepSize),
                description=str(desc),
                disabled=False,
                style = self.style
                )
        return text
        
    def createTextFloat(self, val=0.0, minVal=0.0, maxVal=1.0, stepSize=0.001, desc="d"):
        text = widgets.BoundedFloatText(
                value=float(val),
                min=float(minVal),
                max=float(maxVal),
                step=float(stepSize),
                description=str(desc),
                disabled=False,
                style = self.style
                )
        return text
    
    def createSliderFloat(self, val=0.0, minVal=0.0, maxVal=1.0, stepSize=0.001, desc="d"):
        slider = widgets.FloatSlider(
                value=float(val),
                min=float(minVal),
                max=float(maxVal),
                step=float(stepSize),
                description=str(desc),
                disabled=False,
                continuous_update=False,
                orientation='horizontal',
                readout=True,
                readout_format='.3f',
                style = self.style
                )
        return slider
    
    def createLink(self, a, b):
        link = widgets.jslink((a, 'value'), (b, 'value'))
        return link
    
    def proofValidity(self, val=False, desc="valid"):
        validity= widgets.Valid(
                value=val,
                description=str(desc),
                style = self.style
                )
        return validity
    
    def createSelector(self, opt = ['least squares', 'maximum likelihood'],val = 'least squares', rows=1, desc = 'optimizer'):
        selector = widgets.Select(
                options=opt,
                value=val,
                rows= rows,
                description=desc,
                disabled=False
                )
        return selector
    
    def browseData(self, b):
        root = tk.Tk()
        root.withdraw()
        root.update()
        root.name = askopenfilename(title="import data", filetypes=(("text files", "*.txt;*csv"),
                                       ("All files", "*.*") ))
        self.fileName = root.name
        self.pathText.value = self.fileName
        root.update()
        root.destroy()
        
    def browseImage(self, b):
        root = tk.Tk()
        root.withdraw()
        root.update()
        root.name = askopenfilename(title="import image", filetypes=[("image files", "*.bmp *.tif *.tiff *.jpg *jpeg"),
                                       ("All files", "*.*") ])
        self.fileName = root.name
        self.pathText.value = self.fileName
        root.update()
        root.destroy()
        
    def updateFileName(self, change):
        self.fileName = self.pathText.value
        
    def clearOutput(self):
        clear_output()