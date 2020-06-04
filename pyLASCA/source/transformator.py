#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 16:09:48 2020

@author: malkusch
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class Transformator:
    def __init__(self):
        self._rawData = pd.DataFrame()
        
    @property
    def rawData(self):
        return(self._rawData.copy())
    
    @rawData.setter
    def rawData(self, df: pd.DataFrame):
        self._rawData = df.copy()
        
    def loadExcel(self, fileName: str):
        # =============================================================================
        # load data
        # =============================================================================
        self._rawData = pd.read_excel(io = fileName, sheet_name = "data" )

    def transformToCartesian(self):
        alpha = [0, 45, 90, 135, 180, 225, 270, 315]
        for i in np.arange(0, 8, 1):
            d_title = str("d%i_[cm]" % (i+1))
            x_title = str("x%i_[cm]" % (i+1))
            y_title = str("y%i_[cm]" % (i+1))
            phi = alpha[i]
            self._rawData[x_title] = self._rawData[d_title] * np.cos(phi * np.pi/180.0)
            self._rawData[y_title] = self._rawData[d_title] * np.sin(phi * np.pi/180.0)
     
    def plotTimePoint(self,appl: str, t: int):
        appl_typd_idx = self.rawData["appl_type"] == appl
        effect_allodyn_idx = self.rawData["effect"] == "allodynia"
        effect_hyperal_idx = self.rawData["effect"] == "sec_hyperalgesia"
        t_idx = self.rawData["t_[min]"] == t
        df1 = self.rawData[appl_typd_idx & effect_allodyn_idx & t_idx]
        df2 = self.rawData[appl_typd_idx & effect_hyperal_idx & t_idx]
        x_title = []
        y_title = []
        for i in np.arange(0, 8, 1):
            x_title = np.append(x_title, str("x%i_[cm]" % (i+1)))
            y_title = np.append(y_title, str("y%i_[cm]" % (i+1)))
        x_allodyn = df1[x_title].values[0]
        y_allodyn = df1[y_title].values[0]
        x_hyperal = df2[x_title].values[0]
        y_hyperal = df2[y_title].values[0]

        fig, ax = plt.subplots()  # Create a figure and an axes.
        ax.fill(x_hyperal, y_hyperal, facecolor='none', edgecolor='cyan', linewidth=3, label="sec_hyperalgesia") 
        ax.fill(x_allodyn, y_allodyn, facecolor='none', edgecolor='purple', linewidth=3, label="allodynia")  # Plot some data on the axes.
        
        ax.set_xlim([-12, 12])
        ax.set_ylim([-6, 6])
        ax.set_aspect('equal', 'box')
        ax.set_title("%s at t = %i min" %(appl, t))  # Add a title to the axes.
        ax.legend()  # Add a legend.
        
    
def main():
    fileName = "../../sctcaps01.xlsx"
    tf = Transformator()
    tf.loadExcel(fileName = fileName)
    tf.transformToCartesian()
    for t in [15, 30,45,60, 120, 240]:
        tf.plotTimePoint(appl= "topic", t=t)
     
if __name__ == '__main__':
    main()