#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 14:00:26 2020

@author: malkusch
"""

import numpy as np
import pandas as pd
from scipy.optimize import brute
from matplotlib import pyplot as plt

class LossBaseline(object):
    def __init__(self, df_data: pd.DataFrame, x_min: float, x_max: float):
        self._keep_track = False
        self._df_data = df_data
        self._df_metrics = pd.DataFrame
        self._loss_value = float
        self._x_value = float
        self._x_min = x_min
        self._x_max = x_max
        
    @property
    def df_data(self) -> pd.DataFrame:
        return(self._df_data.copy())
    
    @property
    def df_metrics(self) -> pd.DataFrame:
        return(self._df_metrics)
    
    @property
    def loss_value(self) -> float:
        return(self._loss_value)
    
    @property
    def x_value(self) -> float:
        return(self._x_value)
    
    @property
    def x_min(self) -> float:
        return(self._x_min)
    
    @property
    def x_max(self) -> float:
        return(self._x_max)
    
    def minimize_brute(self):
        ################
        # define range #
        ################
        min_value = self.x_min
        max_value = self.x_max
        
        #######################
        # define fit function #
        #######################
        self._keep_track = True
        self._df_metrics = pd.DataFrame({'x': [np.nan],
                                         'loss': [np.nan]})
        result = brute(func = self.loss_function,
                       ranges = (slice(min_value, max_value, 0.01),)
                       )
        
        ###############
        # log results #
        ###############
        self._keep_track = False
        self._df_metrics = self._df_metrics.sort_values(by='x', ignore_index = True)
        self.loss_function(x = result)
        self._df_data["thr_bl"] = (self._df_data["mean_bl"] + self.x_value * self._df_data["std_bl"])
        self._df_data["residue"] = self._df_data["thr_bl"] - self._df_data["thr"]
        
    
    def loss_function(self, x: float) -> float:
        self._x_value = x[0]
        y = 0.0
        mu = self._df_data["mean_bl"].values
        sigma = self._df_data["std_bl"].values
        thr = self._df_data["thr"].values
        for i in range(mu.shape[0]):
            temp_loss = ((mu[i] + x[0] * sigma[i]) - thr[i])**2
            if (np.isnan(temp_loss)):
                y += 0.0
            else:
                y += temp_loss
        self._loss_value = y
        
        ##############
        # keep track #
        ##############
        if(self._keep_track):
            self._df_metrics = self._df_metrics.append({'x': x[0],
                                                        'loss': self.loss_value},
                                                       ignore_index=True)
        return(self.loss_value)
    
    def plot_minimization_process(self):
        self._df_metrics.plot(x="x", y="loss", kind = "line")
        plt.axvline(x= self.x_value, linestyle='dashed', label = 'threshold')
        plt.grid()
        plt.show()
        plt.close()
        
    def plot_residues(self):
        ax = self.df_data["residue"].plot.hist(bins=50, alpha=0.5)
        ax.set_xlabel('x [a.u.]')
        ax.set_title("residue distribution for x = %.2f" %(self.x_value))
        plt.grid()
        plt.show()
        plt.close()
        
    def thr_scatter_plot(self):
        min_value = self.df_data["thr_bl"].min()
        max_value = self.df_data["thr_bl"].max()
        ax = self.df_data.plot.scatter(x='thr', y='thr_bl')
        ax.plot([min_value, max_value],[min_value, max_value], linestyle = '--', color = "red")
        ax.set_title("thr scatter plot for x = %.2f" %(self.x_value))
        plt.grid()
        plt.show()
        plt.close()
        
        


def main():    
    file_name = "/Users/malkusch/PowerFolders/LaSCA/thr_alpha/SCTCAPS_thr_complete_noUnits_200821.csv"
    df_data = pd.read_csv(file_name)
    
    df_data = df_data[df_data["effect"] == "sec_hyperalgesia"]
    #df_data = df_data[df_data["effect"] == "allodynia"]
    df_data = df_data[df_data["application_type"] == "i.c."]
    #df_data = df_data[df_data["application_type"] == "topic"]
    df_data = df_data[df_data["frame"] == 1]
    df_data = df_data[df_data["pxlSize_bl"] > 1.682e-01]
    df_data = df_data[df_data["pxlSize_bl"] <= 3.141e-01]
    df_data = df_data[df_data["pxlSize"] > 1.682e-01]
    df_data = df_data[df_data["pxlSize"] <= 3.141e-01]
    df_data = df_data[df_data["time"] > -15]
    
    loss = LossBaseline(df_data, x_min = 0.0, x_max = 3.5)
    loss.minimize_brute()
    loss.plot_minimization_process()
    loss.plot_residues()
    loss.thr_scatter_plot()
    print(loss.x_value, loss.loss_value)
    print(loss.df_data[["mean_bl", "std_bl", "thr", "residue"]].head())
    
    
    
    

if __name__ == '__main__':
    main()