#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 13:06:06 2020

@author: malkusch
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
import scipy.stats as stats

class GMM1d(object):
    
    def __init__(self, data: np.ndarray, min_components=1, max_components=3, iterations = 10, seed = 42):
        self._data = np.copy(data)
        self._instances = np.shape(self._data)[0]
        self._seed = seed
        self._iterations = iterations
        self._min_components = min_components
        self._max_components = max_components
        self._fitResult = pd.DataFrame
        self.fit()
        
    def __str__(self):
        message = str("\nInstance of GMM1d:")
        message = str("%s\nseed: %i" %(message, self.seed))
        message = str("%s\ninstances: %i" %(message, self.instances))
        message = str("%s\niterations: %i" %(message, self.iterations))
        message = str("%s\nmin_components: %i" %(message, self.min_components))
        message = str("%s\nmax_components: %i" %(message, self.max_components))
        message = str("%s\n" %(message))
        return(message)
        
    def __del__(self):
        message = str("Instance of GMM1d removed form heap")
        print(message)
        
    @property
    def data(self) -> np.ndarray:
        return(np.copy(self._data))
    
    @property
    def instances(self) -> int:
        return(self._instances)
    
    @property
    def seed(self) -> int:
        return(self._seed)
    
    @property
    def iterations(self) -> int:
        return(self._iterations)
    
    @property
    def min_components(self) -> int:
        return(self._min_components)
    
    @property
    def max_components(self) -> int:
        return(self._max_components)
    
    @property
    def fitResult(self) -> pd.DataFrame:
        return(self._fitResult.copy())
        
    def fit(self):
        comp_seq = np.arange(start = self._min_components, stop = self._max_components+1, step = 1, dtype = int)
        bic_seq = np.zeros(len(comp_seq))
        aic_seq = np.zeros(len(comp_seq))
        
        for i in range(len(comp_seq)):
            gm = GaussianMixture(n_components=comp_seq[i],
                                 n_init=self._iterations,
                                 random_state=self._seed,
                                 covariance_type='full')
            gm.fit(self.data)
            bic_seq[i] = gm.bic(self.data)
            aic_seq[i] = gm.aic(self.data)
        
        self._fitResult = pd.DataFrame({"components": comp_seq,
                                        "BIC": bic_seq,
                                        "AIC": aic_seq})
        
        
    def optimal_components(self, metric = "BIC") -> int:
        component_seq = self._fitResult["components"].values
        if (metric == "BIC"):
            metric_seq = self._fitResult["BIC"].values
        elif (metric == "AIC"):
             metric_seq = self._fitResult["AIC"].values
        else:
            print("Metric %s is not known, retry using 'BIC' or 'AIC'." %(metric))
            return(np.nan)
        return(component_seq[np.argmin(metric_seq)])
        
        
    def model(self, components = 1):
        gm =  GaussianMixture(components,
                              n_init=self._iterations,
                              random_state=self._seed,
                              covariance_type='full')
        gm.fit(self.data)
        return(gm)
    
    def optimal_model(self, metric = "BIC"):
        n_opt_comp = self.optimal_components(metric = metric)
        return(self.model(n_opt_comp))
        
    def plot_fit_procedure(self):
        self._fitResult.plot(x="components", y=["BIC", "AIC"], kind = "line")
        plt.xlabel("number of model components")
        plt.ylabel("information criterion")
        plt.grid()
        plt.show()
        plt.close()
        
    def df_model(self, components = 1):
        gm = self.model(components)
        component = np.arange(0, components, 1)
        weights = gm.weights_
        means = gm.means_.ravel()
        sigma = np.sqrt(gm.covariances_.ravel())
        df = pd.DataFrame({"component": component,
                           "weight": weights,
                           "mu": means,
                           "sigma": sigma})
        return(df)
        
        
    def plot_model(self, components=1):
        gm = self.model(components)
        weights = gm.weights_
        
        means = gm.means_
        covars = gm.covariances_
        plt.hist(self.data, bins=100, histtype='bar', density=True, color="gray", ec="black")
        
        x = np.arange(start = np.min(self.data), stop = np.max(self.data), step = ((np.max(self.data) - np.min(self.data)) / 200))
        y = np.zeros(np.shape(x)[0])
        for i in range(components):
            y_temp = weights[i]*stats.norm.pdf(x,means[i],np.sqrt(covars[i])).ravel()
            y +=  y_temp
            plt.plot(x,y_temp, c='red', linestyle=":")
        plt.plot(x,y, c='blue', linestyle="-", alpha = 0.5)
        
        plt.xlabel("value")
        plt.ylabel("pdf")

        plt.grid()
        plt.show()
        plt.close()
        
    



def main():
    n_iter = 10
    
    file_name = "/Users/malkusch/PowerFolders/LaSCA/thr_alpha/SCTCAPS_thr_complete_noUnits_200821.csv"
    
    df_data = pd.read_csv(file_name)
    
    x = df_data[df_data["frame"] == 0]['pxlSize'].values.reshape(-1, 1)
    
    print(df_data.head())
    
    gm = GMM1d(data = x,
               max_components = 8,
               iterations = 10)
    
    print(gm.optimal_components())
    
    gm.plot_fit_procedure()
    gm.plot_model(3)
    
    print(gm)
    print(gm.fitResult)
    print(gm.df_model(3))
    
    

if __name__ == '__main__':
    main()