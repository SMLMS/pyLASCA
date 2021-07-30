#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 11:08:40 2020

@author: malkusch
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.mixture import BayesianGaussianMixture

class BGMM1d(object):
    
    def __init__(self, data: np.ndarray, max_components=3, iterations = 10, seed = 42):
        self._data = np.copy(data)
        self._instances = np.shape(self._data)[0]
        self._seed = seed
        self._iterations = iterations
        self._max_components = max_components
        self._fitResult = pd.DataFrame
        self.fit()
        
    def __str__(self):
        message = str("\nInstance of BGMM1d:")
        message = str("%s\nseed: %i" %(message, self.seed))
        message = str("%s\ninstances: %i" %(message, self.instances))
        message = str("%s\niterations: %i" %(message, self.iterations))
        message = str("%s\nmax_components: %i" %(message, self.max_components))
        message = str("%s\n" %(message))
        return(message)
        
    def __del__(self):
        message = str("Instance of BGMM1d removed form heap")
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
    def max_components(self) -> int:
        return(self._max_components)
    
    @property
    def fitResult(self) -> pd.DataFrame:
        return(self._fitResult.copy())
        
    def fit(self):
        bgm = BayesianGaussianMixture(n_components=self.max_components,
                                      n_init = self.iterations,
                                      random_state=self.seed,
                                      covariance_type='full',
                                      weight_concentration_prior_type="dirichlet_process",
                                      init_params="random",
                                      weight_concentration_prior=1e-3,
                                      reg_covar=0,
                                      mean_precision_prior=.8,
                                      max_iter=1000)
        bgm.fit(self.data)
        self._fitResult = pd.DataFrame({"component": np.arange(self.max_components),
                                        "omega": bgm.weights_,
                                        "mu": bgm.means_.ravel(),
                                        "sigma": np.sqrt(bgm.covariances_.ravel())})
        self._fitResult = self._fitResult.sort_values("omega", ascending = False, ignore_index = True)
        self._fitResult["cum_omega"] = np.cumsum(bgm.weights_)

    def optimal_components(self, thr = 3) -> int:
        omega = np.round(self.fitResult["omega"].values, thr)
        return(np.sum(omega > 0.0))
        
        
    def plot_fit_procedure(self):
        self._fitResult.plot(x="component", y=["cum_omega"], kind = "line")
        plt.xlabel("number of model components")
        plt.ylabel("cummulative fraction")
        plt.grid()
        plt.show()
        plt.close()

        
    



def main():
    
    file_name = "/Users/malkusch/PowerFolders/LaSCA/thr_alpha/SCTCAPS_thr_complete_noUnits_200821.csv"
    
    df_data = pd.read_csv(file_name)
    
    x = df_data[df_data["frame"] == 0]['pxlSize'].values.reshape(-1, 1)
    
    print(df_data.head())
    
    bgm = BGMM1d(data = x,
                max_components = 8,
                iterations = 10)
    
    print(bgm.fitResult)
    
    bgm.plot_fit_procedure()
    print(bgm.optimal_components())

    
    

if __name__ == '__main__':
    main()