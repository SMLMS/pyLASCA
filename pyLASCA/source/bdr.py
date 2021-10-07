#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 12:38:36 2020

@author: malkusch
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from itertools import combinations
from matplotlib import pyplot as plt

class Bayes_decision_rules(object):
    def __init__(self, gmm_model: pd.DataFrame, x_min: float, x_max: float):
        self._gmm_model = gmm_model
        self._fitResult = pd.DataFrame
        self._x_min = x_min
        self._x_max = x_max
        self._error = float
        
        
    @property
    def gmm_model(self) -> pd.DataFrame:
        return(self._gmm_model.copy())
    
    @property
    def fitResult(self) -> pd.DataFrame:
        return(self._fitResult.copy())
    
    @property
    def x_min(self) -> float:
        return(self._x_min)
    
    @property
    def x_max(self) -> float:
        return(self._x_max)
    
    @property
    def error(self) -> float:
        return(self._error)
    

    def fit(self):
        component = self.gmm_model["component"].values
        omega = self.gmm_model["omega"].values
        mu = self.gmm_model["mu"].values
        sigma = self.gmm_model["sigma"].values
        
        comb = np.array(list(combinations(component, 2)))
        boundaries = np.zeros([comb.shape[0], 2])
        component_pair = np.zeros([comb.shape[0], 2])
        for i in range(comb.shape[0]):
            component_pair[i,:] = comb[i,:]
            boundaries[i,:] = self.decision_boundary(omega_1 = omega[comb[i,0]],
                                                     mu_1 = mu[comb[i,0]],
                                                     sigma_1 = sigma[comb[i,0]],
                                                     omega_2 = omega[comb[i,1]],
                                                     mu_2 = mu[comb[i,1]],
                                                     sigma_2 = sigma[comb[i,1]])
        
        self._fitResult = pd.DataFrame({"component_1": component_pair[:,0],
                                        "component_2": component_pair[:,1],
                                        "boundary_1": boundaries[:,0],
                                        "boundary_2": boundaries[:,1]})
        self._fitResult = pd.melt(self._fitResult,
                                  id_vars=["component_1", "component_2"],
                                  value_vars=['boundary_1', 'boundary_2'],
                                  var_name='boundary',
                                  value_name='value')
        
        
        self._fitResult = self._fitResult[self._fitResult["value"] > self.x_min]
        self._fitResult = self._fitResult[self._fitResult["value"] < self.x_max]
        
        bound_seq = self.fitResult["value"].values
        true_boundary = np.repeat(False, bound_seq.shape[0])
        for i in range (bound_seq.shape[0]):
            p = self.probability(bound_seq[i])
            if not(np.sum(np.isclose(p, np.max(p))) < 2):
                true_boundary[i] = True
            
        self._fitResult["true_b"] = true_boundary
        self._fitResult = self._fitResult[self._fitResult["true_b"] == True]
        self._fitResult = self._fitResult[["component_1", "component_2", "value"]]
        self._fitResult = self._fitResult.sort_values("value", ascending = True, ignore_index = True)
        self.numeric_error_estimation()

            
    def rules(self):
        boundaries = self.fitResult["value"].values
        compounds = np.zeros(boundaries.shape[0] + 1)
        temp_rule = str("if %.3e < x and x <= %.3e  " %(0.0,0.0))
        rules = np.repeat(temp_rule, boundaries.shape[0] + 1)
        l_bound = self.x_min
        for i in range(boundaries.shape[0]):
            h_bound = boundaries[i]
            mean_value = l_bound + ((h_bound - l_bound) / 2.0)
            compounds[i] = self.predict_component(mean_value)
            rules[i] = str("if %.3e < x and x <= %.3e" %(l_bound, h_bound))
            l_bound = h_bound
        h_bound = self.x_max
        mean_value = l_bound + ((h_bound - l_bound) / 2.0)
        compounds[boundaries.shape[0]] = self.predict_component(mean_value)
        rules[boundaries.shape[0]] = str("if %.3e < x and x <= %.3e" %(l_bound, h_bound))
        df = pd.DataFrame({"compound": compounds,
                           "rule": rules})
        return(df.copy())
        
        
    def numeric_error_estimation(self):
        x = np.arange(start = self.x_min, stop = self.x_max, step = ((self.x_max-self.x_min) / 500))
        normalizer = np.zeros(x.shape[0])
        numer = np.zeros(x.shape[0])
        y = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            normalizer[i] = np.sum(self.probability(x[i]))
            numer[i] = np.max(self.probability(x[i]))
        y = normalizer - numer
        self._error = np.trapz(y = y, x=x)
        
        
    def predict_component(self, value):
        p = self.probability(value) / np.sum(self.probability(value))
        component = self.gmm_model["component"].values
        return(component[np.argmax(p)])
        
        
    def decision_boundary(self, omega_1, mu_1, sigma_1, omega_2, mu_2, sigma_2):
        x = []
        if (sigma_1 == sigma_2 and omega_1 == omega_2):
            x.append(self.equal_sd_omega_boundary(mu_1, mu_2))
            x.append(np.nan)
        elif ((sigma_1 == sigma_2 and omega_1 != omega_2)):
            x.append(self.equal_sd_boundary(omega_1, mu_1, omega_2, mu_2, sigma_1))
            x.append(np.nan)
        else:
            x.append(self.positive_boundary(omega_1, mu_1, sigma_1, omega_2, mu_2, sigma_2))
            x.append(self.negative_boundary(omega_1, mu_1, sigma_1, omega_2, mu_2, sigma_2))
        return(np.asarray(x))
            
    
    def positive_boundary(self, omega_1, mu_1, sigma_1, omega_2, mu_2, sigma_2):
        c_1 = np.log(omega_1) + np.log(1.0 / np.sqrt(2 * np.pi * (sigma_1**2)))
        c_2 = np.log(omega_2) + np.log(1.0 / np.sqrt(2 * np.pi * (sigma_2**2)))
        x_1 = np.abs(-2.0 * (c_1 - c_2) * (sigma_1**2 - sigma_2**2)) + ((mu_1 - mu_2)**2)
        x_2 = mu_2 * sigma_1**2
        x_3 = mu_1 * sigma_2**2
        x_4 = sigma_1**2 - sigma_2**2
        if(sigma_1 == sigma_2):
            x = np.nan
        else:
            x = (x_2 - x_3 + np.sqrt(sigma_1**2 * sigma_2**2 * x_1)) / (x_4)
        return(x)
        
    def negative_boundary(self, omega_1, mu_1, sigma_1, omega_2, mu_2, sigma_2):
        c_1 = np.log(omega_1) + np.log(1.0 / np.sqrt(2 * np.pi * (sigma_1**2)))
        c_2 = np.log(omega_2) + np.log(1.0 / np.sqrt(2 * np.pi * (sigma_2**2)))
        x_1 = np.abs(-2.0 * (c_1 - c_2) * (sigma_1**2 - sigma_2**2)) + ((mu_1 - mu_2)**2)
        x_2 = mu_2 * sigma_1**2
        x_3 = mu_1 * sigma_2**2
        x_4 = sigma_1**2 - sigma_2**2
        if(sigma_1 == sigma_2):
            x = np.nan
        else:
            x = (x_2 - x_3 - np.sqrt(sigma_1**2 * sigma_2**2 * x_1)) / (x_4)
        return(x)
        
    def equal_sd_boundary(self, omega_1, mu_1, omega_2, mu_2, sigma):
        c_1 = np.log(omega_1) + np.log(1.0 / np.sqrt(2 * np.pi * (sigma**2)))
        c_2 = np.log(omega_2) + np.log(1.0 / np.sqrt(2 * np.pi * (sigma**2)))
        if(mu_1 == mu_2):
            x = np.nan
        else:
            x = ((2.0 * sigma**2 * (c_2 - c_1)) + (mu_1**2 - mu_2**2)) / (2.0* (mu_1 - mu_2))
        return(x)
    
    def equal_sd_omega_boundary(self, mu_1, mu_2):
        x = (mu_1 + mu_2) / 2.0
        return(x)
        
    def probability(self, value):
        component = self.gmm_model["component"].values
        omega = self.gmm_model["omega"].values
        mu = self.gmm_model["mu"].values
        sigma = self.gmm_model["sigma"].values
        p = np.zeros(component.shape[0])
        for i in range(component.shape[0]):
            p[i] = omega[i]*stats.norm.pdf(value,mu[i],sigma[i])
        return(p)

    def plot_model(self):
        omega = self.gmm_model["omega"].values
        mu = self.gmm_model["mu"].values
        sigma = self.gmm_model["sigma"].values
        
        x = np.arange(start = self.x_min, stop = self.x_max, step = ((self.x_max-self.x_min) / 500))            
        y = np.zeros(np.shape(x)[0])
        
        fig = plt.figure()
        for i in range(self.gmm_model.shape[0]):
            y_temp = omega[i]*stats.norm.pdf(x,mu[i],sigma[i])
            y +=  y_temp
            plt.plot(x,y_temp, c='magenta', alpha = 0.8, linestyle=":", linewidth=5)
        
        
        for i in range(self._fitResult.shape[0]):
            plt.vlines(self._fitResult["value"][i], ymin = 0, ymax = np.max(y), linestyle = "--", colors="black", linewidth=5)
        plt.plot(x,y, c='red', alpha = 0.5, linestyle="-",  linewidth=5)
        
        plt.xlabel("value")
        plt.ylabel("pdf")

        # plt.grid()
        # plt.show()
        # plt.close()
        return fig
            
            
        

def main():
    component = np.arange(0,4,1)
    omega = np.asarray([0.25, 0.25, 0.25, 0.25])
    mu = np.asarray([-0.3, 0.3, 0.5, 0.7])
    sigma = np.asarray([0.1, 0.5, 0.1, 0.2])
    gmm_model = pd.DataFrame({"component": component,
                              "omega": omega,
                              "mu": mu,
                              "sigma": sigma})
    print(gmm_model)
    bdr = Bayes_decision_rules(gmm_model, x_min = -1.5, x_max = 2.0)
    
    bdr.fit()
    print(bdr.fitResult)
    print(bdr.error)
    print(bdr.rules())
    bdr.plot_model()
    
    

if __name__ == '__main__':
    main()