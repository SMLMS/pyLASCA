#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 08:08:35 2020

@author: malkusch
"""

import pandas as pd

def main():
    fileName = "/Users/malkusch/PowerFolders/LaSCA/deadPxl/baseline_dead_pxl.csv"
    df = pd.read_csv(fileName)
    df_hist = df.pivot_table(index=['x','y'], aggfunc='size')
    df2 = df.groupby("x")
    print(df2)
    
    
if __name__ == '__main__':
    main()