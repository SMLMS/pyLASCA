#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 14:44:27 2021

@author: malkusch
"""

import numpy as np
import pandas as pd
import cv2 as cv
import itertools
import os.path

def shoelace(x_y):
    x_y = np.array(x_y)
    x_y = x_y.reshape(-1,2)

    x = x_y[:,0]
    y = x_y[:,1]

    S1 = np.sum(x*np.roll(y,-1))
    S2 = np.sum(y*np.roll(x,-1))

    area = .5*np.absolute(S1 - S2)

    return area

base_folder = str("/Users/malkusch/PowerFolders/LaSCA")


# =============================================================================
# Lumpensammler
# =============================================================================
n=0
nmax = 19*2*6
subject_id = np.zeros(nmax) * np.nan
day_id = np.zeros(nmax) * np.nan
time_id = np.zeros(nmax) * np.nan
area_img_allod = np.zeros(nmax) * np.nan
area_img_hyper = np.zeros(nmax) * np.nan
area_mes_allod = np.zeros(nmax) * np.nan
area_mes_hyper = np.zeros(nmax) * np.nan

# =============================================================================
# permute all combinations
# =============================================================================
subjects = np.arange(19)+1
days = np.arange(2)+1
time = [15,30,45,60,120,240]

measurements = np.asarray(list(itertools.product(subjects, days, time)))
for i,d,t in measurements:
    print([i,d,t])

    subject_id[n] = i
    day_id[n] = d
    time_id[n] = t
    
    # =============================================================================
    # define filenames
    # =============================================================================
    image_folder = str("%s/Probanden Arm Bilder mit Markierung/SCTCAPS %s/Tag %i"
                       %(base_folder, str(i).zfill(2), d))
    image_filename = str("%s/SCTCAPS %s #%i %imin (Colour).jpg"
                     %(image_folder, str(i).zfill(2), d, t))
    
    measurement_folder = str("%s/mechanic/SCTCAPS %s/Tag %i"
                             %(base_folder, str(i).zfill(2), d))
    measurement_filename = str("%s/SCTCAPS %s #%i %imin (Colour)_transformed_200612.csv"
                               % (measurement_folder, str(i).zfill(2), d,t))
    if os.path.isfile(image_filename) & os.path.isfile(measurement_filename):
        # =============================================================================
        # load measurment
        # =============================================================================
        measurement_df = pd.read_csv(measurement_filename)
        coordinates_allod_df = measurement_df[measurement_df["effect"] == "allodynia"][["alpha", "x_[mm]", "y_[mm]", "x_img_[pxl]" , "y_img_[pxl]"]].copy()
        coordinates_allod_df.sort_values(by = "alpha", ascending=True, inplace=True)
        coordinates_hyper_df = measurement_df[measurement_df["effect"] == "sec_hyperalgesia"][["alpha", "x_[mm]", "y_[mm]", "x_img_[pxl]" , "y_img_[pxl]"]].copy()
        coordinates_hyper_df.sort_values(by = "alpha", ascending=True, inplace=True)
        
        # =============================================================================
        # extract coordinates
        # =============================================================================
        coords_img_allod = coordinates_allod_df[["x_img_[pxl]", "y_img_[pxl]"]].to_numpy()
        coords_img_allod = coords_img_allod.reshape(coordinates_allod_df.shape[0],1,2).astype(int)
        coords_img_hyper = coordinates_hyper_df[["x_img_[pxl]", "y_img_[pxl]"]].to_numpy()
        coords_img_hyper = coords_img_hyper.reshape(coordinates_hyper_df.shape[0],1,2).astype(int)
        coords_mes_allod = coordinates_allod_df[["x_[mm]", "y_[mm]"]].to_numpy()
        coords_mes_hyper = coordinates_hyper_df[["x_[mm]", "y_[mm]"]].to_numpy()
        # =============================================================================
        # write image
        # =============================================================================
        # im1 = cv.imread(image_filename)
        # im2 = cv.drawContours(im1, [coords_img_allod], 0, (255,0,255), 3)
        # im3 = cv.drawContours(im2, [coords_img_hyper], 0, (255,255,0), 3)
        # image_outfilename = str("%s/temp/SCTCAPS %s #%i %imin.tif"
        #                         % (base_folder, str(i).zfill(2), d,t))
        # cv.imwrite(image_outfilename, im3)
        
        # =============================================================================
        # measure results
        # =============================================================================
        area_img_allod[n] = cv.contourArea(coords_img_allod) #+ cv.arcLength(coords_img_allod,True)
        area_img_hyper[n] = cv.contourArea(coords_img_hyper) #+ cv.arcLength(coords_img_hyper,True)
        area_mes_allod[n] = shoelace(coords_mes_allod)
        area_mes_hyper[n] = shoelace(coords_mes_hyper)
    
    # =============================================================================
    # iterate
    # =============================================================================
    n += 1

result_img_df = pd.DataFrame({"object_id": subject_id.astype(int),
                              "day": day_id.astype(int),
                              "time": time_id.astype(int),
                              "allodynia": area_img_allod,
                              "sec_hyperalgesia": area_img_hyper})

# pd.melt(result_img_df,
#         id_vars=['object_id', 'day', 'time'],
#         value_vars=['allodynia', "sec_hyperalgesia"],
#         var_name='effect', value_name='img_area_[pxl]')

result_img_df["id_str"] = result_img_df["object_id"].astype(str) + "_" + result_img_df["day"].astype(str) + "_" + result_img_df["time"].astype(str)

result_mes_df = pd.DataFrame({"object_id": subject_id.astype(int),
                              "day": day_id.astype(int),
                              "time": time_id.astype(int),
                              "allodynia": area_mes_allod,
                              "sec_hyperalgesia": area_mes_hyper})

# pd.melt(result_mes_df,
#         id_vars=['object_id', 'day', 'time'],
#         value_vars=['allodynia', "sec_hyperalgesia"],
#         var_name='effect', value_name='mes_area_[mm^2]')

result_mes_df["id_str"] = result_mes_df["object_id"].astype(str) + "_" + result_mes_df["day"].astype(str) + "_" + result_mes_df["time"].astype(str)
# =============================================================================
# load pxl size
# =============================================================================
pxl_df = pd.read_csv("/Users/malkusch/PowerFolders/LaSCA/mechanic/pxl_size.csv")

pxl_df["id_str"] = pxl_df["object_id"].astype(str) + "_" + pxl_df["day"].astype(str) + "_" + pxl_df["time"].astype(str)

pxl_img_df = pd.merge(pxl_df, result_img_df[["id_str", "allodynia", "sec_hyperalgesia"]], how="left", on=["id_str"])

result_df = pd.melt(pxl_img_df,
                    id_vars=["id_str", 'object_id','day', 'time', 'pxlSize_mm'],
                    value_vars=['allodynia', "sec_hyperalgesia"],
                    var_name='effect', value_name='img_area_[pxl]')
result_df["img_area_[mm^2]"] = result_df["img_area_[pxl]"] * result_df["pxlSize_mm"] * result_df["pxlSize_mm"]
result_df.to_csv(
    "/Users/malkusch/PowerFolders/LaSCA/mechanic/projected_mechanic_area.csv")

pxl_mes_df = pd.merge(pxl_df, result_mes_df[["id_str", "allodynia", "sec_hyperalgesia"]], how="left", on=["id_str"])

result_df = pd.melt(pxl_mes_df,
                    id_vars=["id_str", 'object_id','day', 'time', 'pxlSize_mm'],
                    value_vars=['allodynia', "sec_hyperalgesia"],
                    var_name='effect', value_name='mes_area_[mm^2]')
# pxl_df = pd.merge(pxl_df, result_img_df, how="left", on=["id_str"])
result_df.to_csv("/Users/malkusch/PowerFolders/LaSCA/mechanic/mechanic_area.csv")

