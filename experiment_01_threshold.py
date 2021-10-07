#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 08:13:06 2021

@author: malkusch
"""
import numpy as np
import cv2 as cv
import pandas as pd
import seaborn as sns
import os.path
from scipy.stats import ks_2samp
from matplotlib import pyplot as plt

from sklearn.utils import resample
from sklearn.mixture import BayesianGaussianMixture
from pyLASCA.source.bdr import Bayes_decision_rules
from pyLASCA.source.bayes_gmm import BGMM1d

# =============================================================================
# functions
# =============================================================================

# def transform(x):
#     return np.sqrt(x)

# def backtransform(x):
#     return x**2

def clean_defect_pixels(img_raw):
    # =============================================================================
    # remove dead pxl by convolution with median filter
    # =============================================================================

    img2 = cv.medianBlur(img_raw, ksize=3)
    return img2

def binarize_image(img_raw):
    # =============================================================================
    # create binary image
    # dilate and erode (close)
    # =============================================================================
    img1 = clean_defect_pixels(img_raw)
    
    thr_abs = 1
    ret, img2 = cv.threshold(img1, thr_abs, 1, cv.THRESH_BINARY)
    img3 = img2.astype('uint8')
    img4 = cv.dilate(img3, None, iterations=1)
    img_bin = cv.erode(img4, None, iterations=1)
    return img_bin


def detect_contour(img_bin):
    # =============================================================================
    # detect contours
    # =============================================================================

    img5, contours, hierarchy = cv.findContours(
        img_bin, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    # =============================================================================
    # sort contours by area
    # =============================================================================

    contours = sorted(contours, key=lambda x: cv.contourArea(x))
    contour = contours[-1]
    return contour


# def draw_image(img, contour):
#     image = cv.drawContours(cv.cvtColor(img, cv.COLOR_GRAY2RGB), [
#                             contour], 0, (65535, 1, 65535), 3)
#     return(image)
#     # plt.imshow(image)
#     # plt.xticks([]), plt.yticks([])
#     # plt.title("Contour image")
#     # plt.show()


def extract_roi_pixel_values(img_raw, img_bin):
    # =============================================================================
    # extract pixel values
    # =============================================================================
    img_roi = np.multiply(img_raw, img_bin).astype(np.uint16)
    img_ar = np.array(img_roi)
    return(img_ar.reshape(-1))


def bayesian_decision_boundary_thr(data_df, reference_mu):
    model_df = pd.DataFrame()
    model_df = data_df.copy()[data_df.omega > 0.0]
    model_df.loc[:, 'component'] = np.arange(model_df.shape[0])
    bdr = Bayes_decision_rules(model_df[["component", "omega", "mu", "sigma"]],
                               x_min=0,
                               x_max=np.sqrt(2**16))
    bdr.fit()
    decision_boundaries = bdr.fitResult["value"].values
    valid_decision_boundaries = decision_boundaries[decision_boundaries > reference_mu]
    if valid_decision_boundaries.size == 0:
        bayesian_thr = np.sqrt(2**16)
    else:
        bayesian_thr = np.min(valid_decision_boundaries)
    return(bayesian_thr)

def distribution_faithful_resampling(x, sample_size = 250, random_state=42):
    alpha = 0.01
    success = False
    while (success == False):
        random_state += 1
        x_sample = resample(x, n_samples=sample_size, random_state=random_state)
        statistic, pvalue = ks_2samp(x_sample, x, alternative='two-sided', mode='auto')
        if(pvalue > alpha): success = True
    return(x_sample, random_state)

def intensity_threshold(x, random_state = 42):
    omega_thr = 0.1
    sample_size = 250
    n_components = 5
    n_repeats = 2#20
    bdb = np.zeros(n_repeats)
    for i in np.arange(n_repeats):
        random_state += i
        x_sample, random_state = distribution_faithful_resampling(x, sample_size, random_state)
        vb_gmm = BGMM1d(data = x_sample.reshape(-1, 1),
                        max_components = n_components,
                        seed = random_state)
        vb_gmm_df = vb_gmm.fitResult
        vb_gmm_df["optimal_components"] = np.repeat(vb_gmm.optimal_components() , n_components)
        vb_gmm_df["sample_idx"] = np.repeat(i, n_components)
        vb_gmm_df.sort_values(by="mu", axis=0, ascending=True, inplace=True, ignore_index=True)
        vb_gmm_df["component"] = np.arange(n_components)
        reference_mu = vb_gmm_df[vb_gmm_df["omega"] > omega_thr]["mu"].values[0]
        bdb[i] = bayesian_decision_boundary_thr(vb_gmm_df, reference_mu)
        
    return(np.mean(bdb), np.std(bdb))


# =============================================================================
# load data 
# =============================================================================
folder_base_name = "/Users/malkusch/PowerFolders/LaSCA/mechanic"
file_name = "mechanic_and_projected_area_combined.csv"
data_df = pd.read_csv(str("%s/%s" %(folder_base_name, file_name)))

# =============================================================================
# lumpensammler 
# =============================================================================
intensity_df = pd.DataFrame()
data_df["area_lasca_[pxl]"] = np.nan
data_df["area_lasca_[mm^2]"] = np.nan
data_df["mean_bdb"] = np.nan
data_df["std_bdb"] = np.nan
data_df["int_thr"] = np.nan
data_df["signal_lasca"] = np.nan

mean_bdb = data_df["mean_bdb"].values
std_bdb = data_df["std_bdb"].values
int_thr = data_df["int_thr"].values
area = data_df["area_lasca_[pxl]"].values
signal = data_df["signal_lasca"].values
# =============================================================================
# data preprocessing
# =============================================================================
for i in range(data_df.shape[0]):
    baseline_df = pd.DataFrame()
    intensity_df = pd.DataFrame()
    measurement = data_df.iloc[i]
    proband_id = measurement["probandID"]
    day = measurement["day"]
    time_point = measurement["t_[min]"]
    frame_id = 0
    measurement_name = str("/Users/malkusch/PowerFolders/LaSCA/rawData/%s/%s_%i.%i_frame%i.tiff" %
                           (proband_id, proband_id, day, time_point, frame_id))
    baseline_name = str("/Users/malkusch/PowerFolders/LaSCA/rawData/%s/%s_baseline%i_frame%i.tiff" %
                        (proband_id, proband_id, day, frame_id))
    if os.path.isfile(measurement_name):
        raw_bsl =  cv.imread(baseline_name, cv.IMREAD_ANYDEPTH)
        bin_bsl = binarize_image(raw_bsl)
        baseline_df["raw"] = extract_roi_pixel_values(raw_bsl, bin_bsl)
        baseline_df["roi"] =  baseline_df["raw"].replace(to_replace=0, value=np.nan)
        baseline_df["transformed"] = baseline_df["roi"].transform(np.sqrt)
        # reference_mu = np.mean(baseline_df["transformed"].dropna().values)
        
        raw_img =  cv.imread(measurement_name, cv.IMREAD_ANYDEPTH)
        bin_img = binarize_image(raw_img)
        intensity_df["raw"] = extract_roi_pixel_values(raw_img, bin_img)
        intensity_df["roi"] =  intensity_df["raw"].replace(to_replace=0, value=np.nan)
        intensity_df["transformed"] = intensity_df["roi"].transform(np.sqrt)
        mean_bdb[i], std_bdb[i] = intensity_threshold(intensity_df["transformed"].dropna().to_numpy())
        int_thr[i] = np.square(mean_bdb[i])
        area[i] = intensity_df[intensity_df["roi"] > int_thr[i]].shape[0]
        signal[i] = intensity_df[intensity_df["roi"] > int_thr[i]]["roi"].sum()


data_df["mean_bdb"] = mean_bdb
data_df["std_bdb"] = std_bdb
data_df["int_thr"] = int_thr
data_df["area_lasca_[pxl]"] = area
data_df["area_lasca_[mm^2]"] = np.square(data_df["pxl_[mm]"]) * data_df["area_lasca_[pxl]"]
data_df["signal_lasca"] = signal


# =============================================================================
# save results
# =============================================================================
result_file_name = "mechanic_and_lasca_area_combined.csv"
data_df.to_csv(str("%s/%s" %(folder_base_name, result_file_name)))