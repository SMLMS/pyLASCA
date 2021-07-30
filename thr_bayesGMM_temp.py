#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 08:30:45 2021

@author: malkusch
"""
import numpy as np
import cv2 as cv
import pandas as pd
import os.path
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.mixture import BayesianGaussianMixture
from pyLASCA.source.bdr import Bayes_decision_rules
from pyLASCA.source.bayes_gmm import BGMM1d

# =============================================================================
# functions
# =============================================================================

def transform(x):
    return np.sqrt(x)

def backtransform(x):
    return x**2

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


def draw_image(img, contour):
    image = cv.drawContours(cv.cvtColor(img, cv.COLOR_GRAY2RGB), [
                            contour], 0, (65535, 1, 65535), 3)
    return(image)
    # plt.imshow(image)
    # plt.xticks([]), plt.yticks([])
    # plt.title("Contour image")
    # plt.show()


def extract_roi_pixel_values(img_raw, img_bin):
    # =============================================================================
    # extract pixel values
    # =============================================================================
    img_roi = np.multiply(img_raw, img_bin).astype(np.uint16)
    img_ar = np.array(img_roi)
    return(img_ar.reshape(-1))


def fit_vb_gmm(x, n_components=5, random_state=42):
    vb_gmm = BayesianGaussianMixture(n_components=n_components,
                                     covariance_type='spherical',
                                     weight_concentration_prior_type="dirichlet_process",
                                     init_params="random",
                                     weight_concentration_prior=1e-3,
                                     n_init=10,
                                     random_state=random_state,
                                     reg_covar=0,
                                     mean_precision_prior=.8,
                                     max_iter=1000)
    vb_gmm.fit(x)
    return(vb_gmm.weights_, vb_gmm.means_.reshape(-1), np.sqrt(vb_gmm.covariances_))


def bayesian_decision_boundary_thr(data_df, reference_mu):
    model_df = pd.DataFrame()
    model_df = data_df.copy()[data_df.omega > 0.0]
    model_df.loc[:, 'component'] = np.arange(model_df.shape[0])
    bdr = Bayes_decision_rules(model_df[["component", "omega", "mu", "sigma"]],
                               x_min=0,
                               x_max=30)
    bdr.fit()
    decision_boundaries = bdr.fitResult["value"].values
    valid_decision_boundaries = decision_boundaries[decision_boundaries > reference_mu]
    if valid_decision_boundaries.size == 0:
        bayesian_thr = np.nan
    else:
        bayesian_thr = np.min(
            decision_boundaries[decision_boundaries > reference_mu])
    return(bayesian_thr)

# =============================================================================
# define proband
# =============================================================================

# measurement_array = "SCTCAPS10"
# measurement_day = 1
# frame = 0

measurement_idx_arr = np.arange(19) + 1
day_arr = np.arange(2) + 1
for measurement_idx in measurement_idx_arr:
    measurement_name = str("SCTCAPS%02i" %(measurement_idx ))
    for measurement_day in day_arr:
        frame = 0

        # =============================================================================
        # init file names 
        # =============================================================================
        
        contour_plot_name = str("/Users/malkusch/PowerFolders/LaSCA/thr_gmm/%s/contour_%s_gmm_%i_frame%i.pdf" %
                                (measurement_name, measurement_name, measurement_day, frame))
        ts_area_plot_name = str("/Users/malkusch/PowerFolders/LaSCA/thr_gmm/%s/timeseries_area_%s_gmm_%i_frame%i.pdf" %
                                (measurement_name, measurement_name, measurement_day, frame))
        ts_signal_plot_name = str("/Users/malkusch/PowerFolders/LaSCA/thr_gmm/%s/timeseries_signal_%s_gmm_%i_frame%i.pdf" %
                                  (measurement_name, measurement_name, measurement_day, frame))
        ts_data_name = str("/Users/malkusch/PowerFolders/LaSCA/thr_gmm/%s/timeseries_%s_gmm_%i_frame%i.csv" %
                           (measurement_name, measurement_name, measurement_day, frame))
        kde_plot_name = str("/Users/malkusch/PowerFolders/LaSCA/thr_gmm/%s/kde_%s_gmm_%i_frame%i.pdf" %
                            (measurement_name, measurement_name, measurement_day, frame))
        bayes_kde_plot_name = str("/Users/malkusch/PowerFolders/LaSCA/thr_gmm/%s/bayes_kde_%s_gmm_%i_frame%i.pdf" %
                                  (measurement_name, measurement_name, measurement_day, frame))
        csv_name = str("/Users/malkusch/PowerFolders/LaSCA/thr_gmm/%s/%s_gmm_%i_frame%i.csv" %
                       (measurement_name, measurement_name, measurement_day, frame))
        
        # =============================================================================
        # load data
        # =============================================================================
        
        intensity_df = pd.DataFrame()
        baseline_name = str("/Users/malkusch/PowerFolders/LaSCA/rawData/%s/%s_baseline%i_frame%i.tiff" %
                            (measurement_name, measurement_name, measurement_day, frame))
        baseline_img = cv.imread(baseline_name, cv.IMREAD_ANYDEPTH)
        baseline_bin_img = binarize_image(baseline_img)
        baseline_contour = detect_contour(baseline_bin_img)
        # draw_image(baseline_img, baseline_contour)
        intensity_df["0"] = extract_roi_pixel_values(baseline_img, baseline_bin_img)
        
        
        
        fig1 = plt.figure()
        a = fig1.add_subplot(7, 1, 1)
        imgplot = plt.imshow(draw_image(baseline_img, baseline_contour))
        a.set_xticks([])
        a.set_yticks([])
        a.set_ylabel("0 min", fontsize=6)
        
        
        
        time_ar = [15, 30, 45, 60, 120, 240]
        i = 2
        for t in time_ar:
            timepoint_name = str("/Users/malkusch/PowerFolders/LaSCA/rawData/%s/%s_%i.%i_frame%i.tiff" %
                                 (measurement_name, measurement_name, measurement_day, t, frame))
            if os.path.isfile(timepoint_name):
                timepoint_img =  cv.imread(timepoint_name, cv.IMREAD_ANYDEPTH)
                timepoint_bin_img = binarize_image(timepoint_img)
                timepoint_contour = detect_contour(timepoint_bin_img)
                draw_image(timepoint_img, timepoint_contour)
                feature_name = str("%i" % (t))
                intensity_df[feature_name] = extract_roi_pixel_values(
                    timepoint_img, timepoint_bin_img)
                a = fig1.add_subplot(7, 1, i)
                imgplot = plt.imshow(draw_image(timepoint_img, timepoint_contour))
                a.set_xticks([])
                a.set_yticks([])
                a.set_ylabel(str("%i min" % (t)), fontsize=6)
                i += 1
            else:
                continue
        
        plt.savefig(contour_plot_name, dpi=300)
        plt.show()
        plt.close()
        
        intensity_df.replace(to_replace=0, value=np.nan, inplace=True)
        # =============================================================================
        # preprocessing
        # =============================================================================
        transformed_df = pd.DataFrame()
        for feature_name in intensity_df.columns:
            transformed_df[feature_name] = intensity_df[feature_name].transform(
                np.sqrt)
        
        
        # =============================================================================
        # analysis
        # =============================================================================
        model_df = pd.DataFrame()
        random_state = 42
        sample_size = 250
        n_components = 5
        n_repeats = 20
        for i in np.arange(n_repeats):
            random_state += i
            for feature_name in intensity_df.columns:
                x = resample(transformed_df[feature_name].dropna().to_numpy(),
                             n_samples=sample_size,
                             random_state=random_state)
                # omega, mu, sigma = fit_vb_gmm(x.reshape(-1, 1),
                #                               n_components=n_components,
                #                               random_state=random_state)
                
                vb_gmm = BGMM1d(data = x.reshape(-1, 1),
                                max_components = n_components,
                                seed = random_state)
                
                vb_gmm_df = vb_gmm.fitResult
                
                vb_gmm_df["time"] =  np.repeat(int(feature_name), n_components)
                vb_gmm_df["measurement"] = np.repeat(measurement_name, n_components)
                vb_gmm_df["sample_idx"] = np.repeat(i, n_components)
                vb_gmm_df["frame"] = np.repeat(frame, n_components)
                vb_gmm_df["global_pxl"] = np.repeat(
                    transformed_df[feature_name].dropna().shape[0], n_components)
                vb_gmm_df["optimal_components"] = np.repeat(vb_gmm.optimal_components() , n_components)
        
                # time_stamp = np.repeat(int(feature_name), n_components)
                # measurement = np.repeat(measurement_name, n_components)
                # sample_idx = np.repeat(i, n_components)
                # frame_number = np.repeat(frame, n_components)
                # sum_pxl = np.repeat(
                #     transformed_df[feature_name].dropna().shape[0], n_components)
        
                # vb_gmm_df = pd.DataFrame({"measurement": measurement,
                #                           "sample_idx": sample_idx,
                #                           "frame": frame_number,
                #                           "time": time_stamp,
                #                           "omega": np.round(omega, decimals=3)/np.sum(np.round(omega, decimals=3)),
                #                           "mu": mu,
                #                           "sigma": sigma,
                #                           "sum_pxl": sum_pxl})
                vb_gmm_df = vb_gmm_df.round(decimals=3)
                vb_gmm_df.sort_values(by=["mu"],
                                      ascending=True,
                                      inplace=True,
                                      ignore_index=True)
                
                # vb_gmm_df["component"] = np.arange(n_components)
        
                vb_gmm_df = vb_gmm_df[["measurement", "sample_idx", "frame",
                                       "time", "optimal_components", "component", "omega", "mu", "sigma", "global_pxl"]]
        
                model_df = model_df.append(vb_gmm_df, ignore_index=True)
                
        print(model_df)

        # =============================================================================
        # plot kde
        # =============================================================================
        
        kde_df = transformed_df.copy()
        
        kde_df = kde_df.melt(var_name='time', value_name='intensity')
        x_max = np.max(kde_df.intensity)
        
        fig2 = plt.figure()
        kde_ridgeline_grid = sns.FacetGrid(
            kde_df.dropna(), row="time", hue="time", aspect=5, height=1.25)
        kde_ridgeline_grid.map(sns.kdeplot, 'intensity', clip_on=False,
                               shade=True, alpha=0.7, lw=4, bw_method=.2)
        kde_ridgeline_grid.set_ylabels("kde")
        kde_ridgeline_grid.set(xlim=(0, x_max+0.1*x_max))
        kde_ridgeline_grid.savefig(kde_plot_name)
        plt.show()
        plt.close()
        
        # =============================================================================
        # bayes decision boundaries
        # =============================================================================
        
        model_df["bayes_bound"] = np.repeat(np.nan, model_df.shape[0])
        model_df["signal_thr"] = np.repeat(np.nan, model_df.shape[0])
        
        for i in np.arange(n_repeats):
            reference_df = pd.DataFrame()
            reference_df = model_df.copy()[(
                model_df.sample_idx == i) & (model_df.time == 0)]
            reference_df = reference_df[reference_df.omega ==
                                        np.max(reference_df.omega)]
            reference_mu = reference_df["mu"].values[0]
        
            for t in time_ar:
                thr_df = pd.DataFrame()
                thr_df = model_df.copy()[(model_df.sample_idx == i)
                                         & (model_df.time == t)]
                thr = bayesian_decision_boundary_thr(thr_df, reference_mu)
                model_df.loc[(model_df.sample_idx == i) & (
                    model_df.time == t), 'bayes_bound'] = thr
                model_df.loc[(model_df.sample_idx == i) & (
                    model_df.time == t), 'signal_thr'] = backtransform(thr)
        
        fig3 = plt.figure()
        bayes_ridgeline_grid = sns.FacetGrid(
            model_df[model_df.component == 0], row="time", hue="time", aspect=5, height=1.25)
        bayes_ridgeline_grid.map(sns.kdeplot, 'bayes_bound',
                                 clip_on=False, shade=True, alpha=0.7, lw=4, bw_method=.2)
        bayes_ridgeline_grid.set(xlim=(0, x_max+0.1*x_max))
        bayes_ridgeline_grid.set_xlabels("Bayesian Decision Boundary")
        bayes_ridgeline_grid.set_ylabels("kde")
        bayes_ridgeline_grid.savefig(bayes_kde_plot_name)
        plt.show()
        plt.close()
        
        # =============================================================================
        # save model df
        # =============================================================================
        
        model_df.to_csv(path_or_buf=csv_name)
        
        # =============================================================================
        # create time series
        # =============================================================================
        
        time_series_df = pd.DataFrame()
        time_series_df["sample_idx"] = np.arange(n_repeats)
        # time_series_df["0min_fraction"] = np.repeat(0.0, time_series_df.shape[0])
        time_series_df["0min_pixel"] = np.repeat(0.0, time_series_df.shape[0])
        time_series_df["0min_signal"] = np.repeat(0.0, time_series_df.shape[0])
        
        time_point_df = model_df.copy()
        time_point_df.bayes_bound.replace(to_replace=np.nan, value=np.inf, inplace=True)
        
        for time_point in time_ar:
            # fraction_name = str("%imin_fraction" % (time_point))
            # time_series_df[fraction_name] = 1.0 - time_point_df.copy()[(time_point_df.time == time_point) & (time_point_df.mu < time_point_df.bayes_bound)].groupby("sample_idx")["omega"].sum()
            
            area_name = str("%imin_pixel" % (time_point))
            signal_name = str("%imin_signal" % (time_point))
            # sum_pxl = model_df[(model_df.time==time_point) & (model_df.sample_idx==0) & (model_df.component==0)]["global_pxl"].values[0]
            # time_series_df[area_name] = time_series_df[fraction_name] * sum_pxl
            area = np.repeat(0, n_repeats)
            signal = np.repeat(0, n_repeats)
            thr_trans = time_point_df[(time_point_df["time"] == time_point) & (time_point_df["component"] == 0)]["bayes_bound"].values
            thr_signal = time_point_df[(time_point_df["time"] == time_point) & (time_point_df["component"] == 0)]["signal_thr"].values
            for i in np.arange(n_repeats):
                area[i] = np.sum(transformed_df[str(time_point)] >  thr_trans[i])
                signal[i] = np.sum(intensity_df[intensity_df[str(time_point)] >  thr_signal[i]][str(time_point)].dropna())
                                 
            time_series_df[area_name] = area
            time_series_df[signal_name] = signal
        
        # =============================================================================
        # save time series
        # =============================================================================
        
        time_series_df.to_csv(path_or_buf=ts_data_name)
        
        # =============================================================================
        # plot area timeseries
        # =============================================================================
        
        time_series_area_df = time_series_df.copy()[["0min_pixel",
                                                     "15min_pixel",
                                                     "30min_pixel",
                                                     "45min_pixel",
                                                     "60min_pixel",
                                                     "120min_pixel",
                                                     "240min_pixel"]]
        
        time_series_area_df.columns = [0,15,30,45,60,120,240]
        time_series_area_df = time_series_area_df.melt(var_name='time', value_name='intensity')
        
        
        
        # time_series_df = time_series_df[(time_series_df.omega > 1e-3) & (time_series_df.mu < 8)]
        # time_series_df = time_series_df.assign(intensity = lambda x: 1.0 - x.omega)
        # print(time_series_df)
        fig_4 = plt.figure()
        ts_plot = sns.lineplot(data=time_series_area_df,
                               x="time",
                               y="intensity",
                               estimator=np.median)
        ts_plot.set_xlabel("time [min]")
        ts_plot.set_ylabel("active area [pxl]")
        plt.savefig(ts_area_plot_name)
        plt.show()
        plt.close()
        
        # =============================================================================
        # plot signal timeseries
        # =============================================================================
        
        time_series_signal_df = time_series_df.copy()[["0min_signal",
                                                       "15min_signal",
                                                       "30min_signal",
                                                       "45min_signal",
                                                       "60min_signal",
                                                       "120min_signal",
                                                       "240min_signal"]]

        time_series_signal_df.columns = [0,15,30,45,60,120,240]
        time_series_signal_df = time_series_signal_df.melt(var_name='time', value_name='intensity')

        fig_5 = plt.figure()
        ts_plot = sns.lineplot(data=time_series_signal_df,
                               x="time",
                               y="intensity",
                               estimator=np.median)
        ts_plot.set_xlabel("time [min]")
        ts_plot.set_ylabel("cumulative signal [a.u.]")
        plt.savefig(ts_signal_plot_name)
        plt.show()
        plt.close()
        
        # =============================================================================
        # plot example model 
        # =============================================================================
        # feature_name = "15"
        # for random_state in np.arange(start=42, stop=47, step=1):
        for time_point in [0,15,30,45,60,120,240]:
            feature_name = str(time_point)
            random_state = 42
        
            gmm_example_name = str("/Users/malkusch/PowerFolders/LaSCA/thr_gmm/%s/bayes_kde_%s_gmm_%i_frame%i_time%s_state%i.pdf" %
                                      (measurement_name, measurement_name, measurement_day, frame, feature_name, random_state))
            
            x = resample(transformed_df[feature_name].dropna().to_numpy(),
                         n_samples=sample_size,
                         random_state=random_state)
            
            omega, mu, sigma = fit_vb_gmm(x.reshape(-1, 1),
                                          n_components=n_components,
                                          random_state=random_state)
            
            vbgmm_model_df = pd.DataFrame({"component": np.arange(np.shape(omega)[0]),
                                           "omega": omega,
                                           "mu": mu,
                                           "sigma": sigma})
            
            
            bdr = Bayes_decision_rules(vbgmm_model_df,
                                       x_min=0,
                                       x_max=x_max+0.1*x_max)
            bdr.fit()
            
            x = resample(transformed_df[feature_name].dropna().to_numpy(),
                         n_samples=sample_size,
                         random_state=random_state)
            fig6 = plt.figure()
            bdr.plot_model()
            sns.kdeplot(data = x, clip_on=False, shade=True, alpha=0.7, lw=4, bw_method=.05)
            plt.savefig(gmm_example_name)
            plt.show()
            plt.close()
