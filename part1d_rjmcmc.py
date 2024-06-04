# -*- encoding: utf-8 -*-
'''
@File    :   part1d_rjmcmc.py
@Author  :   Yang Liu (yang.liu3@halliburton.com) 
'''
# import lib below

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from collections import Counter
from scipy.interpolate import interp1d
from toolbox import calAngMean, cosd, conffit_linear
from part1d_regression_rj import *
from regression_part1d import *
from rjmcmc_util import *
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import numpy as np


resultset1d_result_t = {
    'MEAN': 0x01,
    'MEDIAN': 0x02,
    'MODE': 0x04,
    'CREDIBLE': 0x08,
    'GRADIENT': 0x10
}

def get_result_mask(result_list):
    mask = 0
    for result in result_list:
        mask |= resultset1d_result_t.get(result.upper(), 0)
    return mask

def assign_mode_value(rjmcmc_mean, dc):
    for idx, row in rjmcmc_mean.iterrows():
        depth_rjmcmc = row['depth']
        filtered_dc = dc[dc['depth'] < depth_rjmcmc]
        
        if not filtered_dc.empty:
            closest_depth = filtered_dc['depth'].max()
            mode_value = dc.loc[dc['depth'] == closest_depth, 'val'].values[0]
            rjmcmc_mean.at[idx, 'mode'] = mode_value
            
    return rjmcmc_mean

def find_local_maxima(arr, n):
    maxima = []
    for i in range(len(arr)):
        # Check if the current index is a local maximum
        if all(arr[i] >= arr[j] for j in range(max(0, i - 3), min(len(arr), i + 4))):
            maxima.append((arr[i], i))
    
    # Sort the maxima based on frequency (value) in descending order
    maxima_sorted = sorted(maxima, key=lambda x: x[0], reverse=True)
    top_maxima = maxima_sorted[:n]
    top_maxima_sorted = sorted(top_maxima, key=lambda x: x[1])
    top_maxima_sorted = np.array([item[1] for item in top_maxima_sorted], dtype=np.int64) if top_maxima_sorted else np.array([], dtype=np.int64)
    return top_maxima_sorted

def get_segment_slope(change_points, x_input, rjmcmc_mean):
    TY_df = pd.DataFrame(np.nan, index=range(len(change_points) + 1), columns=['start', 'end', 'slope', 'pts_num'])
    
    if len(change_points) == 0:
        # If no change points, treat the entire DataFrame as a single segment
        segment = rjmcmc_mean
        modelnom, _, _, _, _ = conffit_linear(segment['depth'],segment['val'],level= 0.95, flagplot=0)
        toolyield_segment = modelnom[0]
        pts_num = len(segment)
        TY_df.iloc[0] = [segment['depth'].iloc[0], segment['depth'].iloc[-1], toolyield_segment, pts_num]
    else:
        for i in range(len(change_points)+1):
            start_idx = change_points[i - 1] if i > 0 else 0
            start_depth = rjmcmc_mean['depth'].iloc[start_idx]
            if i == len(change_points):
                end_idx = len(rjmcmc_mean)-1
            else:
                end_idx = change_points[i]
            
            segment = rjmcmc_mean.iloc[start_idx:end_idx]
            
            # Exclude rows where 'val' is None
            segment = segment.dropna(subset=['val'])
            pts_num = len(segment)
            # Check if the segment is not empty
            if not segment.empty:
                # Perform linear regression
                modelnom, _, _, _, _ = conffit_linear(segment['depth'],segment['val'],level= 0.95, flagplot=0)
                toolyield_segment = modelnom[0]
                    
            else:
                toolyield_segment = None
            TY_df.iloc[i] = [start_idx, end_idx, toolyield_segment, pts_num]

    return TY_df

import pandas as pd

# Function to calculate mean values for each plateau segment
def calculate_each_plateau(rjmcmc_mean, TY_df):
    # Initialize lists to store mean values
    plateau1 = []
    plateau2 = []
    transition_points = []
    transition_start = []
    transition_end = []
    
    i= 0
    while i < len(TY_df) and abs(TY_df['slope'].iloc[i]) < 0.1:
        start = int(TY_df['start'].iloc[i])
        end = int(TY_df['end'].iloc[i])
        plateau1.extend(rjmcmc_mean['val'].iloc[start:end])
        i += 1
    
    while i < len(TY_df):
        if abs(TY_df['slope'].iloc[i]) > 0.1:
            transition_start.append(TY_df['start'].iloc[i])
            transition_end.append(TY_df['end'].iloc[i])
        else:
            start = int(TY_df['start'].iloc[i])
            end = int(TY_df['end'].iloc[i])
            plateau2.extend(rjmcmc_mean['val'].iloc[start:end])
        i += 1
    
    transition_points = [transition_start[0], transition_end[-1]]
    plateau1_mean = sum(plateau1)/len(plateau1)
    plateau2_mean = sum(plateau2)/len(plateau2)
    # Iterate through TY_df to identify segments belonging to plateau 1 and plateau 2
    
    return transition_points, plateau1_mean, plateau2_mean


def plot_rjmcmc(x, meancurve, data, results):
    # Plot the data with black crosses and the mean with a red line
    fig1 = plt.figure(1)

    # fitted mean value and raw dataset
    a = plt.subplot(311)
    a.plot(data.x, data.y, 'ko', ms=2, label = 'Dataset')
    a.plot(x, meancurve, 'r-', label = 'Mean value')
    a.plot(x, results.conf_min, 'k:', label = 'Conf_min')
    a.plot(x, results.conf_max, 'k:', label = 'Conf_max')
    a.fill_between(x, results.conf_min, results.conf_max, color='gray', alpha=0.3)
    a.set_xlim(data.xmin-1.0, data.xmax+1.0)
    a.set_title('Regression Results of RJMCMC')
        
    a.set_xlabel('X')
    a.set_ylabel('Y')
    a.legend(fontsize = 8)

    # histogram of partition locations
    b = plt.subplot(312)
    b.bar(x, results.partition_x_hist, data.x[1] - data.x[0])
    b.set_xlim(data.xmin-1.0, data.xmax+1.0)
    b.set_title('Histogram of Partition Locations')
    b.set_xlabel('X')
    b.set_ylabel('Iterations')

    c = plt.subplot(313)
    c.plot(results.misfit)
    c.set_title('Misfit History')
    c.set_xlabel('Iteration')
    c.set_ylabel('Misfit')

    fig1.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.6)

    fig2 = plt.figure(2)
    # fitted mean value and raw dataset
    a = plt.subplot(111)
    a.plot(data.x, data.y, 'ko', ms=2, label = 'Dataset')
    a.plot(x, meancurve, 'r-', label = 'Mean value')
    a.plot(x, results.conf_min, 'k:', label = 'Conf_min')
    a.plot(x, results.conf_max, 'k:', label = 'Conf_max')
    a.fill_between(x, results.conf_min, results.conf_max, color='gray', alpha=0.3)
    a.set_xlim(data.xmin-1.0, data.xmax+1.0)
    a.set_title('Regression Results of RJMCMC')

    # historgram of partition numbers
    fig3 = plt.figure(3)
    d = plt.subplot(111)
    part_max = max(results.partitions)
    part_min = min(results.partitions)
    bins_part = part_max - part_min + 1
    d.hist(results.partitions, bins=bins_part, align='left', range=(part_min, part_max + 1))
    d.set_title('Histogram of Partition Numbers')
    d.set_xlabel('Number of Partitions')
    d.set_ylabel('Iterations')
    
    plt.show()

def part1d_rjmcmc(dataset, model_par):

    data = dataset1d_load_data_main(dataset, model_par.noise, model_par.noise_flag)

    results = get_result_mask(['mean', 'median', 'mode', 'credible', 'gradient']) # Which results to store during the excution of the regression algorithm

    # Execution of the rjmcmc algorithm
    results = part1d_regression(data,  
                                model_par.burnin, 
                                model_par.total,
                                model_par.min_partitions, 
                                model_par.max_partitions, 
                                model_par.max_order,
                                model_par.xsamples,
                                model_par.ysamples,
                                model_par.credible_interval,
                                model_par.p_d, 
                                rjmcmc_uniform, 
                                rjmcmc_normal, 
                                resultset1d_result_t)
    # Retrieve and plot the results
    x = np.linspace(results.xmin, results.xmax, results.xsamples)
    meancurve = results.mean
    rjmcmc_mean = pd.DataFrame({'depth': [], 'val': [], 'mode': []})
    rjmcmc_mean['depth'] = x
    rjmcmc_mean['val'] = meancurve
    
    parts_count = Counter(results.partitions)
    parts_num = parts_count.most_common(1)[0]
    maxima = find_local_maxima(results.partition_x_hist, parts_num[0]-1)
    x_input = data.x
    change_points = maxima
    idx_drop = np.where(np.diff(change_points) < 2)[0] + 1 #drop point when too close
    change_points = np.delete(change_points, idx_drop)
    change_points = np.sort(change_points)
    TY_df = get_segment_slope(change_points, x, rjmcmc_mean)
    # estimate noise level based on prediction
    interp_func = interp1d(x, results.mean, kind='linear', fill_value='extrapolate')
    interpolated_mean = interp_func(data.x)
    residuals = data.y - interpolated_mean
    
    transition, p1_mean, p2_mean = calculate_each_plateau(rjmcmc_mean, TY_df)
    
    if model_par.flagplot == 1:
        plot_rjmcmc(x, meancurve, data, results)
        
    return data