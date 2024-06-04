# -*- encoding: utf-8 -*-
'''
@File    :   rjmcmc_main.py
@Author  :   Yang Liu (yang.liu3@halliburton.com)
'''
# import lib below
 
import os
import sys
import numpy as np
import pandas as pd
from copy import deepcopy
from scipy import optimize
import matplotlib.pyplot as plt
from pre_processing_rj import *
from part1d_rjmcmc import part1d_rjmcmc
from toolbox import cosd, sind, calAngDif, calAngMean, save_to_pickle, getSRratio, getSRinfo, ssfunMMbp, selectprior, modelfunMM, plotcalfitMM

cwd = os.getcwd()
filepath = os.path.join(cwd, 'test.csv')
dataset = pd.read_csv(filepath)

# Scatter plot
# plt.scatter(dataset['X'], dataset['Y'])
# plt.show()

noise = 0.5
noise_flag = 0 # 0 for constant noise, 1 for noise sampling
flagplotcal = 1
window_size = 10 # number of data pts within the window

for i in range (7, len(dataset['X'])-window_size+1):
    start_idx = i
    end_idx = start_idx + window_size
    dataset_window = dataset.iloc[start_idx:end_idx]



    data = part1d_rjmcmc(dataset_window, noise, noise_flag, flagplotcal)
    
    i += 1

print('Calculation completed')