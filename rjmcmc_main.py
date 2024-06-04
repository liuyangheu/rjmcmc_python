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
filepath = os.path.join(cwd, 'run76_LY2.csv')
dataset = pd.read_csv(filepath)

# Scatter plot
plt.scatter(dataset['X'], dataset['Y'])
plt.show()

noise = 20.0 #(20-50)
noise_flag = 1 # 0 for constant noise, 1 for estimate from historical data
flagplotcal = 1
min_partitions = 2
max_partitions = 10
max_order = 2 # The order of the fitting model, 1 - first order, 2 - second order

data = part1d_rjmcmc(dataset, noise, noise_flag, flagplotcal, min_partitions, max_partitions, max_order)

print('Calculation completed')