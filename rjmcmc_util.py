# -*- encoding: utf-8 -*-
'''
@File    :   resultset1d.py
@Author  :   Yang Liu (yang.liu3@halliburton.com) 
'''
# import lib below

import sys
import random
import numpy as np
import pandas as pd
from copy import deepcopy

RJMCMC_DEBUG = 0
RJMCMC_WARNING = 1
RJMCMC_ERROR = 2
RJMCMC_FATAL = 3

class Point1D:
    def __init__(self, x, y, n, w):
        self.x = x
        self.y = y
        self.n = n
        self.w = w

class DataSet1D:
    def __init__(self):
        self.npoints = 0
        self.xmin = 0.0
        self.xmax = 0.0
        self.ymin = 0.0
        self.ymax = 0.0
        self.points = None
        self.x = None
        self.y = None
        self.w = None
        self.sigma = None
        self.lambdamin = 0.0
        self.lambdamax = 0.0
        self.lambdastd = 0.0

def dataset1d_load_data_main(dataset, noise, noise_flag):
    d = DataSet1D()
    
    dataset_copy = deepcopy(dataset)
    dataset_copy = dataset_copy.reset_index(drop=True)
    if noise_flag:
        window_size = 5 # varies for different dataset
        padding = (window_size - 1) // 2  # Half of the window size
        padded_data = pd.concat([dataset_copy['Y'].head(padding), dataset_copy['Y'], dataset_copy['Y'].tail(padding)], ignore_index=True)
        smooth = np.convolve(padded_data, np.ones(window_size) / window_size, mode='valid')
        bias = smooth - dataset_copy['Y'] # store for noise estimation in calibration
        bias = bias.reset_index(drop=True)
        dataset_copy['bias'] = bias
        sigma = np.std(dataset_copy['bias'], ddof=1)
        dataset_copy['sigma'] = sigma
    else:
        dataset_copy['bias'] = 0.0
        dataset_copy['sigma'] = noise

    # Load Live Inc
    d.x = dataset_copy['X'].values
    d.y = dataset_copy['Y'].values
    d.w = [1.0]*len(d.x)
    d.n = dataset_copy['sigma'].values

    size = len(d.x)
    # d.n = [sigma]*size
    d.npoints = size
    d.xmin = min(d.x)
    d.xmax = max(d.x)
    d.ymin = min(d.y)
    d.ymax = max(d.y)
    d.points = [Point1D(d.x[i], d.y[i], d.n[i], d.w[i]) for i in range(len(d.x))]

    # scaling factor of fixed noise
    d.lambdamin = 0.1  # min scaling of mean
    d.lambdamax = 1.0  # max scaling of mean
    d.lambdastd = 0.1  # scaling of standard deviation (0,1)
    # d.lambdamin = 0.0
    # d.lambdamax = 0.0
    # d.lambdastd = 0.0
    
    return d

# check
def dataset1d_range(data, xl, xr):
    xi = 0
    while xi < data.npoints and data.points[xi].x < xl:
        xi += 1
    if xi == data.npoints:
        return -1
    
    xj = data.npoints - 1
    while xj > xi and data.points[xj].x > xr:
        xj -= 1

    return xj - xi + 1, xi, xj

def rjmcmc_seed(s):
    random.seed(s)

# check
def rjmcmc_uniform():
    return random.random()

# check
def rjmcmc_normal():
    return random.gauss(0, 1)

# check
def rjmcmc_random_choose_int(low, high, custom_rand):
    r = low + int(custom_rand() * (high - low + 1))
    return r

# check
def rjmcmc_random_choose_double(low, high, custom_rand):
    return low + (high - low) * custom_rand()

# check
def rjmcmc_random_choose_interval(cdf, n, custom_rand):
    # iterates through the cumulative distribution function until it finds a interval where u is less than CDF
    # and return the index of the interval (order)
    u = custom_rand()
    for i in range(n):
        if u < cdf[i]:
            return i
    return -1

def rjmcmc_out(level, fmt, *args):
    if level == RJMCMC_DEBUG:
        sys.stderr.write("debug:")
    elif level == RJMCMC_WARNING:
        sys.stderr.write("warning:")
    elif level == RJMCMC_ERROR:
        sys.stderr.write("error:")
    elif level == RJMCMC_FATAL:
        sys.stderr.write("fatal:")
    else:
        sys.stderr.write("unknown:")

    sys.stderr.write(fmt % args)

rd_default = rjmcmc_out
rd_level = RJMCMC_ERROR

def rjmcmc_fatal(fmt, *args):
    assert rd_default is not None
    rd_default(RJMCMC_FATAL, fmt, *args)

# check
def rjmcmc_error(fmt, *args):
    assert rd_default is not None
    if rd_level >= RJMCMC_ERROR:
        rd_default(RJMCMC_ERROR, fmt, *args)

def rjmcmc_warning(fmt, *args):
    assert rd_default is not None
    if rd_level >= RJMCMC_WARNING:
        rd_default(RJMCMC_WARNING, fmt, *args)

def rjmcmc_debug(fmt, *args):
    assert rd_default is not None
    if rd_level >= RJMCMC_DEBUG:
        rd_default(RJMCMC_DEBUG, fmt, *args)

# check
def rjmcmc_polynomial_value(coeff, ncoeff, x):
    xs = coeff[0]
    xp = x
    for i in range(1, ncoeff):
        xs += coeff[i] * xp
        xp *= x
    return xs