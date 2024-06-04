# -*- encoding: utf-8 -*-
'''
@File    :   resultset1d.py
@Author  :   Yang Liu (yang.liu3@halliburton.com) 
'''
# import lib below

import math
from rjmcmc_util import *

class ResultSet1D:
    def __init__(self):
        self.results = 0
        self.burnin = 0
        self.total = 0
        self.xsamples = 0
        self.ysamples = 0
        self.nprocesses = 0
        self.maxpartitions = 0
        self.maxorder = 0
        self.xmin = 0.0
        self.xmax = 0.0
        self.ymin = 0.0
        self.ymax = 0.0
        self.gradmin = 0.0
        self.gradmax = 0.0
        self.propose = []
        self.accept = []
        self.misfit = []
        self.lambda_ = []
        self.order = []
        self.partitions = []
        self.partition_x_hist = []
        self.mean = []
        self.hist = []
        self.mode = []
        self.median = []
        self.conf_interval = None
        self.conf_min = []
        self.conf_max = []
        self.gradient = []
        self.gradient_hist = []
        self.gradient_conf_min = []
        self.gradient_conf_max = []

#check
# Encapsulates the ResultSet1D object within the function, will not share data or state unluess exlicitly returned
def resultset1d_create(burnin, total, xsamples, ysamples, nprocesses, maxpartitions, maxorder, xmin, xmax, ymin, ymax, credible_interval, results):
    self = ResultSet1D()
    self.results = results
    self.burnin = burnin
    self.total = total
    self.xsamples = xsamples
    self.ysamples = ysamples
    # self.gsamples = gsamples
    self.nprocesses = nprocesses
    self.maxpartitions = maxpartitions
    self.maxorder = maxorder
    self.xmin = xmin
    self.xmax = xmax
    self.ymin = ymin
    self.ymax = ymax
    self.gradmin = -(self.ymax - self.ymin) / ((self.xmax - self.xmin) / float(self.maxpartitions + 1))
    self.gradmax = (self.ymax - self.ymin) / ((self.xmax - self.xmin) / float(self.maxpartitions + 1))
    self.propose = [0] * self.nprocesses
    self.accept = [0] * self.nprocesses
    self.misfit = [0.0] * self.total
    self.lambda_ = [0.0] * self.total
    if self.maxorder >= 0:
        self.order = [0] * (self.maxorder + 1)
    else:
        self.order = None
    if self.maxpartitions > 0:
        self.partitions = [0] * self.total
        self.partition_x_hist = [0] * self.xsamples
    else:
        self.partitions = None
        self.partition_x_hist = None
    # if self.results & rd.RESULTSET1D_MEAN:
    if 'MEAN' in self.results:
        self.mean = [0.0] * self.xsamples
    else:
        self.mean = None
    if 'MEDIAN' in self.results or 'MODE' in self.results or 'CREDIBLE' in self.results:
    # if self.results & (results['MEDIAN'] | results['MODE'] | results['CREDIBLE']):
        self.hist = [[0] * self.ysamples for _ in range(self.xsamples)]
    else:
        self.hist = None
    if 'MODE' in self.results:
        self.mode = [0.0] * self.xsamples
    else:
        self.mode = None
    if 'MEDIAN' in self.results:
        self.median = [0.0] * self.xsamples
    else:
        self.median = None
    if 'CREDIBLE' in self.results:
        self.conf_interval = credible_interval
        self.conf_min = [0.0] * self.xsamples
        self.conf_max = [0.0] * self.xsamples
        self.sigma = [0.0] * self.xsamples
    else:
        self.conf_interval = None
        self.conf_min = None
        self.conf_max = None
    if 'GRADIENT' in self.results:
        self.gradient = [0.0] * self.xsamples
    else:
        self.gradient = None
    if 'GRADIENT' in self.results or 'CREDIBLE' in self.results:
    # if self.results & (results['GRADIENT'] | results['CREDIBLE']):
        # self.gradient_hist = [[0] * self.gsamples for _ in range(self.xsamples)]
        self.gradient_hist = [[0] * self.ysamples for _ in range(self.xsamples)]
        self.gradient_conf_min = [0.0] * self.xsamples
        self.gradient_conf_max = [0.0] * self.xsamples
    else:
        self.gradient_hist = None
        self.gradient_conf_min = None
        self.gradient_conf_max = None
    
    return self

# check
def resultset1d_sample(r, i, v):
    if i >= r.total:
        print("resultset1d_sample_gradient: invalid index")
        return

    if i >= r.burnin:
        if r.mean:
            r.mean = [r.mean[j] + v[j] for j in range(r.xsamples)]

        if r.hist is not None:
            for j in range(r.xsamples):
                r.hist[j][rjmcmc_map_to_index(v[j], r.ymin, r.ymax, r.ysamples)] += 1

# check
def resultset1d_sample_partition_x(r, i, x):
    if r.xmin < x < r.xmax:
        j = rjmcmc_map_to_index(x, r.xmin, r.xmax, r.xsamples)
        r.partition_x_hist[j] += 1
# check
def resultset1d_assemble_results(r):
    denom = r.total - r.burnin

    if r.mean:
        r.mean = [mean / denom for mean in r.mean]

    if r.gradient:
        r.gradient = [grad / denom for grad in r.gradient]

    if r.median:
        r.median = [rjmcmc_median_from_histogram(hist, r.ymin, r.ymax, r.ysamples) for hist in r.hist]

    if r.mode:
        r.mode = [rjmcmc_mode_from_histogram(hist, r.ymin, r.ymax, r.ysamples) for hist in r.hist]

    if r.conf_min and r.conf_max:
        conf_samples = int((r.total - r.burnin) * (1.0 - r.conf_interval) / 2.0)
        
        for j in range(r.xsamples):
            r.conf_min[j] = rjmcmc_head_from_histogram(r.hist[j], r.ymin, r.ymax, r.ysamples, conf_samples)
            r.conf_max[j] = rjmcmc_tail_from_histogram(r.hist[j], r.ymin, r.ymax, r.ysamples, conf_samples)
            r.sigma[j] = (r.conf_max[j] - r.conf_min[j]) / (2 * 1.96)  # 1.96 corresponds to 95% confidence interval
    # if r.gradient_conf_min and r.gradient_conf_max:
    #     conf_samples = int((r.total - r.burnin) * (1.0 - r.conf_interval) / 2.0)

    #     r.gradient_conf_min = [rjmcmc_head_from_histogram(hist, r.gradmin, r.gradmax, r.ysamples, conf_samples) for hist in r.gradient_hist]
    #     r.gradient_conf_max = [rjmcmc_tail_from_histogram(hist, r.gradmin, r.gradmax, r.ysamples, conf_samples) for hist in r.gradient_hist]

# check
def rjmcmc_map_to_index(v, vmin, vmax, n):
    i = int(n * (v - vmin) / (vmax - vmin))
    if i < 0:
        return 0
    if i > (n - 1):
        return n - 1
    
    return i

# check
def rjmcmc_median_from_histogram(hist, ymin, ymax, num_bins):
    total = sum(hist)
    target = math.ceil(total / 2.0)
    count = 0

    for i, value in enumerate(hist):
        count += value
        if count >= target:
            return rjmcmc_map_to_value(i, ymin, ymax, num_bins)

# check
def rjmcmc_mode_from_histogram(hist, ymin, ymax, num_bins):
    max_value = max(hist)
    mode_index = hist.index(max_value)
    return rjmcmc_map_to_value(mode_index, ymin, ymax, num_bins)

# check
def rjmcmc_head_from_histogram(hist, vmin, vmax, n, drop):
    i = 0
    ci = 0

    while i < n and ci < drop:
        if hist[i] + ci >= drop:
            break
        ci += hist[i]
        i += 1
    return (i / n) * (vmax - vmin) + vmin

# check
def rjmcmc_tail_from_histogram(hist, vmin, vmax, n, drop):
    i = n - 1
    ci = 0
    while i > 0 and ci < drop:
        if hist[i] + ci >= drop:
            break

        ci += hist[i]
        i -= 1
    return (i / n) * (vmax - vmin) + vmin

# check
def rjmcmc_map_to_value(index, ymin, ymax, num_bins):
    return ymin + (ymax - ymin) * (index + 0.5) / num_bins