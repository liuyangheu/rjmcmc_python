# -*- encoding: utf-8 -*-
'''
@File    :   part1d_regression_rj.py
@Author  :   Yang Liu (yang.liu3@halliburton.com) 
'''
# import lib below

import random
import math
import numpy as np
from curvefit import *
from position_map1d import *
from rjmcmc_util import *

DEFAULT_AUTO_Z = 3.0

# check
class Model:
    def __init__(self):
        self.a = [] # regression coefficient
        self.order = [] # order of each partition
        self.lambda_ = 0.0
        self.pk = []
        self.kcdf = []
        self.prior_product = []
        self.ppratio = []

# check
def models_create(max_part, max_order, ndatasets):

    m = Model()
    # regression coefficient
    m.a = np.zeros((max_part, max_order + 1))
    if m.a is None:
        return None

    m.order = np.zeros(max_part, dtype=int)
    if m.order is None:
        return None

    m.pk = np.zeros((max_part, max_order + 1))
    if m.pk is None:
        return None

    m.kcdf = np.zeros((max_part, max_order + 1))
    if m.kcdf is None:
        return None

    m.prior_product = np.zeros((max_part, max_order + 1))
    if m.prior_product is None:
        return None

    m.ppratio = np.zeros(max_part)
    if m.ppratio is None:
        return None
    
    return m

# check
def models_clone(max_partitions, max_order, ndatasets, src, dst):
    if src is None:
        rjmcmc_error("models_clone: src null\n")
    if dst is None:
        rjmcmc_error("models_clone: src null\n")

    for pi in range(max_partitions):
        for ci in range(max_order + 1):
            dst.a[pi][ci] = src.a[pi][ci]
        dst.order[pi] = src.order[pi]
        
        for oi in range(max_order + 1):
            dst.pk[pi][oi] = src.pk[pi][oi]
            dst.kcdf[pi][oi] = src.kcdf[pi][oi]
            dst.prior_product[pi][oi] = src.prior_product[pi][oi]

        dst.ppratio[pi] = src.ppratio[pi]

    dst.lambda_ = src.lambda_

# check
def models_delete(max_partitions, max_order, ndatasets, del_iy, npart, m):
    for pi in range(del_iy + 1, npart):
        for ci in range(max_order + 1):
            m.a[pi - 1][ci] = m.a[pi][ci]
        m.order[pi - 1] = m.order[pi]
        for oi in range(max_order + 1):
            m.pk[pi - 1][oi] = m.pk[pi][oi]
            m.kcdf[pi - 1][oi] = m.kcdf[pi][oi]
            m.prior_product[pi - 1][oi] = m.prior_product[pi][oi]
        m.ppratio[pi - 1] = m.ppratio[pi]

# check
class Part1dRegressionRJ:
    def __init__(self):
        self.min_partitions = 0
        self.max_partitions = 0
        self.max_order = 0
        self.ndatasets = 0
        self.xmin = 0.0
        self.xmax = 0.0
        self.pd = 0.0
        self.auto_z = 0.0
        self.npartitions = 0
        self.p = PositionMap1D()
        self.models = Model()
        self.cf = CurveFitResult()
        self.mean = None
        self.sigma = None
        self.autoprior = None
        self.detCm = None
        self.mean_misfit = None
        self.S = None

# check
def part1d_regression_rj_initialize(p, datasets, ndatasets, random, normal):
    # the partitions defined is not actually number of partitions but rather number of points
    ############## skip the loop if initial npart = 2 ###############################
    # initial 2 points, 1 partition
    npart = 2 #2, change to 3 for debugging, initialize the number of partitions
    for pi in range(2, npart):
        x = p.xmin + (p.xmax - p.xmin) * random() # Generate a random x value
        position_map1d_insert(p.p, x, pi) # Insert the x value into the partition map
        # Check if the partitions are valid, if not, delete the point and reduce npart
        if not partitions_valid(p, datasets, ndatasets):
            position_map1d_delete(p.p, x, pi)
            npart -= 1
            pi -= 1
    ##################################################################################
    p.npartitions = npart # Set the total number of partitions
    npart -= 1
    xl = position_map1d_position_of_index(p.p, 0)
    data = datasets
    if data.lambdastd > 0.0:
        p.models.lambda_ = rjmcmc_random_choose_double(data.lambdamin, data.lambdamax, random)
    
    for i in range(npart):
        # locate the predecessor pi
        pi = position_map1d_predecessor_of_point(p.p, xl)
        # locate the successor xr
        xr = position_map1d_next_position(p.p, xl)
        if xr < xl:
            rjmcmc_error("part1d_regression_rj_initialize: failed to get next point\n")
            return -1
        data = datasets
        # choose random order within the maximum order
        order = rjmcmc_random_choose_int(0, p.max_order, random)
        p.models.order[pi] = order
        # calculate the range _ and left xi and right bound xj of the current partition
        _, xi, xj = dataset1d_range(data, xl, xr)
        # update the curve properties and probability of each partition
        n, curve_prob = update_partition(p, data, pi, xi, xj, xl, xr, random, normal)
        if n < 0:
            rjmcmc_error("part1d_regression_rj_initialize: failed to update partition\n")
            return -1
    return 0, curve_prob

# check
def part1d_regression_rj_create(min_partitions, max_partitions, max_order, ndatasets, xmin, xmax, pd):
    r = Part1dRegressionRJ()
    r.min_partitions = min_partitions
    r.max_partitions = max_partitions
    r.max_order = max_order
    r.ndatasets = ndatasets
    r.xmin = xmin
    r.xmax = xmax
    r.pd = pd
    r.auto_z = DEFAULT_AUTO_Z
    r.npartitions = 0
    r.p = position_map1d_create(max_partitions, xmin, xmax)
    r.models = models_create(max_partitions, max_order, ndatasets)
    r.cf = curvefit_create(max_order)
    r.mean = np.zeros((max_order + 1, max_order + 1))
    r.sigma = np.zeros((max_order + 1, max_order + 1))
    r.autoprior = np.zeros(max_order + 1)
    r.detCm = np.zeros(max_order + 1)
    r.mean_misfit = np.zeros(max_order + 1)
    r.S = np.zeros((max_order + 1, max_order + 1))
    return r

# check
def part1d_regression_rj_clone(src, dst):
    if src is None:
        rjmcmc_error("part1d_regression_rj_clone: null src\n")
    if dst is None:
        rjmcmc_error("part1d_regression_rj_clone: null dst\n")
    if src.max_partitions != dst.max_partitions:
        rjmcmc_error("part1d_regression_rj_clone: size mismatch\n")
    if src.max_order != dst.max_order:
        rjmcmc_error("part1d_regression_rj_clone: order mismatch\n")
    if src.ndatasets != dst.ndatasets:
        rjmcmc_error("part1d_regression_rj_clone: count mismatch\n")

    position_map1d_clone(src.p, dst.p)
    models_clone(src.max_partitions,
                 src.max_order,
                 src.ndatasets,
                 src.models, dst.models)
    dst.npartitions = src.npartitions

# check
def part1d_regression_rj_evaluate(current, di, xmin, xmax, nsamples, samples):
    samples = []
    for i in range(nsamples):
        xi = xmin + (float(i) * (xmax - xmin)) / float(nsamples - 1)
        sample = value_at(current, xi)
        samples.append(sample)
    return samples, 0

# check
def part1d_regression_rj_misfit(p, datasets, ndatasets):
    sum = 0.0
    dsum = 0.0
    dataset = datasets

    l2 = 1.0
    if dataset.lambdastd > 0.0:
        l2 = p.models.lambda_ * p.models.lambda_
    for i in range(dataset.npoints):
        y = value_at(p, dataset.points[i].x)
        # penalty to force data within the range, if data with varies much at the range value, turn it off
        # if y > dataset.ymax or y < dataset.ymin:
        #     return float('inf')
        dy = y - dataset.points[i].y
        n = dataset.points[i].n
        w = dataset.points[i].w
        dsum += (w * dy * dy) / (2.0 * n * n * l2)
    sum += dsum
    return sum

# check
def value_at(current, x):
    iy = position_map1d_predecessor_of_point(current.p, x)
    if iy < 0:
        return -float('inf')
    if iy == 1:
        iy = position_map1d_predecessor_of_index(current.p, iy)
        if iy < 0 or iy == 1:
            return -float('inf')
    a = current.models.a[iy]
    order = current.models.order[iy]
    y = rjmcmc_polynomial_value(a, order + 1, x)
    return y

# check
def part1d_regression_rj_propose_birth(current, proposed, datasets, ndatasets, random, normal, birth_prob):
    if current.npartitions == current.max_partitions:
    # rjmcmc_error(
    #     "part1d_regression_rj_propose_birth: "
    #     "%d %d\n",
    #     current.npartitions,
    #     current.max_partitions)
        return 0, 1.0 #default prob to make return corret

    part1d_regression_rj_clone(current, proposed)
    
    new_x = random() * (proposed.xmax - proposed.xmin) + proposed.xmin
    new_iy = proposed.npartitions
    if position_map1d_insert(proposed.p, new_x, new_iy) < 0:
        rjmcmc_error("part1d_regression_rj_propose_birth: failed to add new point\n")
        return 0
    if not partitions_valid(proposed, datasets, ndatasets):
        return 0, 1.0 #default prob to make return corret
    proposed.npartitions += 1
    new_x_right = position_map1d_next_position(proposed.p, new_x)
    if new_x_right < new_x:
        rjmcmc_error("part1d_regression_rj_propose_birth: failed to find right extent of new point\n")
        return 0
    prev_iy = position_map1d_predecessor_of_index(proposed.p, new_iy)
    if prev_iy < 0:
        rjmcmc_error("part1d_regression_rj_propose_birth: failed to find predecessor\n")
        return 0
    # Store the prior/proposal ratio as it will be overwritten when the curves are resampled.
    prob = 1.0
    prob /= proposed.models.ppratio[prev_iy]

    # Update the partitions for the new partition (b)
    data = datasets
    n, xi, xj = dataset1d_range(data, new_x, new_x_right)
    if n <= 1:
        return 0, 1.0 #default prob to make return corret
    n_update, curve_prob = update_partition(proposed, data, new_iy, xi, xj, new_x, new_x_right, random, normal)
    if n_update < 0:
        rjmcmc_error("part1d_regression_rj_propose_birth: failed to update new partition\n")
        return 0
    prob *= curve_prob

    # Update the partition that the new partition affected (c)
    prev_x = position_map1d_position_of_index(proposed.p, prev_iy)

    data = datasets
    n, xi, xj = dataset1d_range(data, prev_x, new_x)
    if n <= 1:
        return 0, 1.0 #default prob to make return corret
    n_update, curve_prob = update_partition(proposed, data, prev_iy, xi, xj, prev_x, new_x, random, normal)
    if n_update < 0:
        rjmcmc_error("part1d_regression_rj_propose_birth: failed to update new partition\n")
        return 0
    prob *= curve_prob
    birth_prob = prob
    return -1, birth_prob

# check
def part1d_regression_rj_propose_death(current, proposed, datasets, ndatasets, random, normal, death_prob):
    part1d_regression_rj_clone(current, proposed)
    if proposed.npartitions <= 2 or proposed.npartitions <= proposed.min_partitions:
        return 0, 1.0 #default prob to make return corret

    del_iy = rjmcmc_random_choose_int(2, proposed.npartitions - 1, random)
    prob = 1.0
    prob /= proposed.models.ppratio[del_iy]
    deleted_pos = position_map1d_position_of_index(proposed.p, del_iy)
    if position_map1d_delete(proposed.p, deleted_pos, del_iy) < 0:
        rjmcmc_error("part1d_regression_rj_propose_death: failed to delete position\n")
        return 0
    models_delete(proposed.max_partitions, proposed.max_order, proposed.ndatasets, del_iy, proposed.npartitions, proposed.models)
    proposed.npartitions -= 1
    new_iy = position_map1d_predecessor_of_point(proposed.p, deleted_pos)
    if new_iy < 0:
        rjmcmc_error("part1d_regression_rj_propose_death: failed to find predecessor\n")
        return 0
    prob = 1.0
    xl = position_map1d_position_of_index(proposed.p, new_iy)
    xr = position_map1d_next_position(proposed.p, xl)

    data = datasets
    order = proposed.models.order[new_iy]
    n, xi, xj = dataset1d_range(data, xl, xr)
    n_update, curve_prob  = update_partition(proposed, data, new_iy, xi, xj, xl, xr, random, normal)
    if n_update < 0:
        rjmcmc_error("part1d_regression_rj_propose_death: failed to update partition\n")
        return 0
    prob *= curve_prob
    death_prob = prob
    return -1, death_prob

# check
def part1d_regression_rj_propose_move(current, proposed, datasets, ndatasets, random, normal, move_prob):
    if current.npartitions <= 2:
        return 0, 1.0
    part1d_regression_rj_clone(current, proposed)
    iy = rjmcmc_random_choose_int(2, proposed.npartitions - 1, random)
    old_x = position_map1d_position_of_index(proposed.p, iy)
    new_x = old_x + normal() * proposed.pd

    if new_x <= proposed.xmin or new_x >= proposed.xmax:
        return 0, 1.0 #default prob to make return corret
    
    iyop = position_map1d_predecessor_of_index(proposed.p, iy)
    if iyop < 0:
        rjmcmc_error("part1d_regression_rj_propose_move: failed to find old precedessor of point\n")
        return 0
    
    prob = 1.0
    prob /= proposed.models.ppratio[iy]
    prob /= proposed.models.ppratio[iyop]

    if position_map1d_move(proposed.p, old_x, new_x) < 0:
        rjmcmc_error("part1d_regression_rj_propose_move: failed to move point\n")
        return 0
    if not partitions_valid(proposed, datasets, ndatasets):
        return 0, 1.0 #default prob to make return corret
    iynp = position_map1d_predecessor_of_index(proposed.p, iy)
    if iynp < 0:
        rjmcmc_error("part1d_regression_rj_propose_move: failed to find new predecessor predecessor\n")
        return 0
    
    xl = position_map1d_position_of_index(proposed.p, iy)
    xr = position_map1d_next_position(proposed.p, xl)

    data = datasets
    n, xi, xj = dataset1d_range(datasets, xl, xr)
    if n <= 1:
        return 0, 1.0 #default prob to make return corret
    n_update, curve_prob = update_partition(proposed, data, iy, xi, xj, xl, xr, random, normal)
    if n_update < 0:
        rjmcmc_error("part1d_regression_rj_propose_move: failed to update part\n")
        return 0
    prob *= curve_prob
    
    xl = position_map1d_position_of_index(proposed.p, iyop)
    xr = position_map1d_next_position(proposed.p, xl)

    data = datasets
    n, xi, xj = dataset1d_range(datasets, xl, xr)
    if n <= 1:
        return 0, 1.0 #default prob to make return corret
    n_update, curve_prob = update_partition(proposed, data, iyop, xi, xj, xl, xr, random, normal)
    if n_update < 0:
        rjmcmc_error("part1d_regression_rj_propose_move: failed to update part\n")
        return 0
    prob *= curve_prob
    
    if iyop != iynp:
        prob /= proposed.models.ppratio[iynp]
        xl = position_map1d_position_of_index(proposed.p, iynp)
        xr = position_map1d_next_position(proposed.p, xl)
        data = datasets
        n, xi, xj = dataset1d_range(datasets, xl, xr)
        if n <= 1:
            return 0, 1.0 #default prob to make return corret
        n_update, curve_prob = update_partition(proposed, data, iynp, xi, xj, xl, xr, random, normal)
        if n_update < 0:
            rjmcmc_error("part1d_regression_rj_propose_move: failed to update part\n")
            return 0
        prob *= curve_prob
    move_prob = prob
    return 1, move_prob

def part1d_regression_rj_propose_value(current, proposed, datasets, ndatasets, random, normal, value_prob):
    part1d_regression_rj_clone(current, proposed)
    iy = rjmcmc_random_choose_int(0, proposed.npartitions - 2, random)
    if iy > 0:
        iy += 1
    xl = position_map1d_position_of_index(proposed.p, iy)
    xr = position_map1d_next_position(proposed.p, xl)
    order = proposed.models.order[iy]
    data = datasets
    n, xi, xj = dataset1d_range(datasets, xl, xr)
    if n <= 0:
        return 0, 1.0 #default prob to make return corret
    prob = 1.0 / proposed.models.ppratio[iy]
    n_update, curve_prob = update_partition(proposed, data, iy, xi, xj, xl, xr, random, normal)
    if n_update < 0:
        rjmcmc_error("part1d_regression_rj_propose_value: failed to update part\n")
        return 0
    value_prob = prob * curve_prob
    return 1, value_prob

def part1d_regression_rj_propose_lambda(current, proposed, datasets, ndatasets, random, normal, lambda_prob):
    data = datasets
    part1d_regression_rj_clone(current, proposed)
    new_s = proposed.models.lambda_ + normal() * data.lambdastd
    if new_s < data.lambdamin or new_s > data.lambdamax:
        return 0, 1.0 #default prob to make return corret
    lambda_prob = (proposed.models.lambda_ / new_s)**data.npoints
    proposed.models.lambda_ = new_s
    return -1, lambda_prob

# check
def resample_partition(proposed, data, pi, xi, xj, xl, xr, random, normal):
    # select an order using a random number generator
    order = rjmcmc_random_choose_interval(proposed.models.kcdf[pi], proposed.max_order + 1, random)
    # compute the curve fit using the selected order and provided data within the rangge (xi, xj)
    if curvefit_compute(data, xi, xj, order, proposed.cf) < 0:
        rjmcmc_error("update_partition: failed to compute curvefit (%d %d %d)\n", xi, xj, order)
        return -1
    
    # a = np.zeros(order + 1)
    # sample the curve and update the model information
    n, curve_prob = curvefit_sample(proposed.cf, normal, proposed.models.a[pi], order + 1)
    if n < 0:
        rjmcmc_error("update_partition: failed to sample curve\n")
        return -1
    
    # proposed.models[di].a[pi][:order+1] = a
    proposed.models.order[pi] = order
    proposed.models.ppratio[pi] = proposed.models.prior_product[pi][order] / (proposed.models.pk[pi][order] * curve_prob)
    if proposed.models.ppratio[pi] == 0.0:
        rjmcmc_error("ppratio underflow: %g %g %g\n", proposed.models.prior_product[pi][order], proposed.models.pk[pi][order], curve_prob)
    prob = proposed.models.ppratio[pi]
    return 0, prob

# check
def update_partition(proposed, data, pi, xi, xj, xl, xr, random, normal):
    # evaluate the curve properties and order probabilities within the specified partition
    if curvefit_evaluate_pk(proposed.cf, data, xi, xj, proposed.max_order, None, proposed.auto_z, proposed.mean_misfit, proposed.detCm, proposed.autoprior, proposed.S, proposed.models.pk[pi], proposed.models.kcdf[pi], proposed.mean, proposed.sigma) < 0:
        rjmcmc_error("update_partition: failed to determine pk\n")
        return -1
    
    for oi in range(proposed.max_order + 1):
        proposed.models.prior_product[pi][oi] = 1.0
        for i in range(oi + 1):
            proposed.models.prior_product[pi][oi] *= 2.0 * DEFAULT_AUTO_Z * proposed.sigma[oi][i]
    
    # resample the curve based on the calculated curve properties
    n_resample, prob_resample = resample_partition(proposed, data, pi, xi, xj, xl, xr, random, normal)
    if n_resample < 0:
        rjmcmc_error("update_partition: failed to resample curve\n")
        return -1
    return 0, prob_resample

class PartitionsValidData:
    def __init__(self):
        self.current = None
        self.datasets = None
        self.ndatasets = 0
        self.invalid_count = 0
        self.max_order = 0

# check
def partitions_valid_cb(user_arg, xmin, xmax, iy, riy):
    d = user_arg
    data = d.datasets
    c = 0
    for i in range(data.npoints):
        if data.points[i].x >= xmin and data.points[i].x <= xmax:
            c += 1
    if c <= d.max_order:
        d.invalid_count += 1
    return 0

# check
def partitions_valid(p, datasets, ndatasets):
    d = PartitionsValidData()
    d.current = p
    d.datasets = datasets
    d.ndatasets = ndatasets
    d.invalid_count = 0
    d.max_order = p.max_order
    if position_map1d_traverse_intervals(p.p, partitions_valid_cb, d) < 0:
        rjmcmc_error("partitions_valid: failed to traverse intervals\n")
        return 0
    return d.invalid_count == 0

class PartitionsValidData:
    def __init__(self, current, datasets, ndatasets, max_order):
        self.current = current
        self.datasets = datasets
        self.ndatasets = ndatasets
        self.invalid_count = 0
        self.max_order = max_order

# def partitions_valid_cb(user_arg, xmin, xmax, iy, riy):
#     d = user_arg
#     i = 0
#     data = d.datasets
#     c = 0
#     for i in range(data.npoints):
#         if data.points[i].x >= xmin and data.points[i].x <= xmax:
#             c += 1
#     if c <= d.max_order:
#         d.invalid_count += 1

#     return 0

def partitions_valid(p, datasets, ndatasets):
    d = PartitionsValidData(p, datasets, ndatasets, p.max_order)
    if position_map1d_traverse_intervals(p.p, partitions_valid_cb, d) < 0:
        rjmcmc_error("partitions_valid: failed to traverse intervals\n")
        return False
    return d.invalid_count == 0