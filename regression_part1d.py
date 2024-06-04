# -*- encoding: utf-8 -*-
'''
@File    :   regression_part1d.py
@Author  :   Yang Liu (yang.liu3@halliburton.com) 
'''
# import lib below

import math
import random
from engine import *
from resultset1d import *
from part1d_regression_rj import *
from rjmcmc_util import *

# part1d_init, select, perturb, misfit, accept, and sample


class RJMCMCCallback:
    def __init__(self):
        self.arg = None
        self.current_like = 0.0
        self.initialize_state = None
        self.select_process = None
        self.perturb_process = None
        self.compute_misfit = None
        self.accept = None
        self.sample = None

class Part1D:
    def __init__(self):
        self.results = ResultSet1D()  # Initialize the correct type of each attributes
        self.current = Part1dRegressionRJ()
        self.current_like = 0.0
        self.proposed = Part1dRegressionRJ()
        self.proposed_like = 0.0
        self.nprocesses = 0
        self.out = 0
        self.accepted = 0
        self.process = 0
        self.birth_prob = 0.0
        self.death_prob = 0.0
        self.move_prob = 0.0
        self.value_prob = 0.0
        self.lambda_prob = 0.0
        self.dataset = DataSet1D()
        self.dy = 0.0
        self.random = rjmcmc_uniform
        self.normal = rjmcmc_normal
        self.xsamples = 0
        self.v = None
        self.dk = 0.0
        self.partitions = None

# make method encapsulate in the class since part1d is shared within all methods

# check
def part1d_init(self):
    n_init, _ = part1d_regression_rj_initialize(
        self.current, self.dataset, 1, self.random, self.normal)
    if n_init < 0:
        rjmcmc_error("part1d_init: failed to initialize\n")
        return -1
    self.current_like = part1d_regression_rj_misfit(
        self.current, self.dataset, 1)
    # self.current_like = part1d_regression_rj.misfit(self.current, self.dataset, 1)
    return self.current_like

# check
def part1d_select(self):
    return random.randint(0, self.nprocesses - 1)

# check
def part1d_perturb(self, proc):
    self.process = proc
    self.results.propose[self.process] += 1
    # resultset1d_propose(self.results, self.process)
    if proc == 0:
        # print('birth')
        self.out, self.birth_prob = part1d_regression_rj_propose_birth(
            self.current, self.proposed, self.dataset, 1, self.random, self.normal, self.birth_prob)
    elif proc == 1:
        # print('death')
        self.out, self.death_prob = part1d_regression_rj_propose_death(
            self.current, self.proposed, self.dataset, 1, self.random, self.normal, self.death_prob)
    elif proc == 2:
        # print('move')
        self.out, self.move_prob = part1d_regression_rj_propose_move(
            self.current, self.proposed, self.dataset, 1, self.random, self.normal, self.move_prob)
    elif proc == 3:
        # print('value')
        self.out, self.value_prob = part1d_regression_rj_propose_value(
            self.current, self.proposed, self.dataset, 1, self.random, self.normal, self.value_prob)
    elif proc == 4:
        # print('lambda')
        self.out, self.lambda_prob = part1d_regression_rj_propose_lambda(
            self.current, self.proposed, self.dataset, 1, self.random, self.normal, self.lambda_prob)
    else:
        return None
    return self.proposed

# check
def part1d_misfit(self, state):
    if self.out == 0:
        self.proposed_like = float('inf')
    else:
        self.proposed_like = part1d_regression_rj_misfit(
            self.proposed, self.dataset, 1)
    return self.proposed_like

# check
def part1d_accept(self, current_like, proposed_like):
    if self.out == 0:
        return 0
    u = math.log(self.random())
    if self.out == 0:
        return 0
    if self.process == 0:
        self.accepted = u < (
            math.log(self.birth_prob/self.dk) + current_like - proposed_like)
    elif self.process == 1:
        self.accepted = u < (
            math.log(self.death_prob*self.dk) + current_like - proposed_like)
    elif self.process == 2:
        self.accepted = u < (math.log(self.move_prob) +
                             current_like - proposed_like)
    elif self.process == 3:
        self.accepted = u < (math.log(self.value_prob) +
                             current_like - proposed_like)
    elif self.process == 4:
        self.accepted = u < (math.log(self.lambda_prob) +
                             current_like - proposed_like)
    else:
        return 0
    if self.accepted:
        part1d_regression_rj_clone(self.proposed, self.current)
        self.current_like = self.proposed_like
        # resultset1d_accept(self.results, self.process)
        if self.process < 0 or self.process >= self.results.nprocesses:
            rjmcmc_error("resultset1d_accept: invalid index\n")
        self.results.accept[self.process] += 1
    return self.accepted

# check
def part1d_sample(self, i):
    npart = self.current.npartitions - 1
    # calculate mean value self.v with current coefficient
    self.v, n = part1d_regression_rj_evaluate(
        self.current, 0, self.dataset.xmin, self.dataset.xmax, self.xsamples, self.v)
    if n < 0:
        print("part1d_sample: failed to evaluate current state")
        return -1
    # update self.mean and self.hist (put value into x-y resolution matrix)
    resultset1d_sample(self.results, i, self.v)
    # resultset1d_sample_gradient(self.results, i, self.v, self.models.a, self.p)
    # misfit = current_like
    
    if i < 0 or i >= self.results.total:
        rjmcmc_error("resulset1d_sample_misfit,lambda,npartitions: invalid index\n")
    self.results.misfit[i] = self.current_like
    # resultset1d_sample_misfit(self.results, i, self.current_like)
    self.results.lambda_[i] = self.current.models.lambda_
    # resultset1d_sample_lambda(self.results, i, self.current.models.lambda_)
    self.results.partitions[i] = npart
    # resultset1d_sample_npartitions(self.results, i, npart)
    
    if self.current.models.order[0] < 0 or self.current.models.order[0] >= self.results.maxorder + 1:
        rjmcmc_error("resultset1d_sample_order: invalid order\n")
    self.results.order[self.current.models.order[0]] += 1
    # resultset1d_sample_order(
        # self.results, i, self.current.models.order[0])

    for j in range(2, npart):
        # for j in range(2, npart+1):
        # resultset1d_sample_partition_x(self.results, i, part1d_regression_rj_partition_position(self.current, j))
        # loc = part1d_regression_rj_partition_position(self.current, j)
        loc = position_map1d_position_of_index(self.current.p, j)
        resultset1d_sample_partition_x(self.results, i, loc)
        # ind = self.current.p.ind[self.current.p.pos.index(loc)]
        # resultset1d_sample_gradient(self.results, i, self.current.models.a, loc, ind)
    # do_user_callback(self)
    return 0

# check
def part1d_regression(dataset, burnin, total, min_part, max_part, max_order, xsamples, ysamples, credible_interval, pd, random, normal, results):
    # Create the class contains both current and proposed states, simulation variables
    part1d = Part1D()
    # determin number of processes
    if dataset.lambdastd == 0.0:
        part1d.nprocesses = 4
    else:
        part1d.nprocesses = 5
    part1d.results = resultset1d_create(burnin, total, xsamples, ysamples, part1d.nprocesses, max_part,
                                        max_order, dataset.xmin, dataset.xmax, dataset.ymin, dataset.ymax, credible_interval, results)
    part1d.current = part1d_regression_rj_create(
        min_part, max_part, max_order, 1, dataset.xmin, dataset.xmax, pd)
    part1d.proposed = part1d_regression_rj_create(
        min_part, max_part, max_order, 1, dataset.xmin, dataset.xmax, pd)
    part1d.dataset = dataset
    part1d.dy = dataset.ymax - dataset.ymin
    # need to double check the usage for uniform and normal
    part1d.random = rjmcmc_uniform
    part1d.normal = rjmcmc_normal
    part1d.xsamples = xsamples
    part1d.v = np.zeros(xsamples)
    part1d.dk = float(max_order + 1)
    part1d.partitions = np.zeros(max_part)
    cb = RJMCMCCallback()
    # Set the engine callbacks
    cb.arg = part1d  # input argument
    # initial state and likelihood, the initial likelihood is returned from
    cb.initialize_state = part1d_init
    # this function and stored in cb.current_like.
    # select one of the pertubation process (0 to s.nprocesses - 1)
    cb.select_process = part1d_select
    # that will be applied to the current state in the next iteration. The selected process is returned.
    # generate perturbation based on selected process, propsosed state is returend
    cb.perturb_process = part1d_perturb
    cb.compute_misfit = part1d_misfit     # likehood of the propoed state
    cb.accept = part1d_accept             # accpet/reject the proposed state
    # sample current state and store the sampled values to s.results
    cb.sample = part1d_sample

    if rjmcmc_engine_run(cb, burnin, total, 1) < 0:
        return None
    resultset1d_assemble_results(part1d.results)

    return part1d.results
