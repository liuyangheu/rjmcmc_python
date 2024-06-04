# -*- encoding: utf-8 -*-
'''
@File    :   engine.py
@Author  :   Yang Liu (yang.liu3@halliburton.com) 
'''
# import lib below

from rjmcmc_util import rjmcmc_error

# check
def rjmcmc_engine_run(cb, burn_iterations, total_iterations, sample_rate):
    if rjmcmc_engine_init(cb, burn_iterations, total_iterations, sample_rate) < 0:
        return -1
    while (i := rjmcmc_engine_step(cb)) == 1:
        pass
    if i < 0:
        return -1
    
    return 0

# check
def rjmcmc_engine_init(cb, burn_iterations, total_iterations, sample_rate):
    if cb == None:
        print("rjmcmc_engine_init: null callback")
        return -1
    if burn_iterations >= total_iterations:
        print("rjmcmc_engine_init: number of iterations must be greater than burnin")
        return -1
    cb.current_like = cb.initialize_state(cb.arg)
    if cb.current_like <= 0.0:
        print("rjmcmc_engine_init: invalid initial misfit value")
        return -1
    cb.burnin = burn_iterations
    cb.total = total_iterations
    cb.sample_rate = sample_rate
    cb.i = 0
    return 0

# check
def rjmcmc_engine_step(cb):
    state = None
    p = cb.select_process(cb.arg)
    if p < 0:
        rjmcmc_error("rjmcmc_engine_run: invalid process\n")
        return -1
    state = cb.perturb_process(cb.arg, p)
    if state != None:
        prop_like = cb.compute_misfit(cb.arg, state)
        if prop_like <= 0.0:
            rjmcmc_error("rjmcmc_engine_run: invalid misfit value\n")
            return -1
        if cb.accept(cb.arg, cb.current_like, prop_like):
            cb.current_like = prop_like
    if cb.sample(cb.arg, cb.i) < 0:
        rjmcmc_error("rjmcmc_engine_run: sampling error\n")
        return -1
    cb.i += 1
    print(cb.i)
    if cb.i >= cb.total:
        return 0
    return 1