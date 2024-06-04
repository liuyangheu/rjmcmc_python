# -*- encoding: utf-8 -*-
'''
@File    :   position_map1d.py
@Author  :   Yang Liu (yang.liu3@halliburton.com) 
'''
# import lib below

from rjmcmc_util import *

class PositionMap1D:
    def __init__(self):
        self.max_partitions = 0
        self.npartitions = 0
        self.pos = []
        self.ind = []

# check
def position_map1d_create(max_partitions, minx, maxx):
    p = PositionMap1D()
    p.max_partitions = max_partitions
    if max_partitions < 2:
        rjmcmc_error("position_map1d_create: invalid no. partitions")
    if minx >= maxx:
        rjmcmc_error("position_map1d_create: maxx must be greater than minx")
    p.npartitions = 2
    p.pos = [minx, maxx]
    p.ind = [0, 1]
    return p

# check
def position_map1d_clone(src, dst):
        if src is None:
            rjmcmc_error("position_map1d_clone: null src")
        if dst is None:
            rjmcmc_error("position_map1d_clone: null dst")
        if src.max_partitions != dst.max_partitions:
            rjmcmc_error("position_map1d_clone: size mismatch")
        dst.npartitions = src.npartitions
        dst.pos = src.pos.copy()
        dst.ind = src.ind.copy()

# check
def position_map1d_insert(p, x, iy):
    insertion_index = 0
    while insertion_index < len(p.pos) and p.pos[insertion_index] <= x:
        insertion_index += 1
    p.pos.insert(insertion_index, x)
    p.ind.insert(insertion_index, iy)
    # p.ind = list(range(iy + 1))
    p.npartitions += 1
    return 0

# check
def position_map1d_delete(p, x, iy):
    if p is None:
        rjmcmc_error("position_map1d_delete: null map")
    if p.npartitions <= 2:
        rjmcmc_error("position_map1d_delete: min partitions")
    di = -1
    for i in range(p.npartitions):
        if p.pos[i] == x:
            di = i
            break
    if di < 0:
        rjmcmc_error("position_map1d_delete: failed to find point")
    for i in range(di, p.npartitions - 1):
        p.pos[i] = p.pos[i + 1]
        p.ind[i] = p.ind[i + 1]
    p.npartitions -= 1
    for i in range(p.npartitions):
        if p.ind[i] > iy:
            p.ind[i] -= 1
        elif p.ind[i] == iy:
            rjmcmc_error("position_map1d_delete: invalid entry")
    return position_map1d_valid(p)

# check
def position_map1d_move(p, x, new_x):
    if p is None:
        rjmcmc_error("position_map1d_t: null map")
    mi = -1
    for i in range(1, p.npartitions):
        if p.pos[i] == x:
            mi = i
            break
    if mi < 0:
        rjmcmc_error("position_map1d_move: failed to find old point")
    p.pos[mi] = new_x
    if new_x < x:
        while p.pos[mi] < p.pos[mi - 1]:
            tx = p.pos[mi - 1]
            ti = p.ind[mi - 1]
            p.pos[mi - 1] = p.pos[mi]
            p.ind[mi - 1] = p.ind[mi]
            p.pos[mi] = tx
            p.ind[mi] = ti
            mi -= 1
    else:
        while p.pos[mi] > p.pos[mi + 1]:
            tx = p.pos[mi + 1]
            ti = p.ind[mi + 1]
            p.pos[mi + 1] = p.pos[mi]
            p.ind[mi + 1] = p.ind[mi]
            p.pos[mi] = tx
            p.ind[mi] = ti
            mi += 1
    return 0

# check
def position_map1d_position_of_index(p, iy):
    if p is None:
        rjmcmc_error("position_map1d_position_of_index: null map")
    for i in range(p.npartitions):
        if p.ind[i] == iy:
            return p.pos[i]
    rjmcmc_error("position_map1d_position_of_index: failed to find interval")

# check
def position_map1d_next_position(p, x):
    if p is None:
        rjmcmc_error("position_map1d_next_position: null map")
    for i in range(1, p.npartitions):
        if p.pos[i] > x:
            return p.pos[i]
    rjmcmc_error("position_map1d_next_position: failed to find interval")

# check
def position_map1d_predecessor_of_point(p, x):
    if p is None:
        rjmcmc_error("position_map1d_predecessor_of_point: null map")
    if x >= p.pos[p.npartitions - 1]:
        return 1
    for i in range(p.npartitions - 1):
        if p.pos[i] <= x and p.pos[i + 1] > x:
            return p.ind[i]
    return -1

# check
def position_map1d_predecessor_of_index(p, iy):
    if p is None:
        rjmcmc_error("position_map1d_predecessor_of_index: null map")
    if iy == 0:
        rjmcmc_error("position_map1d_predecessor_of_index: invalid index")
    for i in range(1, p.npartitions):
        if p.ind[i] == iy:
            return p.ind[i - 1]
    return -1

# check
def position_map1d_traverse_intervals(p, interval_cb, user_arg):
    if p is None:
        rjmcmc_error("position_map1d_traverse_intervals: null map")
    if interval_cb is None:
        rjmcmc_error("position_map1d_traverse_intervals: null cb")
    for i in range(1, p.npartitions):
        if interval_cb(user_arg, p.pos[i - 1], p.pos[i], p.ind[i - 1], p.ind[i]) < 0:
            return -1
    return 0

# check
def position_map1d_valid(p):
    if p is None:
        rjmcmc_error("position_map1d_valid: null map")
    if p.ind[0] != 0:
        rjmcmc_error("position_map1d_valid: invalid first index")
    if p.ind[p.npartitions - 1] != 1:
        rjmcmc_error("position_map1d_valid: invalid last index")
    lastx = p.pos[0]
    for i in range(1, p.npartitions):
        if p.pos[i] < lastx:
            rjmcmc_error("position_map1d_valid: out of order")
        lastx = p.pos[i]
    return 0
