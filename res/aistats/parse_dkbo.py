import os
import numpy as np
import re
from copy import deepcopy

from matplotlib import pyplot as plt

n_init=10
RES_num = {}
res_collect = {}
acqs = ["ci", "ts","ucb"]

# load ballet & dkbo
for acq in acqs:
    dir = f"./{acq}"
    for filename in os.listdir(dir):
        _filename = deepcopy(filename)
        if not filename.endswith("npy") or not filename.startswith("Pure-DK"):
            continue

        # if "FB" not in _filename:

        if "R10" not in _filename and "R30" not in _filename:
            continue

        linear = '_hd-' in _filename
        
        _attr = list(filename.split("-"))
        _name   = _attr[2]

        res = np.load(f"{dir}/{_filename}")

        n_iter = res.shape[1]
        n_repeat = res.shape[0]
        CI = 1
        sqrt_n = np.sqrt(int(n_repeat))
        coef = CI /sqrt_n
        _mean = res[:,-1].mean()
        _sterr = res[:,-1].std() * coef
        res_collect = {"_mean": _mean, "_sterr": _sterr}
        # print(f"{_name}-{turbo_k}")
        # print(_filename)

        method = f"{_name}-{acq}-{'Lin' if linear else 'RBF'}"
        # print(method)
        RES_num[method] = RES_num.get(f"{method}", [])
        RES_num[method].append(deepcopy(res_collect))


    for record in RES_num.keys(): 
        # _record = list(record.split("-"))
        # _name, turbo_k = _record[0], _record[1]
        
        for res_collect in RES_num[record]:
            # if record.startswith("eg1d"):
            if record.startswith("hdbo"):
                print(f"{record:<40}\t \t\t${res_collect['_mean']:.2f}\pm {res_collect['_sterr']:.2f}$")


