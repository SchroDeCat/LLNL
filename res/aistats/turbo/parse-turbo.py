import os
import numpy as np
import re
import pandas as pd
from copy import deepcopy

from matplotlib import pyplot as plt

n_init=10
RES_num = {}
res_collect = {}

for filename in os.listdir("./"):
    _filename = deepcopy(filename)
    if not filename.endswith("npy"):
        continue

    # if "R2" not in _filename:
        # continue
    linear = '-hd-' in _filename
    _attr = list(filename.split("-"))
    _name   = _attr[2] if linear else _attr[1]

    res = np.load(_filename)

    n_iter = res.shape[1]
    n_repeat = res.shape[0]
    CI = 1
    sqrt_n = np.sqrt(int(n_repeat))
    coef = CI /sqrt_n
    _mean = res[:,-1].mean()
    _sterr = res[:,-1].std() * coef
    res_collect = { "_mean": _mean, "_sterr": _sterr}
    # print(f"{_name}-{turbo_k}")

    RES_num[_name] = RES_num.get(f"{_name}", [])
    RES_num[_name].append(deepcopy(res_collect))



df_dict = {}
for record in RES_num.keys(): 
    # _record = list(record.split("-"))
    # _name, turbo_k = _record[0], _record[1]
    _name = record
    
    for res_collect in RES_num[record]:
            print(f"{_name:<20}\t TuRBO-1: \t\t${res_collect['_mean']:.2f}\pm {res_collect['_sterr']:.2f}$")


# df = pd.DataFrame(RES_num)

