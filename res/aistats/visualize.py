import os
import numpy as np
import re
from copy import deepcopy

from matplotlib import pyplot as plt

n_init=10
RES_num = {}
names = ['eg1d', "water_converter", "rosetta", "nano", "hdbo", "gb1"]
lamcts_names = {'OneDeg':'eg1d', "water_converter":"water_converter", "oct_x_Rosetta":"rosetta", "nano":"nano", "hdbo200":"hdbo", "gb1_embed": "gb1"}
# res_collect = {name:deepcopy(RES_num) for name in names}
res_collect = {}
# acqs = ["ci", 'ucb', 'ts']
# acqs = ["ci", "ts"]
acqs = ["ci"]

# load ballet & dkbo
for acq in acqs:
    dir = f"./{acq}"
    for filename in os.listdir(dir):
        _filename = deepcopy(filename)
        if not filename.endswith("npy"):
            continue

        dkbo = filename.startswith("Pure-DK")
        if dkbo:
            filename = filename[5:]
            # print("dkbo")

        
        _attr = list(filename.split("-"))
        if dkbo:
            pass
            _name = _attr[1][:-2]
            if not _name in names:
                continue
            hd          = "hd"  in _attr
            n_repeat    = int(_attr[3][1:])
            exp_name = f"dkbo{'-hd' if hd else ''}-{acq}-{_attr[1][-2:]}-{_attr[3]}-{_attr[4]}"

        else:
            _nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", filename)
            if not len(_nums)>= 7:
                continue
            _name = _attr[1]
            if not _name in names:
                continue
            # print(filename)
            if "interval" in _attr:
                continue

            # print(_attr, len(_attr), _attr[6])
            intersec = 'sec.npy' in _attr
            hd       = "hd"  in _attr
            if hd:
                _attr.remove("hd")
            # print(_attr, len(_attr), _attr[6])
            n_repeat   = int(_attr[6][1:])
            
            exp_name = f"ballet{'-hd' if hd else ''}-{acq}-{_attr[2]}-{_attr[3]}-{_attr[6]}-{_attr[7]}-{_attr[8]}-{_attr[9]}{'-sec' if intersec else ''}"
        
        print(f"{_name}-{n_repeat}-{exp_name}")
        if not f"{_name}-{n_repeat}" in res_collect.keys():
            res_collect[f"{_name}-{n_repeat}"] = deepcopy(RES_num)
        res_collect[f"{_name}-{n_repeat}"][exp_name] = np.load(f"{dir}/{_filename}")
        # res_collect[f"{_name}-{n_repeat}"][exp_name] = np.load(f"{dir}/{_filename}")

# init_regret = RES_num[f"DKBO-AE-{n_iter}"][:,0].mean(axis=0)

# load LAMCTS

dir = f"./lamcts"
for filename in os.listdir(dir):
    _filename = deepcopy(filename)
    if not filename.endswith("npy"):
        continue

    _attr = list(filename.split("-"))
    hd       = "hd"  in _attr
    if hd:
        _attr.remove("hd")

    _name = lamcts_names[_attr[2]]
    n_repeat    = int(_attr[3][1:])

    
    exp_name = f"lamcts{'-hd' if hd else ''}-{_attr[5]}"
    print(f"{_name}-{n_repeat}-{exp_name}")
    if not f"{_name}-{n_repeat}" in res_collect.keys():
        res_collect[f"{_name}-{n_repeat}"] = deepcopy(RES_num)
    res_collect[f"{_name}-{n_repeat}"][exp_name] = np.load(f"{dir}/{_filename}")[:,n_init:]


for key, RES_num in res_collect.items():
    _name, n_repeat = tuple(key.split("-"))

    fig = plt.figure(figsize=[20,10])
    ax = plt.subplot(1,1,1)
    for method in RES_num.keys():
        CI = 1
        sqrt_n = np.sqrt(int(n_repeat))
        coef = CI /sqrt_n
        # RES_num[method][:,0] = init_regret
        RES_num[method] = np.minimum.accumulate(RES_num[method], axis=1)
        ax.plot(RES_num[method].mean(axis=0), label=method)
        # ax.plot(RES_num[method][:,:n_iter].mean(axis=0), label=method)
        # ax.fill_between(np.arange(n_iter), RES_num[method][:,:n_iter].mean(axis=0) - RES_num[method][:,:n_iter].std(axis=0) * coef, 
                        # RES_num[method][:,:n_iter].mean(axis=0) + RES_num[method][:,:n_iter].std(axis=0) * coef, alpha=0.3)

    # plt.title("Simple Regret Comparison")
    # plt.legend()
    ax.set_xlabel("Iteration", fontsize=20)
    ax.set_ylabel("Simple Regret")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=int(np.ceil(len(labels)/5)))
    plt.savefig(f"{key}.png")
    plt.close()