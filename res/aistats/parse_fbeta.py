import os
import numpy as np
import re
from copy import deepcopy

from matplotlib import pyplot as plt

n_init=10
RES_num = {}
res_collect = {}
# acqs = ["ci", "ts"]
acqs = ['ci']

# load ballet & dkbo
for acq in acqs:
    dir = f"./{acq}"
    for filename in os.listdir(dir):
        _filename = deepcopy(filename)
        if not filename.endswith("npy"):
            continue

        # if "FB" not in _filename:
        # if "FB0.2" not in _filename and "FB80.0" not in _filename :
        #     continue

        if "TI100-" not in _filename:
            continue

        if "R10" not in _filename:
            continue

        if "interval" in _filename:
            continue

        inter = "sec" in _filename
        linear = '-hd-' in _filename
        
        _attr = list(filename.split("-"))
        _name   = _attr[1]

        res = np.load(f"{dir}/{_filename}")

        n_iter = res.shape[1]
        n_repeat = res.shape[0]
        CI = 0.5
        sqrt_n = np.sqrt(int(n_repeat))
        coef = CI /sqrt_n
        _mean = res[:,:].mean(axis=0)
        _sterr = res[:,:].std(axis=0) * coef
        res_collect = {"_mean": _mean, "_sterr": _sterr, "n_repeat": n_repeat}
        # print(f"{_name}-{turbo_k}")

        method = f"{_name}-{'i' if inter else 'r'}{acq}-{'Lin' if linear else 'RBF'}-{_attr[3] if not linear else _attr[4]}"
        # print(method)
        RES_num[method] = RES_num.get(f"{method}", [])
        RES_num[method].append(deepcopy(res_collect))


    fig = plt.figure(figsize=[10,10])
    ax = plt.subplot(111)

    for record in sorted(RES_num.keys()): 
        # _record = list(record.split("-"))
        # _name, turbo_k = _record[0], _record[1]

        for res_collect in RES_num[record]:
            # if record.startswith("eg1d"):
            res_collect['_mean'], res_collect['_sterr'] = res_collect['_mean'][n_init:], res_collect['_sterr'][n_init:]
            if record.startswith("nano") and 'Lin' in record and "ici" in record:
                print(f"{record:<40}\t \t\t{res_collect['_mean'][-1]:.2f}\pm {res_collect['_sterr'][-1]:.2f}")
                
                ax.plot(res_collect['_mean'], label=r"$\beta^{\frac{1}{2}}_{T}$="+record[-3:])
                ax.fill_between(np.arange(res_collect['_mean'].shape[0]), res_collect['_mean'] - res_collect['_sterr'], 
                                res_collect['_mean'] + res_collect['_sterr'], alpha=0.3)
            # ax.plot(RES_num[method][:,:n_iter].mean(axis=0), label=method)
            # ax.fill_between(np.arange(n_iter), RES_num[method][:,:n_iter].mean(axis=0) - RES_num[method][:,:n_iter].std(axis=0) * coef, 
                            # RES_num[method][:,:n_iter].mean(axis=0) + RES_num[method][:,:n_iter].std(axis=0) * coef, alpha=0.3)

    # plt.title("Simple Regret Comparison")
    # plt.legend()
    fontsize = 16
    ax.set_xlabel("Iteration", fontsize=fontsize)
    ax.set_ylabel("Simple Regret",  fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=int(np.ceil(len(labels))//2), prop={'size': fontsize})
    plt.tight_layout()
    plt.savefig(f"fbeta.png")
    plt.savefig(f"fbeta.pdf")


    plt.close()
