'''
Full pipeline for DKBO-OLP
'''

import gpytorch
import os
import random
import torch
import tqdm
import time
import matplotlib
import math
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime
import itertools

from ..models import DKL, AE
from ..utils import save_res, load_res, clustering_methods
from .dkbo_olp import DK_BO_OLP
from sparsemax import Sparsemax
from scipy.stats import ttest_ind
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ol_partition_dkbo(x_tensor, y_tensor, init_strategy:str="kmeans", n_init=10, n_repeat=2, num_GP=2, train_times=10,
                    n_iter=40, cluster_interval=1, acq="ts", verbose=True, lr=1e-2, name="test", return_result=True,
                    plot_result=False, save_result=False, save_path=None, fix_seed=False,  pretrained=False, ae_loc=None,):
    max_val = y_tensor.max()
    reg_record = np.zeros([n_repeat, n_iter])
     # init dkl and generate ucb for partition
    data_size = x_tensor.size(0)
    util_array = np.arange(data_size)

    if pretrained:
        assert not (ae_loc is None)
        ae = AE(x_tensor, lr=1e-3)
        ae.load_state_dict(torch.load(ae_loc, map_location=DEVICE))
    else:
        ae = None


    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for rep in tqdm.tqdm(range(n_repeat), desc=f"{init_strategy}"):
            # set seed
            if fix_seed:
                # print(n_init+n_repeat*n_iter)
                # _seed = rep*n_iter + n_init
                _seed = 70
                torch.manual_seed(_seed)
                np.random.seed(_seed)
                random.seed(_seed)
                torch.cuda.manual_seed(_seed)
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True


            # init in each round
            observed = np.zeros(data_size)
            observed[:n_init] = 1
            init_x = x_tensor[:n_init]
            init_y = y_tensor[:n_init]
            _dkl = DKL(init_x, init_y.squeeze(), n_iter=10, low_dim=True)
            _dkl.model.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                _observed_pred = _dkl.likelihood(_dkl.model(x_tensor.to(DEVICE)))
                _, ucb = _observed_pred.confidence_region()

            cluster_id = clustering_methods(x_tensor, ucb.to('cpu'), num_GP, init_strategy)
            cluster_id_init = cluster_id[observed==1]

            # each test instance
            for iter in range(0, n_iter, cluster_interval):
                scaled_input_list, scaled_output_list = [], []
                init_x_list, init_y_list = [], []
                for idx in range(num_GP):
                    cluster_filter = cluster_id == idx
                    cluster_filter[:n_init] = True
                    # print(f"cluster {idx} size {sum(cluster_filter)}")
                    scaled_input_list.append(x_tensor[cluster_filter])
                    scaled_output_list.append(y_tensor[cluster_filter])

                    cluster_init_filter = cluster_id_init == idx
                    cluster_init_filter[:n_init] = True
                    # print(f"cluster init {idx} size {sum(cluster_init_filter)} {cluster_id_init}")
                    # print(f"init_x size {init_x.size()}, observed num {sum(observed)}")
                    init_x_list.append(init_x[cluster_init_filter])
                    init_y_list.append(init_y[cluster_init_filter].reshape([-1,1]))
                    # print(f"sizes {init_x_list[0].size()}, {init_y_list[0].size()} num_GP {num_GP}")

                dkbo_olp = DK_BO_OLP(init_x_list, init_y_list, scaled_input_list, scaled_output_list, lr=lr, train_iter=train_times,
                                n_init=n_init, regularize=False, dynamic_weight=False, max_val=max_val, num_GP=num_GP, pretrained_nn=ae)
                # print("dkbo y list", torch.cat(dkbo_olp.init_y_list).max(), "dkbo Y list shape", torch.cat(dkbo_olp.init_y_list).size())

                # record regret
                query_num = min(cluster_interval, n_iter-iter)
                # print(f"query num {query_num}")
                dkbo_olp.query(n_iter=query_num, acq=acq, verbose=verbose)
                # print(dkbo_olp.regret.shape)
                # print("dkbo Reg", dkbo_olp.regret)
                reg_record[rep, iter:min(iter+cluster_interval, n_iter)] = dkbo_olp.regret
                
                # update ucb for clustering
                ucb = dkbo_olp.ucb_func(x_tensor=x_tensor)
                for loc in dkbo_olp.observed:
                    cluster_filter = cluster_id==loc[0]
                    cluster_filter[:n_init] = True  # align with previous workaround
                    if sum(cluster_filter) > 0:
                        observed[util_array[cluster_filter][loc[1]]] = 1
                cluster_id = clustering_methods(x_tensor, ucb, num_GP, init_strategy)

                # update observed clustering
                # init_x, init_y = torch.cat(dkbo_olp.init_x_list), torch.cat(dkbo_olp.init_y_list)
                init_x, init_y = x_tensor[observed==1], y_tensor[observed==1]
                # print(f"max n_init init_y {init_y[:n_init].max()} reg {obj - init_y[:n_init].max()}")
                cluster_id_init = cluster_id[observed==1]

    for rep in range(n_repeat):
        reg_record[rep] = np.minimum.accumulate(reg_record[rep])
    reg_output_record = reg_record.mean(axis=0)
    
    if plot_result:
        plt.plot(reg_output_record)
        plt.ylabel("regret")
        plt.xlabel("Iteration")
        plt.title(f'simple regret for {name}')
        plt.show()

    if save_result:
        assert not (save_path is None)
        save_res(save_path=save_path, name=name, res=reg_record, n_repeat=n_repeat, num_GP=num_GP, n_iter=n_init, train_iter=train_times,
                init_strategy=init_strategy, cluster_interval=cluster_interval, acq=acq, lr=lr, verbose=verbose,)

    if return_result:
        return reg_record