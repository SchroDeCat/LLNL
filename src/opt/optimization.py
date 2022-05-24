'''
Full pipeline for DKBO
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
from .dkbo_ae import DK_BO_AE
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
# STUDY_PARTITION = True
STUDY_PARTITION = False

def dkl_opt_test(x_tensor, y_tensor, name, n_repeat=2, lr=1e-2, n_init=10, n_iter=40, train_iter=100, return_result=True, fix_seed=True,
                    pretrained=False, ae_loc=None, plot_result=False, save_result=False, save_path=None, acq="ts", verbose=True, study_partition=STUDY_PARTITION):
    '''
    Check DKL opt behavior on full dataset
    '''
    max_val = y_tensor.max()
    reg_record = np.zeros([n_repeat, n_iter])

    if pretrained:
        assert not (ae_loc is None)
        ae = AE(x_tensor, lr=1e-3)
        ae.load_state_dict(torch.load(ae_loc, map_location=DEVICE))
    else:
        ae = None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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
        for rep in range(n_repeat):
            # no AE pretrain 
            if verbose:
                print("DKBO")
            sim_dkl = DKL(x_tensor, y_tensor.squeeze(), lr=lr, n_iter=train_iter, low_dim=True,  pretrained_nn=ae)
            sim_dkl.train_model(loss_type="nll", verbose=True)
            

    if plot_result:
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = sim_dkl.likelihood(sim_dkl.model(x_tensor))
            _, ucb = observed_pred.confidence_region()
        _path = f"{save_path}/DKL-R{n_repeat}-whole_L{int(-np.log10(lr))}-TI{train_iter}"
        # _file_path = _path(save_path=save_path, name=name, init_strategy=init_strategy, n_repeat=n_repeat, n_iter=n_iter, num_GP=num_GP, cluster_interval=cluster_interval,  acq=acq, train_iter=train_times)
        fig = plt.figure()
        plt.scatter(y_tensor, ucb, s=2)
        plt.scatter(y_tensor, ucb, color='r', marker="*", s=5)
        plt.xlabel("Label")
        plt.ylabel("UCB")
        plt.colorbar()
        plt.title(f"{name}")
        plt.savefig(f"{_path}.png")
        print(f"Fig stored to {_path}")

    if save_result:
        pass

    if return_result:
        pass


def pure_dkbo(x_tensor, y_tensor, name, n_repeat=2, lr=1e-2, n_init=10, n_iter=40, train_iter=100, return_result=True, fix_seed=True,
                    pretrained=False, ae_loc=None, plot_result=False, save_result=False, save_path=None, acq="ts", verbose=True, study_partition=STUDY_PARTITION):
    max_val = y_tensor.max()
    reg_record = np.zeros([n_repeat, n_iter])

    if pretrained:
        assert not (ae_loc is None)
        ae = AE(x_tensor, lr=1e-3)
        ae.load_state_dict(torch.load(ae_loc, map_location=DEVICE))
    else:
        ae = None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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
        for rep in range(n_repeat):
            # no AE pretrain 
            if verbose:
                print("DKBO")
            sim_dkbo = DK_BO_AE(x_tensor, y_tensor, lr=lr,
                                n_init=n_init,  train_iter=train_iter, regularize=False, dynamic_weight=False, 
                                max=max_val, pretrained_nn=ae, verbose=verbose)
            sim_dkbo.query(n_iter=n_iter, acq=acq, study_ucb=STUDY_PARTITION, study_interval=10, study_res_path=save_path)
            reg_record[rep, :] = sim_dkbo.regret

    reg_output_record = reg_record.mean(axis=0)
    if plot_result:
        plt.plot(reg_output_record.squeeze(), label="dkbo")
        plt.legend()
        plt.xlabel("Iteration")
        plt.ylabel("Regret")
        plt.title("DKBO performance")
        plt.show()

    if save_result:
        assert not (save_path is None)
        save_res(save_path=save_path, name=name, res=reg_record, n_repeat=n_repeat, num_GP=1, n_iter=n_init, train_iter=train_iter,
                init_strategy='none', cluster_interval=1, acq=acq, lr=lr, verbose=verbose,)

    if return_result:
        return reg_record


def ol_filter_dkbo(x_tensor, y_tensor, n_init=10, n_repeat=2, train_times=10,
                   n_iter=40, filter_interval=1, acq="ts", verbose=True, lr=1e-2, name="test", return_result=True, 
                   plot_result=False, save_result=False, save_path=None, fix_seed=False,  pretrained=False, ae_loc=None, study_partition=STUDY_PARTITION):
    # print(ucb_strategy)
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
        for rep in tqdm.tqdm(range(n_repeat), desc=f"Experiment Rep"):
            # set seed
            if fix_seed:
                # print(n_init+n_repeat*n_iter)
                _seed = rep * n_iter + n_init
                # _seed = 70
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
            _dkl.train_model()
            _dkl.model.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                _observed_pred = _dkl.likelihood(_dkl.model(x_tensor.to(DEVICE)))
                lcb, ucb = _observed_pred.confidence_region()
            # print(f"pretrained {pretrained} ae {type(ae)}  n_iter {train_times}")
            

            # each test instance
            for iter in range(0, n_iter, filter_interval):
                ucb_filter = ucb >= lcb[observed==1].max() # filtering
                observed_unfiltered = np.min([observed, ucb_filter.numpy()], axis=0)      # observed and not filtered outs
                # observed_unfiltered[:n_init]=0
                # print(f"observed unfiltered {observed_unfiltered.sum()}/data_size {data_size} unfiltered {ucb_filter.sum()} observed {observed.sum()}")
                # print(f"beginning init_y range {init_y.min(), init_y.max()}")
                init_x = torch.cat([init_x, x_tensor[observed_unfiltered==1]])
                init_y = torch.cat([init_y, y_tensor[observed_unfiltered==1]])
                # print(f"init_y range {init_y.min(), init_y.max()}")
                # print(f"observed unfiltered range {y_tensor[observed_unfiltered==1].min(), y_tensor[observed_unfiltered==1].max()}")
                # print(f"unfiltered range {y_tensor[ucb_filter==1].min(), y_tensor[ucb_filter==1].max()} / truemax {max_val}")
                # print(f"unfiltered range -2 {y_tensor[util_array[ucb_filter==1]].min(), y_tensor[util_array[ucb_filter==1]].max()} / truemax {max_val}")
                # print(f"observed range {y_tensor[observed==1].min(), y_tensor[observed==1].max()}")
                # print(f"verify ucb {ucb[ucb_filter==1].min(), ucb[ucb_filter==1].max()} > lcb {lcb[observed==1].max()}")
                
                sim_dkbo = DK_BO_AE(x_tensor[ucb_filter], y_tensor[ucb_filter], lr=lr,
                                    n_init=n_init,  train_iter=train_times, regularize=False, dynamic_weight=False, 
                                    max=max_val, pretrained_nn=ae, verbose=verbose, init_x=init_x, init_y=init_y)
                query_num = min(filter_interval, n_iter-iter)
                sim_dkbo.query(n_iter=query_num, acq=acq, study_ucb=STUDY_PARTITION, study_interval=10, study_res_path=save_path)
                # update records
                _step_size = min(iter+filter_interval, n_iter)
                reg_record[rep, iter:_step_size] = sim_dkbo.regret[-_step_size:]
                ucb_filtered_idx = util_array[ucb_filter]
                # print(sim_dkbo.observed.sum(), np.sum(sim_dkbo.observed==1))
                # print(ucb_filtered_idx[sim_dkbo.observed== 1].shape)
                observed[ucb_filtered_idx[sim_dkbo.observed==1]] = 1
                # print(f"{sim_dkbo.observed.shape} {observed.shape} {ucb_filtered_idx.shape} {y_tensor.size()}")
                # print(f"post-opt observed range {y_tensor[observed].min(), y_tensor[observed].max()}")
                # print(f"post-opt observed range {y_tensor[observed==1].min(), y_tensor[observed==1].max()}")
                # print(f"post-opt observed range {y_tensor[ucb_filtered_idx][sim_dkbo.observed].min(), y_tensor[ucb_filtered_idx][sim_dkbo.observed].max()}")
                # print(f"post-opt observed range {y_tensor[ucb_filtered_idx][sim_dkbo.observed==1].min(), y_tensor[ucb_filtered_idx][sim_dkbo.observed==1].max()}")
                # print(f"post-opt observed range {y_tensor[ucb_filtered_idx[sim_dkbo.observed==1]].min(), y_tensor[ucb_filtered_idx[sim_dkbo.observed==1]].max()}")
                # print(f"post-opt observed range {y_tensor[ucb_filter][sim_dkbo.observed].min(), y_tensor[ucb_filter][sim_dkbo.observed].max()}")
                # _debug = y_tensor[ucb_filtered_idx] - y_tensor[ucb_filter]
                # print(f"debug {_debug.sum()} {_debug} {np.sum(sim_dkbo.observed==1 - sim_dkbo.observed)}")
                # update ucb for filtering
                _dkl = DKL(x_tensor[ucb_filter], y_tensor[ucb_filter].squeeze(), n_iter=10, low_dim=True)
                _dkl.train_model()
                _dkl.model.eval()
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    _observed_pred = _dkl.likelihood(_dkl.model(x_tensor))
                    lcb, ucb = _observed_pred.confidence_region()

                if study_partition:
                    _path = f"{save_path}/Filter-{name}-{acq}-R{n_repeat}-P{1}-T{n_iter}_I{filter_interval}_L{int(-np.log10(lr))}-TI{train_times}"
                    # _file_path = _path(save_path=save_path, name=name, init_strategy=init_strategy, n_repeat=n_repeat, n_iter=n_iter, num_GP=num_GP, cluster_interval=cluster_interval,  acq=acq, train_iter=train_times)
                    fig = plt.figure()
                    plt.scatter(y_tensor, ucb, c=ucb_filter, s=2)
                    plt.scatter(y_tensor[observed==1], ucb[observed==1], color='r', marker="*", s=5)
                    plt.xlabel("Label")
                    plt.ylabel("UCB")
                    plt.colorbar()
                    plt.title(f"{name} Iter {iter}")
                    plt.savefig(f"{_path}-Iter{iter}.png")
                    print(f"Fig stored to {_path}")

    for rep in range(n_repeat):
        reg_record[rep] = np.minimum.accumulate(reg_record[rep])
    reg_output_record = reg_record.mean(axis=0)
    
    if plot_result:
        plt.plot(reg_output_record)
        plt.ylabel("regret")
        plt.xlabel("Iteration")
        plt.title(f'simple regret for {name}')
        _path = f"{save_path}/Filter-{name}-{acq}-R{n_repeat}-P{1}-T{n_iter}_I{filter_interval}_L{int(-np.log10(lr))}-TI{train_times}"
        plt.savefig(f"{_path}.png")
        # plt.show()

    if save_result:
        assert not (save_path is None)
        save_res(save_path=save_path, name=name, res=reg_record, n_repeat=n_repeat, num_GP=2, n_iter=n_iter, train_iter=train_times,
                init_strategy='none', cluster_interval=filter_interval, acq=acq, lr=lr, ucb_strategy="exact", verbose=verbose,)

    if return_result:
        return reg_record

def ol_partition_dkbo(x_tensor, y_tensor, init_strategy:str="kmeans", n_init=10, n_repeat=2, num_GP=2, train_times=10,
                    n_iter=40, cluster_interval=1, acq="ts", verbose=True, lr=1e-2, name="test", return_result=True, ucb_strategy="exact",
                    plot_result=False, save_result=False, save_path=None, fix_seed=False,  pretrained=False, ae_loc=None, study_partition=STUDY_PARTITION):
    # print(ucb_strategy)
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
                _seed = rep * n_iter + n_init
                # _seed = 70
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
            _dkl.train_model()
            _dkl.model.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                _observed_pred = _dkl.likelihood(_dkl.model(x_tensor.to(DEVICE)))
                _, ucb = _observed_pred.confidence_region()
            # print(f"pretrained {pretrained} ae {type(ae)}  n_iter {train_times}")
            cluster_id = clustering_methods(x_tensor, ucb.to('cpu'), num_GP, init_strategy, dkl=True, pretrained_nn=ae, train_iter=train_times)
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
                ucb = dkbo_olp.ucb_func(x_tensor=x_tensor, method=ucb_strategy, cluster_id=cluster_id)
                for loc in dkbo_olp.observed:
                    cluster_filter = cluster_id==loc[0]
                    cluster_filter[:n_init] = True  # align with previous workaround
                    if sum(cluster_filter) > 0:
                        observed[util_array[cluster_filter][loc[1]]] = 1
                
                cluster_id = clustering_methods(x_tensor, ucb, num_GP, init_strategy, dkl=True, pretrained_nn=ae, train_iter=train_times)

                # update observed clustering
                # init_x, init_y = torch.cat(dkbo_olp.init_x_list), torch.cat(dkbo_olp.init_y_list)
                init_x, init_y = x_tensor[observed==1], y_tensor[observed==1]
                # print(f"max n_init init_y {init_y[:n_init].max()} reg {obj - init_y[:n_init].max()}")
                cluster_id_init = cluster_id[observed==1]

                if study_partition:
                    _path = f"{save_path}/OL-{name}-{init_strategy}-{acq}-R{n_repeat}-P{num_GP}-T{n_iter}_I{cluster_interval}_L{int(-np.log10(lr))}-TI{train_times}"
                    # _file_path = _path(save_path=save_path, name=name, init_strategy=init_strategy, n_repeat=n_repeat, n_iter=n_iter, num_GP=num_GP, cluster_interval=cluster_interval,  acq=acq, train_iter=train_times)
                    fig = plt.figure()
                    plt.scatter(y_tensor, ucb, c=cluster_id, s=2)
                    plt.scatter(y_tensor[observed==1], ucb[observed==1], color='r', marker="*", s=5)
                    plt.xlabel("Label")
                    plt.ylabel("UCB")
                    plt.colorbar()
                    plt.title(f"{name} Iter {iter}")
                    plt.savefig(f"{_path}-Iter{iter}.png")
                    print(f"Fig stored to {_path}")

    for rep in range(n_repeat):
        reg_record[rep] = np.minimum.accumulate(reg_record[rep])
    reg_output_record = reg_record.mean(axis=0)
    
    if plot_result:
        plt.plot(reg_output_record)
        plt.ylabel("regret")
        plt.xlabel("Iteration")
        plt.title(f'simple regret for {name}')
        # plt.show()

    if save_result:
        assert not (save_path is None)
        save_res(save_path=save_path, name=name, res=reg_record, n_repeat=n_repeat, num_GP=num_GP, n_iter=n_iter, train_iter=train_times,
                init_strategy=init_strategy, cluster_interval=cluster_interval, acq=acq, lr=lr, ucb_strategy=ucb_strategy, verbose=verbose,)

    if return_result:
        return reg_record