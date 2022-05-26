from email import iterators
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
import argparse

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

from src.models import AE
from src.utils import save_res, _path, _random_seed_gen, Data_Factory
from src.opt import pure_dkbo, DKBO_OLP_MenuStrategy, RandomOpt
from src.lamcts import MCTS



if __name__ == "__main__":
    # parse the cli
    cli_parser = argparse.ArgumentParser(description="configuration of mf test")
    cli_parser.add_argument("--name", nargs='?', default="test", type=str, help="name of experiment",)
    cli_parser.add_argument("--subdir", nargs='?', default="./res", type=str, help="directory to store the scripts and results",)
    cli_parser.add_argument("--datadir", nargs='?', default="./data", type=str, help="directory of the test datasets")
    cli_parser.add_argument("-v",  action="store_true", default=False, help="flag of if verbose")
    cli_parser.add_argument("-p",  action="store_true", default=False, help="flag of if plotting result")
    cli_parser.add_argument("-s",  action="store_true", default=False, help="flag of if storing result")
    cli_parser.add_argument("-n", action="store_true", default=False, help="flag of if negate the obj value to maximize")
    cli_parser.add_argument("--init_num", nargs='?', default=10, type=int, help="number of initial random points")
    cli_parser.add_argument("--run_times", nargs='?', default=5, type=int, help="run times of the tests")
    cli_parser.add_argument("--opt_horizon", nargs='?', default=40, type=int, help="horizon of the optimization")
    
    
    cli_args = cli_parser.parse_args()
    ### load dataset
    assert not (cli_args.datadir is None)
    if cli_args.datadir.endswith(".pt"):
        dataset = torch.load(cli_args.datadir).numpy()  # want to find the maximal
    elif cli_args.datadir.endswith(".npy"):
        dataset = np.load(cli_args.datadir)
    data_dim = dataset.shape[1]-1

    # original Objective
    ROBUST = True
    ScalerClass = RobustScaler if ROBUST else StandardScaler
    scaler = ScalerClass().fit(dataset[:,:-1])                   # obj foldx -> rosetta
    scaled_f_data = scaler.transform(dataset[:,:-1])
    scaled_input = scaled_f_data
    output = dataset[:,-1]
    shuffle_filter = np.random.choice(scaled_f_data.shape[0], scaled_f_data.shape[0], replace=False)
    train_output = -1 * output.reshape([-1,1]) if cli_args.n else output.reshape([-1,1])

    # LAMCTS experiment
    verbose = cli_args.v

    approximate = lambda x: Data_Factory.nearest(dataset, x)

    def approx_obj_func(multi_rows):
        row_num = multi_rows.shape[0]
        # print(multi_rows.shape)
        assert row_num > 1
        res = np.array([approximate(multi_rows[i])[0][-1] for i in range(row_num)])
        return -res

    def obj_func_simple(config):
        _, index = Data_Factory.nearest(dataset, config)
        return dataset[index, -1]

    def obj_func_simple_neg(config):
        _, index = Data_Factory.nearest(dataset, config)
        return -dataset[index, -1]

    lb = scaled_input.min(axis=0) - 1e-10
    ub = dataset[:,:-1].max(axis=0)
    optima = dataset[:,-1].max()
    Cp = 10
    leaf_size = 8
    kernel_type = 'rbf'
    gamma_type = "auto"
    solve_type = 'bo'   # turbo not available
    obj_func_simple.lb = lb
    obj_func_simple.ub = ub

    agent = MCTS(
             lb = lb,              # the lower bound of each problem dimensions
             ub = ub,              # the upper bound of each problem dimensions
             dims = data_dim,          # the problem dimensions
             ninits = cli_args.init_num,      # the number of random samples used in initializations 
             func = obj_func_simple,          # function object to be optimized
             Cp = Cp,              # Cp for MCTS
             leaf_size = leaf_size, # tree leaf size
             kernel_type = kernel_type, #SVM configruation
             gamma_type = gamma_type,   #SVM configruation
             solver_type= solve_type, # either "bo" or "turbo"
             verbose = cli_args.v # if printing verbose messages
             )

    agent.search(iterations = cli_args.opt_horizon)

    values = [sample[1] for sample in agent.sampels]
    regret = np.minimum.accumulate(optima - np.array(values))

    if cli_args.s:
        # opt.store_results(cli_args.subdir)
        pass
    if cli_args.p:
        # opt.plot_results(cli_args.subdir)
        pass



