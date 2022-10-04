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
from src.opt import ol_partition_dkbo, pure_dkbo, ol_filter_dkbo
# from src.utils import 

acq = ['ts', 'ci', 'ucb']

# parse the cli
class Configuration():
    def __init__(self, name:str, ae_dir:str, data_dir:str, run_times:int=1, horizon:int=10, acq:str='ci', fbeta=0.2,
        intersection=False, ballet=True) -> None:
        self.name   = name
        self.aedir  = f"./tmp/{ae_dir}"
        self.subdir = f"./res/aistats/{acq}"  
        self.datadir=f"./data/{data_dir}" 
        self.run_times  =run_times
        self.acq_func =acq
        self.learning_rate=4  
        self.intersection=intersection  
        self.a=False
        self.r=False
        self.n=False
        self.high_dim=False
        self.init_num=10
        self.o=ballet   # BALLET or not
        self.p=True     # plot
        self.s=True     # store results
        self.f=True     # fix seed
        self.return_model=False 
        self.v=False
        self.beta=0     # true beta
        self.filter_interval=horizon//10
        self.opt_horizon=horizon
        self.train_times=10
        self.learning_rate=4
        self.fbeta = fbeta


def process(config):
    ### load dataset
    assert not (config.datadir is None)
    if config.datadir.endswith(".pt"):
        dataset = torch.load(config.datadir)  # want to find the maximal
    elif config.datadir.endswith(".npy"):
        dataset = torch.from_numpy(np.load(config.datadir))
    data_dim = dataset.shape[1]-1
    # dataset = torch.load(config.datadir)  # want to find the maximal
    # data_dim = dataset.shape[1]-1

    # original Objective
    ROBUST = True
    ScalerClass = RobustScaler if ROBUST else StandardScaler
    scaler = ScalerClass().fit(dataset[:,:-1])                   # obj foldx -> rosetta
    scaled_f_data = scaler.transform(dataset[:,:-1])
    scaled_input_tensor = torch.from_numpy(scaled_f_data).float()
    output_tensor = dataset[:,-1].float()
    shuffle_filter = np.random.choice(scaled_f_data.shape[0], scaled_f_data.shape[0], replace=False)
    train_output = -1 * output_tensor.reshape([-1,1]).float() if config.n else output_tensor.reshape([-1,1]).float()

    # dkbo experiment
    verbose = config.v
    fix_seed = config.f

    low_dim = not config.high_dim
    pretrained = not (config.aedir is None)

    # pretrain AE
    if not (config.aedir is None) and config.a:
        ae = AE(scaled_input_tensor, lr=1e-3)
        # print(scaled_input_tensor.shape)
        ae.train_ae(epochs=10, batch_size=200, verbose=True)
        torch.save(ae.state_dict(), config.aedir)
        if config.v:
            print(f"pretrained ae stored in {config.aedir}")
 
    lr_rank = -config.learning_rate
    learning_rate = 10 ** lr_rank

    print(f"Learning rate {learning_rate} Filtering {config.o} fix_seed {fix_seed} beta {config.beta} Regularize {config.r} Low dim {low_dim} CI intersection {config.intersection} verbose={verbose}")
    if config.o:
        res = ol_filter_dkbo(x_tensor=scaled_input_tensor, y_tensor=train_output, n_init=config.init_num, n_repeat=config.run_times, low_dim=low_dim, beta=config.beta, regularize=config.r,   ci_intersection=config.intersection,
                        n_iter=config.opt_horizon, filter_interval=config.filter_interval, acq=config.acq_func, verbose=verbose, lr=learning_rate, name=config.name, train_times=config.train_times, filter_beta=config.fbeta,
                        plot_result=config.p, save_result=config.s, save_path=config.subdir, return_result=not config.return_model, fix_seed=fix_seed,  pretrained=pretrained, ae_loc=config.aedir)
    else:
        pure_dkbo(x_tensor=scaled_input_tensor, y_tensor=train_output,  n_init=config.init_num, n_repeat=config.run_times, low_dim=low_dim, beta=config.beta,
                        n_iter=config.opt_horizon, acq=config.acq_func, verbose=verbose, lr=learning_rate, name=config.name, train_iter=config.train_times,
                        plot_result=config.p, save_result=config.s, save_path=config.subdir, return_result=True, fix_seed=fix_seed,  pretrained=pretrained, ae_loc=config.aedir,)



if __name__ == "__main__":
    run_times = 10
    exps = [{"name": "eg1d",            "ae_dir": "1deg_ae",            "data_dir":"opt_eg1d.npy",              "fbeta":0.2, "horizon":50,},
            {"name": "nano",            "ae_dir": "nano_mf_ae",         "data_dir":"data_nano_mf.pt",           "fbeta":0.2, "horizon":100,},
            {"name": "hdbo",            "ae_dir": "200d_ae",            "data_dir":"HDBO200.npy",               "fbeta":0.2, "horizon":100,},
            {"name": "rosetta",         "ae_dir": "x_rosetta_ae",       "data_dir":"data_oct_x_to_Rosetta.pt",  "fbeta":0.2, "horizon":100,},
            {"name": "water_converter", "ae_dir": "water_converter_ae", "data_dir":"water_converter.npy",       "fbeta":0.02, "horizon":100,},
            {"name": "gb1",             "ae_dir": "gb1_embed_ae",       "data_dir":"gb1_embed.npy",             "fbeta":0.08, "horizon":200,}]
    
    for exp in exps:
        for ballet in [True, False]:
            if ballet:
                for intersection in [True, False]:
                    acqs = ['ucb','ci'] if intersection else ['ts', 'ucb','CI']
                    for acq in acqs:
                        print(acq, exp)
                        config = Configuration(name=exp['name'], ae_dir=exp["ae_dir"], data_dir=exp["data_dir"], 
                                    run_times=run_times, horizon=exp["horizon"], acq=acq, fbeta=exp["fbeta"],
                                    intersection=intersection, ballet=ballet)
                        process(config)