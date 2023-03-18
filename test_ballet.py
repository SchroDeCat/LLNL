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

acq = ['ts', 'ci', 'ucb', 'qei']
# EXACT=True # if globally using exact gp
EXACT=False

# parse the cli
class Configuration():
    def __init__(self, name:str, ae_dir:str, data_dir:str, run_times:int=1, horizon:int=10, acq:str='ci', fbeta=0.2, train_time=10,
        intersection=False, ballet=True, high_dim=False, exact_gp=EXACT) -> None:
        self.name   = name
        self.aedir  = f"./tmp/{ae_dir}"
        # self.subdir = f"./res/aistats/{acq}"  
        self.subdir = f"./res/icml/{acq}"
        # self.subdir = './res/ei'
        self.datadir=f"./data/{data_dir}" 
        self.run_times  =run_times
        self.acq_func =acq
        self.learning_rate=4  
        self.intersection=intersection  
        self.a=False
        self.r=False
        self.n=False
        self.high_dim=high_dim
        self.init_num=10
        self.o=ballet   # BALLET or not
        self.p=True     # plot
        self.s=True     # store results
        self.f=True     # fix seed
        self.return_model=False 
        # self.v=True    # verbose
        self.v=False
        self.beta=0     # true beta
        # self.filter_interval=10 if horizon > 100 else horizon //10
        self.filter_interval=1
        self.opt_horizon=horizon
        self.train_times=train_time
        self.learning_rate=4
        self.fbeta = fbeta
        self.exact = exact_gp

def plot_pure_dkbo(config):
    save_path = config.subdir
    n_repeat = config.run_times
    lr_rank = -config.learning_rate
    learning_rate = 10 ** lr_rank
    lr = learning_rate
    beta = config.beta
    train_iter = config.train_times
    low_dim = not config.high_dim
    name = config.name + f"b{beta}"
    n_iter = config.opt_horizon

    
    _path = f"{save_path}/Pure-DK-{name}{'-Exact' if config.exact else ''}-{acq}-R{n_repeat}-T{n_iter}_L{int(-np.log10(lr))}-TI{train_iter}.npy"
    reg_record = np.load(_path)
    reg_output_record = reg_record.mean(axis=0)


    if not low_dim:
        name = name + "_hd"

    plt.figure()
    plt.plot(reg_output_record.squeeze(), label="dkbo")
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Regret")
    plt.title("DKBO performance")
    # plt.show()
    _path = f"{save_path}/Pure-DK-{name}{'-Exact' if config.exact else ''}-{acq}-R{n_repeat}-T{n_iter}_L{int(-np.log10(lr))}-TI{train_iter}"
    plt.savefig(f"{_path}.png")
    plt.close()


        


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
                        plot_result=config.p, save_result=config.s, save_path=config.subdir, return_result=not config.return_model, fix_seed=fix_seed,  pretrained=pretrained, ae_loc=config.aedir, exact_gp=config.exact)
    else:
        pure_dkbo(x_tensor=scaled_input_tensor, y_tensor=train_output,  n_init=config.init_num, n_repeat=config.run_times, low_dim=low_dim, beta=config.beta,
                        n_iter=config.opt_horizon, acq=config.acq_func, verbose=verbose, lr=learning_rate, name=config.name, train_iter=config.train_times, exact_gp=config.exact,
                        plot_result=config.p, save_result=config.s, save_path=config.subdir, return_result=True, fix_seed=fix_seed,  pretrained=pretrained, ae_loc=config.aedir,)



if __name__ == "__main__":
    # run_times = 10
    # run_times = 5
    run_times = 4
    # run_times = 3

    exps = [{"name": "gb1",             "ae_dir": "gb1_embed_ae",       "data_dir":"gb1_embed.npy",             "fbeta":.2, "horizon":300, "high_dim": False, 'train_iter':5},
            {"name": "nano",            "ae_dir": "nano_mf_ae",         "data_dir":"data_nano_mf.pt",           "fbeta":0.8, "horizon":200, "high_dim": True, 'train_iter':5},
            {"name": "hdbo",            "ae_dir": "200d_ae",            "data_dir":"HDBO200.npy",               "fbeta":0.2, "horizon":300, "high_dim": False, 'train_iter':5},
            {"name": "rosetta",         "ae_dir": "x_rosetta_ae",       "data_dir":"data_oct_x_to_Rosetta.pt",  "fbeta":0.2, "horizon":300, "high_dim": False, 'train_iter':5},
            {"name": "water_converter", "ae_dir": "water_converter_ae", "data_dir":"water_converter.npy",       "fbeta":80, "horizon":200, "high_dim": True, 'train_iter':5},
            ]
    # exps = [{"name": "water_converter", "ae_dir": "water_converter_ae", "data_dir":"water_converter.npy",       "fbeta":80, "horizon":20, "high_dim": True, 'train_iter':10},]
    # exps = [{"name": "water_converter", "ae_dir": "water_converter_ae", "data_dir":"water_converter.npy",       "fbeta":80, "horizon":100, "high_dim": True, 'train_iter':10},]
    # exps =  [{"name": "eg1d",           "ae_dir": "1deg_ae",            "data_dir":"opt_eg1d.npy",              "fbeta":0.2, "horizon":50, "high_dim": False, 'train_iter':10},
    #         {"name": "nano",            "ae_dir": "nano_mf_ae",         "data_dir":"data_nano_mf.pt",           "fbeta":0.8, "horizon":100, "high_dim": True, 'train_iter':10},
    #         {"name": "water_converter", "ae_dir": "water_converter_ae", "data_dir":"water_converter.npy",       "fbeta":80, "horizon":100, "high_dim": True, 'train_iter':10},
    #         ]
    # exps = [{"name": "hdbo",            "ae_dir": "200d_ae",            "data_dir":"HDBO200.npy",               "fbeta":0.2, "horizon":100, "high_dim": False, 'train_iter':15},]
    # exps = [{"name": "gb1",             "ae_dir": "gb1_embed_ae",       "data_dir":"gb1_embed.npy",             "fbeta":.2, "horizon":100, "high_dim": False, 'train_iter':10},
    #         {"name": "eg1d",            "ae_dir": "1deg_ae",            "data_dir":"opt_eg1d.npy",              "fbeta":0.2, "horizon":50, "high_dim": False, 'train_iter':10},
    #         {"name": "nano",            "ae_dir": "nano_mf_ae",         "data_dir":"data_nano_mf.pt",           "fbeta":0.8, "horizon":100, "high_dim": True, 'train_iter':10},
    #         {"name": "hdbo",            "ae_dir": "200d_ae",            "data_dir":"HDBO200.npy",               "fbeta":0.2, "horizon":100, "high_dim": False, 'train_iter':10},
    #         {"name": "rosetta",         "ae_dir": "x_rosetta_ae",       "data_dir":"data_oct_x_to_Rosetta.pt",  "fbeta":0.2, "horizon":100, "high_dim": False, 'train_iter':10},
    #         {"name": "water_converter", "ae_dir": "water_converter_ae", "data_dir":"water_converter.npy",       "fbeta":80, "horizon":100, "high_dim": True, 'train_iter':10},
    #         ]
    # exps = [{"name": "nano",            "ae_dir": "nano_mf_ae",         "data_dir":"data_nano_mf.pt",           "fbeta":0.2, "horizon":100, "high_dim": True, 'train_iter':10},]
            # {"name": "water_converter", "ae_dir": "water_converter_ae", "data_dir":"water_converter.npy",       "fbeta":0.2, "horizon":100, "high_dim": True, 'train_iter':1},]
    # exps = [{"name": "nano",            "ae_dir": "nano_mf_ae",         "data_dir":"data_nano_mf.pt",           "fbeta":0.2, "horizon":100, "high_dim": True, 'train_iter':10},
            # {"name": "water_converter", "ae_dir": "water_converter_ae", "data_dir":"water_converter.npy",       "fbeta":0.2, "horizon":100, "high_dim": True, 'train_iter':10},]
            # {"name": "eg1d",            "ae_dir": "1deg_ae",            "data_dir":"opt_eg1d.npy",              "fbeta":0.2, "horizon":50, "high_dim": False, 'train_iter':10}]
    # exps = [{"name": "eg1d",            "ae_dir": "1deg_ae",            "data_dir":"opt_eg1d.npy",              "fbeta":0.2, "horizon":50, "high_dim": False, 'train_iter':10}]
    # exps = [{"name": "gb1",             "ae_dir": "gb1_embed_ae",       "data_dir":"gb1_embed.npy",             "fbeta":.2, "horizon":100, "high_dim": False, 'train_iter':10},
    #         {"name": "eg1d",            "ae_dir": "1deg_ae",            "data_dir":"opt_eg1d.npy",              "fbeta":0.2, "horizon":50, "high_dim": False, 'train_iter':10},
    #         {"name": "nano",            "ae_dir": "nano_mf_ae",         "data_dir":"data_nano_mf.pt",           "fbeta":0.2, "horizon":100, "high_dim": False, 'train_iter':10},
    #         {"name": "hdbo",            "ae_dir": "200d_ae",            "data_dir":"HDBO200.npy",               "fbeta":0.2, "horizon":100, "high_dim": False, 'train_iter':10},
    #         {"name": "rosetta",         "ae_dir": "x_rosetta_ae",       "data_dir":"data_oct_x_to_Rosetta.pt",  "fbeta":0.2, "horizon":100, "high_dim": False, 'train_iter':10},
    #         {"name": "water_converter", "ae_dir": "water_converter_ae", "data_dir":"water_converter.npy",       "fbeta":0.2, "horizon":100, "high_dim": False, 'train_iter':10},
    #         ]
    # exps = [{"name": "nano",            "ae_dir": "nano_mf_ae",         "data_dir":"data_nano_mf.pt",        "fbeta":0.2, "horizon":100, "high_dim": False, 'train_iter':10},]
    # exps = [{"name": "water_converter", "ae_dir": "water_converter_ae", "data_dir":"water_converter.npy",       "fbeta":80, "horizon":100, "high_dim": True, 'train_iter':100},]
    # exps = [{"name": "hdbo",            "ae_dir": "200d_ae",            "data_dir":"HDBO200.npy",               "fbeta":0.8, "horizon":100, "high_dim": False, 'train_iter':100},]
    # exps = [{"name": "gb1",             "ae_dir": "gb1_embed_ae",       "data_dir":"gb1_embed.npy",             "fbeta":.05, "horizon":100, "high_dim": True, 'train_iter':200}]
    # exps = [{"name": "gb1",             "ae_dir": "gb1_embed_ae",       "data_dir":"gb1_embed.npy",             "fbeta":.1, "horizon":100, "high_dim": True},
    #         {"name": "eg1d",            "ae_dir": "1deg_ae",            "data_dir":"opt_eg1d.npy",              "fbeta":0.2, "horizon":50, "high_dim": True},
    #         {"name": "nano",            "ae_dir": "nano_mf_ae",         "data_dir":"data_nano_mf.pt",           "fbeta":0.8, "horizon":100, "high_dim": True},
    #         {"name": "hdbo",            "ae_dir": "200d_ae",            "data_dir":"HDBO200.npy",               "fbeta":0.8, "horizon":100, "high_dim": True},
    #         {"name": "rosetta",         "ae_dir": "x_rosetta_ae",       "data_dir":"data_oct_x_to_Rosetta.pt",  "fbeta":0.2, "horizon":100, "high_dim": True},
    #         {"name": "water_converter", "ae_dir": "water_converter_ae", "data_dir":"water_converter.npy",       "fbeta":0.8, "horizon":100, "high_dim": True},
    #         {"name": "gb1",             "ae_dir": "gb1_embed_ae",       "data_dir":"gb1_embed.npy",             "fbeta":.1, "horizon":100, "high_dim": False},
    #         {"name": "eg1d",            "ae_dir": "1deg_ae",            "data_dir":"opt_eg1d.npy",              "fbeta":0.2, "horizon":50, "high_dim": False},
    #         {"name": "nano",            "ae_dir": "nano_mf_ae",         "data_dir":"data_nano_mf.pt",           "fbeta":0.8, "horizon":100, "high_dim": False},
    #         {"name": "hdbo",            "ae_dir": "200d_ae",            "data_dir":"HDBO200.npy",               "fbeta":0.8, "horizon":100, "high_dim": False},
    #         {"name": "rosetta",         "ae_dir": "x_rosetta_ae",       "data_dir":"data_oct_x_to_Rosetta.pt",  "fbeta":0.2, "horizon":100, "high_dim": False},
    #         {"name": "water_converter", "ae_dir": "water_converter_ae", "data_dir":"water_converter.npy",       "fbeta":0.8, "horizon":100, "high_dim": False},
    #         ]
    # exps = [{"name": "rosetta",         "ae_dir": "x_rosetta_ae",       "data_dir":"data_oct_x_to_Rosetta.pt",  "fbeta":0.2, "horizon":100, "high_dim": False, 'train_iter':100},
    #         {"name": "rosetta",         "ae_dir": "x_rosetta_ae",       "data_dir":"data_oct_x_to_Rosetta.pt",  "fbeta":0.2, "horizon":100, "high_dim": True,  'train_iter':100}
    #          ]
    # exps = [{"name": "nano",            "ae_dir": "nano_mf_ae",         "data_dir":"data_nano_mf.pt",           "fbeta":1.0, "horizon":100, "high_dim": True, 'train_iter':100},
    #         {"name": "nano",            "ae_dir": "nano_mf_ae",         "data_dir":"data_nano_mf.pt",           "fbeta":0.6, "horizon":100, "high_dim": True, 'train_iter':100},
    #         {"name": "nano",            "ae_dir": "nano_mf_ae",         "data_dir":"data_nano_mf.pt",           "fbeta":0.8, "horizon":100, "high_dim": True, 'train_iter':100},
    #         {"name": "nano",            "ae_dir": "nano_mf_ae",         "data_dir":"data_nano_mf.pt",           "fbeta":0.4, "horizon":100, "high_dim": True, 'train_iter':100},
    #         {"name": "nano",            "ae_dir": "nano_mf_ae",         "data_dir":"data_nano_mf.pt",           "fbeta":0.2, "horizon":100, "high_dim": True, 'train_iter':100},
    #          ]
    # exps = [{"name": "nano",            "ae_dir": "nano_mf_ae",         "data_dir":"data_nano_mf.pt",           "fbeta":0.8, "horizon":100, "high_dim": True, 'train_iter':100},]
            # {"name": "nano",            "ae_dir": "nano_mf_ae",         "data_dir":"data_nano_mf.pt",           "fbeta":0.8, "horizon":100, "high_dim": True, 'train_iter':1000},]
    # exps = [{"name": "gb1",             "ae_dir": "gb1_embed_ae",       "data_dir":"gb1_embed.npy",             "fbeta":.1, "horizon":100, "high_dim": False},
    #         {"name": "eg1d",            "ae_dir": "1deg_ae",            "data_dir":"opt_eg1d.npy",              "fbeta":0.2, "horizon":50, "high_dim": False},
    #         {"name": "nano",            "ae_dir": "nano_mf_ae",         "data_dir":"data_nano_mf.pt",           "fbeta":0.8, "horizon":100, "high_dim": True},
    #         {"name": "hdbo",            "ae_dir": "200d_ae",            "data_dir":"HDBO200.npy",               "fbeta":0.8, "horizon":100, "high_dim": True},
    #         {"name": "rosetta",         "ae_dir": "x_rosetta_ae",       "data_dir":"data_oct_x_to_Rosetta.pt",  "fbeta":0.2, "horizon":100, "high_dim": False},
    #         {"name": "water_converter", "ae_dir": "water_converter_ae", "data_dir":"water_converter.npy",       "fbeta":0.8, "horizon":100, "high_dim": True},
    #         ]
            # {"name": "gb1",             "ae_dir": "gb1_embed_ae",       "data_dir":"gb1_embed.npy",             "fbeta":0.05, "horizon":200, "high_dim": True},
            # {"name": "nano",            "ae_dir": "nano_mf_ae",         "data_dir":"data_nano_mf.pt",           "fbeta":0.2, "horizon":100, "high_dim": False},
            # {"name": "hdbo",            "ae_dir": "200d_ae",            "data_dir":"HDBO200.npy",               "fbeta":0.2, "horizon":100, "high_dim": False},
            # {"name": "water_converter", "ae_dir": "water_converter_ae", "data_dir":"water_converter.npy",       "fbeta":0.02, "horizon":100,},
            # {"name": "water_converter", "ae_dir": "water_converter_ae", "data_dir":"water_converter.npy",       "fbeta":0.005, "horizon":100, "high_dim": True},
            # {"name": "gb1",             "ae_dir": "gb1_embed_ae",       "data_dir":"gb1_embed.npy",             "fbeta":0.08, "horizon":200,}]
    
    for exp in exps:
        for ballet in [True, False]:
            if ballet:
                # continue
                for intersection in [True, False]:
                    # if not intersection:
                    #     continue
                    # acqs = ['ci']
                    # acqs = ['ucb']
                    # acqs = ['ci', 'ucb'] if intersection else ['ts', 'ucb','ci', 'qei']
                    if intersection:
                        continue
                    acqs = ['qei', 'ts']

                    for acq in acqs:
                        print(acq, exp, "ballet", ballet, 'intersection', intersection)
                        config = Configuration(name=exp['name'], ae_dir=exp["ae_dir"], data_dir=exp["data_dir"], 
                                    run_times=run_times, horizon=exp["horizon"], acq=acq, fbeta=exp["fbeta"],
                                    intersection=intersection, ballet=ballet, high_dim=exp["high_dim"], train_time=exp["train_iter"])
                        process(config)
            else:
                # acqs = ['rci']
                # acqs = ['ts', 'ucb','ci', 'rci']
                # acqs = ['ts']
                acqs = ['qei', 'ts']
                for acq in acqs:
                    print(acq, exp, "ballet", ballet)
                    config = Configuration(name=exp['name'], ae_dir=exp["ae_dir"], data_dir=exp["data_dir"], 
                                        run_times=run_times, horizon=exp["horizon"], acq=acq, fbeta=exp["fbeta"],
                                        intersection=False, ballet=ballet, high_dim=exp["high_dim"], train_time=exp["train_iter"])
                    process(config)
                    # plot_pure_dkbo(config)
