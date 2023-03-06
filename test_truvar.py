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
from src.opt import ol_partition_dkbo, pure_dkbo, ol_filter_dkbo, truvar
# from src.utils import 


if __name__ == "__main__":
    # parse the cli
    cli_parser = argparse.ArgumentParser(description="configuration of mf test")
    cli_parser.add_argument("--name", nargs='?', default="test", type=str, help="name of experiment",)
    cli_parser.add_argument("--aedir", nargs='?', default=None, type=str, help="directory of the pretrained Autoencoder",)
    cli_parser.add_argument("--subdir", nargs='?', default="./res", type=str, help="directory to store the scripts and results",)
    cli_parser.add_argument("--datadir", nargs='?', default="./data", type=str, help="directory of the test datasets")
    cli_parser.add_argument("-a",  action="store_true", default=False, help="flag of if retrain AE")
    cli_parser.add_argument("-v",  action="store_true", default=False, help="flag of if verbose")
    cli_parser.add_argument("-f",  action="store_true", default=False, help="flag of if using fixed seed")
    cli_parser.add_argument("-r",  action="store_true", default=False, help="flag of if using regularization")
    cli_parser.add_argument("-p",  action="store_true", default=False, help="flag of if plotting result")
    cli_parser.add_argument("-s",  action="store_true", default=False, help="flag of if storing result")
    cli_parser.add_argument("-n", action="store_true", default=False, help="flag of if negate the obj value to maximize")
    cli_parser.add_argument("-o", action="store_true", default=False, help="if filtering the space")
    cli_parser.add_argument("--high_dim", action="store_true", default=False, help="if using high dim latent space")
    cli_parser.add_argument("--learning_rate", nargs='?', default=2, type=int, help="rank of the learning rate")
    cli_parser.add_argument("--init_num", nargs='?', default=10, type=int, help="number of initial random points")
    cli_parser.add_argument("--beta", nargs='?', default=2, type=float, help="confidence factor")
    cli_parser.add_argument("--fbeta", nargs='?', default=2, type=float, help="filtering factor")
    cli_parser.add_argument("--run_times", nargs='?', default=5, type=int, help="run times of the tests")
    cli_parser.add_argument("--opt_horizon", nargs='?', default=40, type=int, help="horizon of the optimization")
    cli_parser.add_argument("--train_times", nargs='?', default=100, type=int, help="number of training iterations")

    # cli_parser.add_argument("--train_interval", nargs='?', default=1, type=int, help="retrain interval")
    
    # cli_parser.add_argument("--acq_func", nargs='?', default="ts", type=str, help="acquisition function")
    cli_parser.add_argument("--filter_interval", nargs='?', default=10, type=int, help="filtering interval")
    cli_parser.add_argument("--intersection", action="store_false", default=True, help="if using CI's intersection")
    
    
    cli_args = cli_parser.parse_args()
    ### load dataset
    assert not (cli_args.datadir is None)
    if cli_args.datadir.endswith(".pt"):
        dataset = torch.load(cli_args.datadir)  # want to find the maximal
    elif cli_args.datadir.endswith(".npy"):
        dataset = torch.from_numpy(np.load(cli_args.datadir))
    data_dim = dataset.shape[1]-1

    # original Objective
    ROBUST = True
    ScalerClass = RobustScaler if ROBUST else StandardScaler
    scaler = ScalerClass().fit(dataset[:,:-1])                   # obj foldx -> rosetta
    scaled_f_data = scaler.transform(dataset[:,:-1])
    scaled_input_tensor = torch.from_numpy(scaled_f_data).float()
    output_tensor = dataset[:,-1].float()
    shuffle_filter = np.random.choice(scaled_f_data.shape[0], scaled_f_data.shape[0], replace=False)
    train_output = -1 * output_tensor.reshape([-1,1]).float() if cli_args.n else output_tensor.reshape([-1,1]).float()

    # dkbo experiment
    verbose = cli_args.v
    fix_seed = cli_args.f
    lr_rank = -cli_args.learning_rate
    learning_rate = 10 ** lr_rank
    low_dim = not cli_args.high_dim
    pretrained = not (cli_args.aedir is None)

    # pretrain AE
    if not (cli_args.aedir is None) and cli_args.a:
        ae = AE(scaled_input_tensor, lr=1e-3)
        # print(scaled_input_tensor.shape)
        ae.train_ae(epochs=10, batch_size=200, verbose=True)
        torch.save(ae.state_dict(), cli_args.aedir)
        if cli_args.v:
            print(f"pretrained ae stored in {cli_args.aedir}")


    print(f"Learning rate {learning_rate} Filtering {cli_args.o} fix_seed {fix_seed} beta {cli_args.beta} Regularize {cli_args.r} Low dim {low_dim} CI intersection {cli_args.intersection} verbose={verbose}")
    if True:
        res = truvar(x_tensor=scaled_input_tensor, y_tensor=train_output, n_init=cli_args.init_num, n_repeat=cli_args.run_times, low_dim=low_dim, beta=cli_args.beta, ci_intersection=cli_args.intersection,
                        n_iter=cli_args.opt_horizon, filter_interval=cli_args.filter_interval, verbose=verbose, lr=learning_rate, name=cli_args.name, train_times=cli_args.train_times, filter_beta=cli_args.fbeta,
                        plot_result=cli_args.p, save_result=cli_args.s, save_path=cli_args.subdir, return_result=True, fix_seed=fix_seed,  pretrained=pretrained, ae_loc=cli_args.aedir)
