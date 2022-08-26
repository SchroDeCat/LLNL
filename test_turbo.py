from email import iterators
import gpytorch
import os
import random
import torch
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

from tqdm import tqdm
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
from src.utils import save_res, _path, _random_seed_gen
from src.turbo import TuRBO


if __name__ == "__main__":
    # parse the cli
    cli_parser = argparse.ArgumentParser(description="configuration of mf test")
    cli_parser.add_argument("--name", nargs='?', default="test", type=str, help="name of experiment",)
    cli_parser.add_argument("--aedir", nargs='?', default="none", type=str, help="directory of the pretrained Autoencoder",)
    cli_parser.add_argument("--subdir", nargs='?', default="./res", type=str, help="directory to store the scripts and results",)
    cli_parser.add_argument("--datadir", nargs='?', default="./data", type=str, help="directory of the test datasets")
    cli_parser.add_argument("-a",  action="store_true", default=False, help="flag of if retrain AE")
    cli_parser.add_argument("-c",  action="store_true", default=False, help="flag of if continuous")
    cli_parser.add_argument("-v",  action="store_true", default=False, help="flag of if verbose")
    cli_parser.add_argument("-f",  action="store_true", default=False, help="flag of if using fixed seed")
    cli_parser.add_argument("-p",  action="store_true", default=False, help="flag of if plotting result")
    cli_parser.add_argument("-s",  action="store_true", default=False, help="flag of if storing result")
    cli_parser.add_argument("-n", action="store_true", default=False, help="flag of if negate the obj value to maximize")
    # cli_parser.add_argument("-o", action="store_true", default=False, help="if partition the space")
    cli_parser.add_argument("--learning_rate", nargs='?', default=2, type=int, help="rank of the learning rate")
    cli_parser.add_argument("--init_num", nargs='?', default=10, type=int, help="number of initial random points")
    cli_parser.add_argument("--run_times", nargs='?', default=5, type=int, help="run times of the tests")
    cli_parser.add_argument("--opt_horizon", nargs='?', default=40, type=int, help="horizon of the optimization")
    cli_parser.add_argument("--train_times", nargs='?', default=100, type=int, help="number of training iterations")
    cli_parser.add_argument("--batch-size", nargs="?", default=1, type=int,help="Batch size of each round of query")    
    cli_parser.add_argument("--acq_func", nargs='?', default="ts", type=str, help="acquisition function")
    # cli_parser.add_argument("--clustering", nargs='?', default="kmeans-y", type=str, help="cluster strategy")
    # cli_parser.add_argument("--n_partition", nargs='?', default=2, type=int, help="number of partition")
    # cli_parser.add_argument("--ucb_strategy", nargs='?', default="max", type=str, help="strategy to combine ucbs")
    
    
    cli_args = cli_parser.parse_args()

    ### load dataset
    assert not (cli_args.datadir is None)
    if cli_args.datadir.endswith(".pt"):
        dataset = torch.load(cli_args.datadir)  # want to find the maximal
    elif cli_args.datadir.endswith(".npy"):
        dataset = torch.from_numpy(np.load(cli_args.datadir))
    # ub = dataset.numpy().max(axis=0)
    # lb = dataset.numpy().min(axis=0)
    # dataset = torch.from_numpy(dataset.numpy()[:, ub>lb])
    data_dim = dataset.shape[1]-1

    # original Objective
    ROBUST = True
    ScalerClass = RobustScaler if ROBUST else StandardScaler
    scaler = ScalerClass().fit(dataset[:,:-1])                   # obj foldx -> rosetta
    scaled_f_data = scaler.transform(dataset[:,:-1])
    scaled_input_tensor = torch.from_numpy(scaled_f_data).double()
    output_tensor = dataset[:,-1].double()
    shuffle_filter = np.random.choice(scaled_f_data.shape[0], scaled_f_data.shape[0], replace=False)
    train_output = -1 * output_tensor.reshape([-1,1]).double() if cli_args.n else output_tensor.reshape([-1,1]).float()

    # dkbo experiment
    verbose = cli_args.v
    fix_seed = cli_args.f
    lr_rank = -cli_args.learning_rate
    learning_rate = 10 ** lr_rank
    pretrained = not (cli_args.aedir == 'none')

    # pretrain AE
    # print("Load AE")
    if pretrained and cli_args.a:
        ae = AE(scaled_input_tensor, lr=1e-3)
        ae.train_ae(epochs=100, batch_size=200, verbose=True)
        torch.save(ae.state_dict(), cli_args.aedir)
        if cli_args.v:
            print(f"pretrained ae stored in {cli_args.aedir}")
    elif pretrained:
        assert not (cli_args.aedir is None)
        ae = AE(scaled_input_tensor, lr=1e-3)
        ae.load_state_dict(torch.load(cli_args.aedir, map_location="cpu"))
    else:
        ae = None

    name = cli_args.name
    verbose = cli_args.v
    save_path = cli_args.subdir
    n_repeat = cli_args.run_times
    n_init = cli_args.init_num
    n_iter = cli_args.opt_horizon
    discrete = not cli_args.c
    test_x, test_y = scaled_input_tensor, output_tensor
    batch_size = cli_args.batch_size
    reg_record = np.zeros([n_repeat, batch_size * n_iter + n_init])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for idx in range(n_repeat):
            print(f"Round {idx}")
            if cli_args.f:
                # print(n_init+n_repeat*n_iter)
                # _seed = rep*n_iter + n_init
                _seed = _random_seed_gen()[idx]
                torch.manual_seed(_seed)
                np.random.seed(_seed)
                random.seed(_seed)
                torch.cuda.manual_seed(_seed)
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True


            t = TuRBO(test_x, test_y, n_init=n_init, verbose=verbose, discrete=discrete, batch_size=batch_size, train_iter=cli_args.train_times, pretrained_nn=ae)
            t.opt(n_iter,)
            reg_record[idx] = t.regret.squeeze()

    if cli_args.p:
        plt.plot(reg_record.mean(axis=0))


    if cli_args.s:
        np.save(f"{save_path}/turbo{'-bo' if not discrete else ''}-{name}-R{n_repeat}-T{n_iter}.npy", reg_record)

    if cli_args.p:
        plt.plot(reg_record.mean(axis=0))
        plt.ylabel("regret")
        plt.xlabel("Iteration")
        plt.title(f'simple regret for {name}')
        _path = f"{save_path}/turbo{'-bo' if not discrete else ''}-{name}-R{n_repeat}-T{n_iter}"
        plt.savefig(f"{_path}.png")
