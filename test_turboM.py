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
from src.utils import save_res, _path, _random_seed_gen, Data_Factory
from src.turbo import TurboM       

if __name__ == "__main__":
    # parse the cli
    cli_parser = argparse.ArgumentParser(description="configuration of mf test")
    cli_parser.add_argument("--name", nargs='?', default="test", type=str, help="name of experiment",)
    cli_parser.add_argument("--aedir", nargs='?', default="none", type=str, help="directory of the pretrained Autoencoder",)
    cli_parser.add_argument("--subdir", nargs='?', default="./res", type=str, help="directory to store the scripts and results",)
    cli_parser.add_argument("--datadir", nargs='?', default="./data", type=str, help="directory of the test datasets")
    cli_parser.add_argument("-v",  action="store_true", default=False, help="flag of if verbose")
    cli_parser.add_argument("-p",  action="store_true", default=False, help="flag of if plotting result")
    cli_parser.add_argument("-s",  action="store_true", default=False, help="flag of if storing result")
    cli_parser.add_argument("-n", action="store_true", default=False, help="flag of if negate the obj value to maximize")
    cli_parser.add_argument("-f",  action="store_true", default=False, help="flag of if using fixed seed")

    cli_parser.add_argument("--init_num", nargs='?', default=10, type=int, help="number of initial random points")
    cli_parser.add_argument("--run_times", nargs='?', default=5, type=int, help="run times of the tests")
    cli_parser.add_argument("--opt_horizon", nargs='?', default=40, type=int, help="horizon of the optimization")
    cli_parser.add_argument("--n_trust_region", nargs='?', default=5, type=int, help="number of trust regions")
    cli_parser.add_argument("--solver", nargs='?', default='dkbo', type=str, help="solver type: dkbo,bo, turbo")
    # cli_parser.add_argument("--learning_rate", nargs='?', default=2, type=int, help="rank of the learning rate")
    cli_parser.add_argument("-a",  action="store_true", default=False, help="flag of if retrain AE")
    
    
    cli_args = cli_parser.parse_args()
    ### load dataset
    assert not (cli_args.datadir is None)
    if cli_args.datadir.endswith(".pt"):
        dataset = torch.load(cli_args.datadir).numpy()  # want to find the maximal
    elif cli_args.datadir.endswith(".npy"):
        dataset = np.load(cli_args.datadir)
    # ub = dataset.max(axis=0)
    # lb = dataset.min(axis=0)

    # dataset = dataset[:, ub>lb]
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

    # dkbo parameters
    verbose = cli_args.v
    fix_seed = cli_args.f
    # lr_rank = -cli_args.learning_rate
    # learning_rate = 10 ** lr_rank
    pretrained = not (cli_args.aedir == 'none')

    # pretrain AE
    if pretrained and cli_args.a:
        ae = AE(torch.from_numpy(scaled_input), lr=1e-3)
        ae.train_ae(epochs=100, batch_size=200, verbose=True)
        torch.save(ae.state_dict(), cli_args.aedir)
        if cli_args.v:
            print(f"pretrained ae stored in {cli_args.aedir}")
    elif pretrained:
        assert not (cli_args.aedir is None)
        ae = AE(torch.from_numpy(scaled_input), lr=1e-3)
        ae.load_state_dict(torch.load(cli_args.aedir, map_location="cpu"))
    else:
        ae = None

    # LAMCTS experiment
    verbose = cli_args.v
    dataset = np.hstack([scaled_input, output[:,np.newaxis]])
    approximate = lambda x: Data_Factory.nearest(dataset, x)

    def approx_obj_func(multi_rows):
        row_num = multi_rows.shape[0]
        assert row_num > 1
        res = np.array([approximate(multi_rows[i])[0][-1] for i in range(row_num)])
        return res

    def obj_func_simple(config):
        _, index = Data_Factory.nearest(dataset, config)
        return dataset[index, -1]

    def obj_func_simple_neg(config):
        _, index = Data_Factory.nearest(dataset, config)
        return -dataset[index, -1]


    lb = scaled_input.min(axis=0) - 1e-10
    ub = scaled_input.max(axis=0)
    optima = dataset[:,-1].max()
    kernel_type = 'rbf'
    gamma_type = "auto"
    # solver_type = 'bo'   # turbo not available
    # solver_type = 'dkbo'
    solver_type = cli_args.solver
    obj_func_simple.lb = lb
    obj_func_simple.ub = ub

    name = cli_args.name
    save_path = cli_args.subdir
    n_repeat = cli_args.run_times
    n_iter = cli_args.opt_horizon
    # init_x = dataset[:cli_args.init_num,:-1]
    # init_y = approx_obj_func(init_x)
    # print(f"init max {init_y.max()} {dataset[:cli_args.init_num,-1].max()} global maximum {dataset[:,-1].max()}")
    reg_record = np.zeros([n_repeat, n_iter])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for idx in tqdm(range(n_repeat)):
            # print(f"Round {idx}")
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


            # minimize the negative function
            turbo5 = TurboM(f=obj_func_simple_neg, lb=lb, ub=ub, n_init=cli_args.init_num,  verbose=cli_args.v,
                            max_evals=cli_args.opt_horizon, n_trust_regions=cli_args.n_trust_region)
            turbo5.optimize()  # Run optimization
            X, fX = turbo5.X, -turbo5.fX  # Evaluated points
            reg_record[idx] = np.minimum.accumulate(optima - fX).squeeze()


            # t = TuRBO(test_x, test_y, n_init=n_init, verbose=verbose, discrete=discrete, batch_size=batch_size, pretrained_nn=ae)
            # t.opt(n_iter,)
            # reg_record[idx] = t.regret.squeeze()

    if cli_args.p:
        plt.plot(reg_record.mean(axis=0))


    if cli_args.s:
        np.save(f"{save_path}/turbo-{cli_args.n_trust_region}-{name}-R{n_repeat}-T{n_iter}.npy", reg_record)

    if cli_args.p:
        plt.plot(reg_record.mean(axis=0))
        plt.ylabel("regret")
        plt.xlabel("Iteration")
        plt.title(f'simple regret for {name}')
        _path = f"{save_path}/turbo-{cli_args.n_trust_region}-{name}-R{n_repeat}-T{n_iter}"
        plt.savefig(f"{_path}.png")


