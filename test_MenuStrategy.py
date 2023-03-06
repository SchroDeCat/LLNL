# from email import iterators
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
from src.utils import save_res, _path, _random_seed_gen
from src.opt import pure_dkbo, DKBO_OLP_MenuStrategy

def _batched_test(x_tensor, y_tensor, init_strategy:str="kmeans", n_init=10, n_repeat=2, num_GP=2, train_times=10, batch_size:int=5,
                    n_iter=40, acq="ts", verbose=True, lr=1e-2, name="test", return_result=True, ucb_strategy="exact",
                    plot_result=False, save_result=False, save_path=None, fix_seed=False,  pretrained=False, ae_loc=None):
    '''
    Test the batched DKBO-OLP
    '''
    obj = y_tensor.max().item()
    regret = np.zeros([n_repeat, n_iter])
    for r in range(n_repeat):

        # fix the seed
        if fix_seed:
            _seed = int(_random_seed_gen()[r])
            torch.manual_seed(_seed)
            np.random.seed(_seed)
            random.seed(_seed)
            torch.cuda.manual_seed(_seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        observed_y = []
        x_init, y_init = x_tensor[:n_init], y_tensor[:n_init]

        _dkbo_olp_batch = DKBO_OLP_MenuStrategy(x_tensor=x_init, y_tensor=y_init, partition_strategy=init_strategy, num_GP=num_GP,
                                                train_times=train_times, acq=acq, verbose=verbose, lr=lr, name=name, ucb_strategy=ucb_strategy,
                                                train=True, pretrained=pretrained, ae_loc=ae_loc)

        assert n_iter % batch_size == 0
        iterator = tqdm.tqdm(range(0, n_iter, batch_size), desc=f"R {r}") if verbose else range(0, n_iter, batch_size)

        for t in iterator:
            candidates_idx = _dkbo_olp_batch.select(x_tensor, i=0, k=batch_size)
            _x_query, y_query = x_tensor[candidates_idx], y_tensor[candidates_idx]
            _dkbo_olp_batch.update(X=_x_query, y=y_query)
            observed_y.extend([val.item() for val in y_query])

        regret[r] = obj - np.maximum.accumulate(observed_y)
    

    reg_output_record = regret.mean(axis=0)
    
    name = f"{name}-B{batch_size}"
    if save_result:
        assert not (save_path is None)
        save_res(save_path=save_path, name=name, res=regret, n_repeat=n_repeat, num_GP=num_GP, n_iter=n_iter, train_iter=train_times,
                init_strategy=init_strategy, cluster_interval=0, acq=acq, lr=lr, ucb_strategy=ucb_strategy, verbose=verbose,)

    if plot_result:
        plt.plot(reg_output_record)
        plt.ylabel("regret")
        plt.xlabel("Iteration")
        plt.title(f'simple regret for {name}')
        _img_path = _path(save_path, name, init_strategy, n_repeat, num_GP, n_iter, 0, acq, lr, train_times, ucb_strategy)
        plt.savefig(_img_path)
        if verbose:
            print(f"IMG saved to {_img_path}")


    if return_result:
        return regret

if __name__ == "__main__":
    # parse the cli
    cli_parser = argparse.ArgumentParser(description="configuration of mf test")
    cli_parser.add_argument("--name", nargs='?', default="test", type=str, help="name of experiment",)
    cli_parser.add_argument("--aedir", nargs='?', default="none", type=str, help="directory of the pretrained Autoencoder",)
    cli_parser.add_argument("--subdir", nargs='?', default="./res", type=str, help="directory to store the scripts and results",)
    cli_parser.add_argument("--datadir", nargs='?', default="./data", type=str, help="directory of the test datasets")
    cli_parser.add_argument("-a",  action="store_true", default=False, help="flag of if retrain AE")
    cli_parser.add_argument("-v",  action="store_true", default=False, help="flag of if verbose")
    cli_parser.add_argument("-f",  action="store_true", default=False, help="flag of if using fixed seed")
    cli_parser.add_argument("-p",  action="store_true", default=False, help="flag of if plotting result")
    cli_parser.add_argument("-s",  action="store_true", default=False, help="flag of if storing result")
    cli_parser.add_argument("-n", action="store_true", default=False, help="flag of if negate the obj value to maximize")
    cli_parser.add_argument("-o", action="store_true", default=False, help="if partition the space")
    cli_parser.add_argument("--learning_rate", nargs='?', default=2, type=int, help="rank of the learning rate")
    cli_parser.add_argument("--init_num", nargs='?', default=10, type=int, help="number of initial random points")
    cli_parser.add_argument("--run_times", nargs='?', default=5, type=int, help="run times of the tests")
    cli_parser.add_argument("--opt_horizon", nargs='?', default=40, type=int, help="horizon of the optimization")
    cli_parser.add_argument("--train_times", nargs='?', default=100, type=int, help="number of training iterations")
    cli_parser.add_argument("--batch-size", nargs="?", default=5, type=int,help="Batch size of each round of query")    
    cli_parser.add_argument("--acq_func", nargs='?', default="ts", type=str, help="acquisition function")
    cli_parser.add_argument("--clustering", nargs='?', default="kmeans-y", type=str, help="cluster strategy")
    cli_parser.add_argument("--n_partition", nargs='?', default=2, type=int, help="number of partition")
    cli_parser.add_argument("--ucb_strategy", nargs='?', default="max", type=str, help="strategy to combine ucbs")
    
    
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
    pretrained = not (cli_args.aedir == 'none')

    # pretrain AE
    if pretrained and cli_args.a:
        ae = AE(scaled_input_tensor, lr=1e-3)
        ae.train_ae(epochs=100, batch_size=200, verbose=True)
        torch.save(ae.state_dict(), cli_args.aedir)
        if cli_args.v:
            print(f"pretrained ae stored in {cli_args.aedir}")


    print(f"Learning rate {learning_rate} Partition {cli_args.o} ucb strategy {cli_args.ucb_strategy}")
    if cli_args.o:
        _batched_test(x_tensor=scaled_input_tensor, y_tensor=train_output, init_strategy=cli_args.clustering, n_init=cli_args.init_num, n_repeat=cli_args.run_times, num_GP=cli_args.n_partition, batch_size=cli_args.batch_size,
                        n_iter=cli_args.opt_horizon, acq=cli_args.acq_func, verbose=verbose, lr=learning_rate, name=cli_args.name, train_times=cli_args.train_times,
                        plot_result=cli_args.p, save_result=cli_args.s, save_path=cli_args.subdir, return_result=True, fix_seed=fix_seed,  pretrained=pretrained, ae_loc=cli_args.aedir, ucb_strategy=cli_args.ucb_strategy)
    else:
        pure_dkbo(x_tensor=scaled_input_tensor, y_tensor=train_output,  n_init=cli_args.init_num, n_repeat=cli_args.run_times, 
                        n_iter=cli_args.opt_horizon, acq=cli_args.acq_func, verbose=verbose, lr=learning_rate, name=cli_args.name, train_iter=cli_args.train_times,
                        plot_result=cli_args.p, save_result=cli_args.s, save_path=cli_args.subdir, return_result=True, fix_seed=fix_seed,  pretrained=pretrained, ae_loc=cli_args.aedir,)
