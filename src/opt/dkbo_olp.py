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

class DK_BO_OLP():
    """
    distinguish history
    """
    def __init__(self, init_x_list, init_y_list, train_x, train_y, n_init:int=10, regularize=True, dynamic_weight=False, verbose=False, max_val:float=None, num_GP:int=2, pretrained_nn=None):
        # scale input
        ROBUST = True
        ScalerClass = RobustScaler if ROBUST else StandardScaler
        # init vars
        self.regularize = regularize
        self.verbose = verbose
        self.n_init = n_init // num_GP
        self.n_neighbors = min(self.n_init, 10)
        self.Lambda = 1
        self.num_GP = num_GP
        self.dynamic_weight = dynamic_weight
        self.train_x_list, self.train_y_list, self.init_x_list, self.init_y_list = [], [], init_x_list, init_y_list
        self.dkl_list = []
        self.maximum = max_val
        self.pretrained_nn = pretrained_nn

        for idx in range(num_GP):
            self.train_x_list.append(torch.from_numpy(ScalerClass().fit_transform(train_x[idx])).float())
            self.train_y_list.append(train_y[idx])
            self.train_iter = 10
            # self.init_x_list.append(self.train_x_list[-1][:n_init])
            # self.init_y_list.append(self.train_y_list[-1][:n_init])
            tmp_dkl = DKL(self.init_x_list[idx], self.init_y_list[idx].squeeze(), n_iter=self.train_iter, low_dim=True,  pretrained_nn=self.pretrained_nn)
            self.dkl_list.append(tmp_dkl)

        self.cuda = torch.cuda.is_available()
        # gpytorch.add_jitter(self.dkl.model.)

        for idx in range(num_GP):
            self.train(idx)
    
    def train(self, idx):
        if self.regularize:
            self.dkl_list[idx].train_model_kneighbor_collision(self.n_neighbors, Lambda=self.Lambda, dynamic_weight=self.dynamic_weight, return_record=False, verbose=self.verbose)
        else:
            self.dkl_list[idx].train_model()  

    def query(self, n_iter:int=10, acq="ts", verbose=False):
        self.observed = []
        self.regret = np.zeros(n_iter)
        iterator = tqdm.notebook.tqdm(range(n_iter)) if verbose else range(n_iter)

        for i in iterator:
            candidate_idx_list, candidate_acq_values = [], []
            for idx in range(self.num_GP):
                candidate_idx_list.append(self.dkl_list[idx].next_point(self.train_x_list[idx], acq, "love", return_idx=True))
                candidate_acq_values.append(self.dkl_list[idx].acq_val[candidate_idx_list[-1]].to("cpu"))

            candidate_model_idx = np.argmax(candidate_acq_values)
            candidate_idx = candidate_idx_list[candidate_model_idx]
            self.observed.append([candidate_model_idx, candidate_idx])
            
            # update specific model
            self.init_x_list[candidate_model_idx] = torch.cat([self.init_x_list[candidate_model_idx], self.train_x_list[candidate_model_idx][candidate_idx].reshape(1,-1)], dim=0)
            self.init_y_list[candidate_model_idx] = torch.cat([self.init_y_list[candidate_model_idx], self.train_y_list[candidate_model_idx][candidate_idx].reshape(1,-1)])
            # retrain
            self.dkl_list[candidate_model_idx] = DKL(self.init_x_list[candidate_model_idx], self.init_y_list[candidate_model_idx].squeeze(), n_iter=self.train_iter, low_dim=True,  pretrained_nn=self.pretrained_nn)
            # self.dkl.train_model_kneighbor_collision(self.n_neighbors, Lambda=self.Lambda, dynamic_weight=self.dynamic_weight, return_record=False)
            self.train(candidate_model_idx)
            # regret
            # print(torch.cat(self.init_y_list).numpy(), self.maximum)
            self.regret[i] = self.maximum - torch.max(torch.cat(self.init_y_list))
            if self.regret[i] < 1e-10:
                break
            if verbose:
                iterator.set_postfix(loss=self.regret[i])
    
    def ucb_func(self):
        for id, _dkl in enumerate(self.dkl_list):
            _dkl.model.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    _observed_pred = _dkl.likelihood(_dkl.model(scaled_input_tensor.to(DEVICE)))
                    _, _upper = _observed_pred.confidence_region()
            ucb = _upper if id == 0 else torch.max(ucb, _upper)
        return ucb.to('cpu')
