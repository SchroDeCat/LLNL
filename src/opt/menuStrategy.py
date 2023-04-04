
'''
Implementation for experiments on LLNL simulation env.
'''

import pandas as pd
import numpy as np
from sklearn import cluster
import torch
import sys
import os
import dill as pickle

import gpytorch
import random
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
import copy


from ..models import DKL, AE, beta_CI
from .baseStrategy import MenuStrategy
from ..models import AE, beta_CI, SGLD
from ..utils import save_res, load_res, clustering_methods
from .dkbo_olp import DK_BO_OLP, DK_BO_OLP_Batch
from .dkbo_ae import DK_BO_AE
from abc import ABCMeta, abstractmethod
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
from sklearn.preprocessing import StandardScaler, RobustScaler
# from abag_model_development.models.active_learning.base_strategy import BaseStrategy, MenuStrategy

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore')


class DKBO_OLP(MenuStrategy):
# class DKBO_OLP():
    '''
    DKBO-Online Partition inherting Menu Strategy

    Attributes
    ----------
    name: str
        Name of active learning algo for record keeping.
    model: `DKL_BO_OLP_Batch`
        underlying model.
    train: bool
        Boolean whether we should train underlying model on some data before running selection.
    iters: int
        number of iterations to train underlying model for during intialization. Only used if train is `self.train = True`.
    partition_strategy:
        strategy of the online partition. (kmeans-y, kmeans, linear_rc, gp_rc).
    n_init: int
        number of points to initialize GP.
    num_GP: int
        number of partitions/ local GPs.
    train_times: int
        training iteration for each GP.
    acq: str
        acquisition function (UCB, TS).
    verbose: bool
        if printing verbose information.
    lr: float
        learning rate of the model.
    ucb_strategy: str
        strategy of coordinating UCB from multiple local GPs (exact, max, sum).
    pretrained: bool
        if using pretrained model to initialize the feature extracter of the DKL.
    ae_loc:
        location of the pretrained Auto Encoder.
    exact_gp:
        if using exact_gp or Deep Kernel
    '''

    def __init__(self, x_tensor:torch.tensor, y_tensor:torch.tensor, partition_strategy:str="ballet", num_GP:int=3, low_dim:bool=True,
                    train_times:int=10, acq:str="ts", verbose:bool=True, lr:float=1e-2, name:str="test", ucb_strategy:str="exact",
                    train:bool=True, pretrained:bool=False, ae_loc:str=None, abag_transform=False, exact_gp:bool=False):
        '''
        Args:
            @x_tensor: init x
            @y_tensor: init y
            @partition_strategy: strategy for the partition.
            @num_GP: desired number of local GPs.
            @train_times: number of training iteration for the models.
            @acq: acquisition function choice.
            @verbose: if printing verbose info.
            @lr: learning rate of the model.
            @name: instance name.
            @ucb_strategy: strategy to coordinate multiple local UCBs.
            @train: if updating the model.
            @pretrained: if using pretrained model to initilize the NN's weight.
            @ae_loc: location of the pretrained Auto Encoder.
            @abag_transform: boolean that indicates if we need to transform to abag closed loop

        '''
        use_gpu = (
            True if torch.cuda.is_available() or torch.backends.mps.is_available() else False
        )
        use_gpu = False # for current version
        if use_gpu:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")
        else:
            self.device = 'cpu'
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.partition_strategy = partition_strategy
        self.num_GP = num_GP
        self.acq = acq
        self.verbose = verbose
        self.lr = lr
        self.ucb_strategy = ucb_strategy
        self.pretrained = pretrained
        self.ae_loc = ae_loc
        self.low_dim = low_dim
        self.exact_gp = exact_gp
        # super().__init__(name=name, model=None, train=train, iters=train_times)
        self.iters = train_times
        self.train = train
        self.train_times = self.iters
        self.if_train = self.train
        self.abag_transform = abag_transform
        if self.abag_transform:
            x_tensor,y_tensor = self.transform_xy(x_tensor,y_tensor)
        
        # load pretrained model
        if self.pretrained:
            assert not (ae_loc is None)
            self.ae = AE(x_tensor, lr=1e-3)
            self.ae.load_state_dict(torch.load(ae_loc, map_location=self.device))
        else:
            self.ae = None

       
        # initialize the model and partitioning
        self.init_x = x_tensor
        self.init_y = y_tensor
        self.n_init = self.init_x.size(0)
        self._dkl = DKL(self.init_x, self.init_y.squeeze(), n_iter=self.train_times, low_dim=self.low_dim,  pretrained_nn=self.ae, exact_gp=self.exact_gp)
        self._dkl.train_model(verbose=verbose)
        self._dkl.model.eval()

    def fit(self,X,y,**kwargs):
        print("DKL OLP has no fit")
        pass

    def transform_xy(self,x,y, cat=False):
        y = - copy.deepcopy(y)
        if cat:
            x = torch.cat([x,y.unsqueeze(-1)],axis=1)
        x = torch.from_numpy(RobustScaler().fit_transform(copy.deepcopy(x))).float()
        return x,y
    
    def pretrain_ae(self, init_x=None, ae_path=None):
        if ae_path:
            self.ae_loc = ae_path
        ae = AE(init_x, lr=1e-3)
        ae.train_ae(epochs=100, batch_size=200, verbose=True)
        torch.save(ae.state_dict(), self.ae_loc)
        print(f"pretrained ae stored in {self.ae_loc}")

    def select(self, X:torch.tensor, i:int, k:int, **kwargs):
        """
        Select k points from the N*D search space X of the fidelity i.

        Args:
            @X: N * D matrix to select from.
            @i: fidelity number.
            @k: number of points to be picked from X

        Return:
            :return: list of indices that were selected. 
            :rtype: list
        """
        if self.abag_transform:
            X,_ = self.transform_xy(X,X, cat=False) # y doesn't matter
        x_tensor = torch.cat([self.init_x, X]) # could result in duplication
        _data_size = x_tensor.size(0)
        _util_array = np.arange(_data_size)
        self.observed = np.zeros(_data_size)
        self.observed[:self.n_init] = 1
        # partition
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            _observed_pred = self._dkl.likelihood(self._dkl.model(x_tensor.to(self.device)))
            _, ucb = _observed_pred.confidence_region()
        if self.partition_strategy in ["kmeans-y"]:
            self.cluster_id = clustering_methods(x_tensor, ucb.to('cpu'), self.num_GP, self.partition_strategy, 
                                        dkl=True, pretrained_nn=self.ae, train_iter=self.train_times)
            self.cluster_id_init = self.cluster_id[self.observed==1]

            init_x_list, init_y_list, test_x_list = [], [], []
            self.cluster_filter_list, self.cluster_idx_list = [], []
            for idx in range(self.num_GP):
                cluster_filter = self.cluster_id == idx
                cluster_filter[:self.n_init] = False # avoid querying the initial pts since it is not necessarily in X
                if np.sum(cluster_filter) == 0:   # avoid empty class -- regress to single partition.
                    cluster_filter[self.n_init:] = True 
                self.cluster_filter_list.append(cluster_filter)
                self.cluster_idx_list.append(_util_array[cluster_filter])
                test_x_list.append(x_tensor[cluster_filter])
                init_x_list.append(self.init_x)
                init_y_list.append(self.init_y.reshape([-1,1]))

        elif self.partition_strategy.lower() in ["ballet"]:
            # Identify ROI as in BALLET
            self.num_GP = 2 # override input parameter
            lcb, ucb = self._dkl.CI(x_tensor.to(self.device))
            _delta = kwargs.get("delta", 0.2)
            _filter_beta = kwargs.get("filter_beta", 0.2)
            _beta = (2 * np.log((x_tensor.shape[0] * (np.pi * (self.n_init + 1)) ** 2) /(6 * _delta))) ** 0.5 # analytic beta
            _filter_lcb, _filter_ucb = beta_CI(lcb, ucb, _filter_beta)
            ucb_filter = _filter_ucb >= _filter_lcb.max()
            _minimum_pick = min(kwargs.get("minimum_pick", 4), self.n_init)
            if sum(ucb_filter[self.observed==1]) <= _minimum_pick:
                _, indices = torch.topk(ucb[self.observed==1], min(_minimum_pick, _data_size))
                for idx in indices:
                    ucb_filter[_util_array[[self.observed==1]][idx]] = 1
            observed_unfiltered = np.min([self.observed, ucb_filter.numpy()], axis=0)      # observed and not filtered outs
            # two parition, one global, one ROI
            self.cluster_filter_list = [np.ones(_data_size), ucb_filter]
            self.cluster_idx_list = [_util_array, _util_array[ucb_filter]]
            test_x_list = [x_tensor, x_tensor[ucb_filter==1]]
            init_x_list = [self.init_x, x_tensor[observed_unfiltered==1]]
            init_y_list = [self.init_y.reshape([-1,1]), self.init_y.reshape([-1,1])[observed_unfiltered[:self.n_init]==1]]

        else:
            raise NotImplementedError(f"{self.partition_strategy} Unknown")

        self.model = DK_BO_OLP_Batch(init_x_list, init_y_list, lr=self.lr, train_iter=self.train_times,
                        n_init=self.n_init, num_GP=self.num_GP, pretrained_nn=self.ae)

        # batched query
        acq = kwargs.get("acq", "ts")
        verbose = kwargs.get("verbose", False)
        candidate_indices = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            indices = self.model.query(test_x_list=test_x_list, n_iter=k, acq=acq, verbose=verbose) # list of (model_idx, candidate_idx)
        
        # recover indices in original input X
        # Question: what if the initial pts are queried? --> avoid picking these n_init pts
        selected_idx_list = []
        for (_model_idx, _candidate_idx) in indices:
            _original_idx = self.cluster_idx_list[_model_idx][_candidate_idx] - self.n_init
            assert _candidate_idx >= 0 and _candidate_idx < _data_size
            selected_idx_list.append(_original_idx)
        return selected_idx_list, _observed_pred.mean[self.n_init:], _observed_pred.stddev[self.n_init:], _observed_pred.covariance_matrix[self.n_init:, self.n_init:]

    def visualize(self,X_tensor,y_tensor, save_path='./'):
        colors = np.array(['r','g','b'])
        labels = np.array(['gp1','gp2','gp3'])
        fig = plt.figure()
        _path = f"{save_path}"
        # should use likliehood from the global GP not the local GP
        #  use alll X,y pairs
        # partition
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            _observed_pred = self._dkl.likelihood(self._dkl.model(X_tensor.to(self.device)))
            _, ucb = _observed_pred.confidence_region()
        self.cluster_id = clustering_methods(X_tensor, ucb.to('cpu'), self.num_GP, self.partition_strategy, 
                                        dkl=True, pretrained_nn=self.ae, train_iter=self.train_times)
        # for gp_idx, (X_tensor,y_tensor,gp_model) in enumerate(zip(self.model.init_x_list, self.model.init_y_list, self.model.dkl_list)):
        #     with torch.no_grad(), gpytorch.settings.fast_pred_var():
        #         observed_pred = gp_model.likelihood(gp_model.model(X_tensor))
        #         _, ucb = observed_pred.confidence_region()
            # _file_path = _path(save_path=save_path, name=name, init_strategy=init_strategy, n_repeat=n_repeat, n_iter=n_iter, num_GP=num_GP, cluster_interval=cluster_interval,  acq=acq, train_iter=train_times)
            # plt.scatter(y_tensor, ucb, s=2)
        # for gp_idx, (X_tensor,y_tensor,gp_model) in enumerate(zip(self.model.init_x_list, self.model.init_y_list, self.model.dkl_list)):
        #     with torch.no_grad(), gpytorch.settings.fast_pred_var():
        #         observed_pred = gp_model.likelihood(gp_model.model(X_tensor))
        #         _, ucb = observed_pred.confidence_region()
        #     # _file_path = _path(save_path=save_path, name=name, init_strategy=init_strategy, n_repeat=n_repeat, n_iter=n_iter, num_GP=num_GP, cluster_interval=cluster_interval,  acq=acq, train_iter=train_times)
        #     # plt.scatter(y_tensor, ucb, s=2)
        fig  = plt.figure()
        for cluster_idx in self.cluster_id:
            inds = np.where(self.cluster_id  == cluster_idx)[0]
            plt.scatter(y_tensor[inds], ucb[inds], color=colors[cluster_idx], marker="*", s=5) # label=labels[cluster_idx]
        plt.xlabel("ddG")
        plt.ylabel("UCB")
        # plt.colorbar()
        plt.legend()
        plt.title(f"Add iteration name here")
        print('starting save')
        plt.savefig(f"{_path}.png", bbox_inches='tight', dpi=100)
        print(f"Fig stored to {_path}")
        plt.close(fig)

    def update(self,X,y,**kwargs):
        """
        Update internal model at `X` seqs with `y` observations
        """
        if self.abag_transform:
            X,y = self.transform_xy(X,y)
        # update the partitions
        self.init_x = torch.cat([self.init_x, X])
        self.init_y = torch.cat([self.init_y, y])
        self.n_init = self.init_x.size(0)
        self._dkl = DKL(self.init_x, self.init_y.squeeze(), n_iter=self.train_times, low_dim=self.low_dim,  pretrained_nn=self.ae, exact_gp=self.exact_gp)
        # settings to print NLL
        self._dkl.train_model(record_mae=True, record_interval=1, verbose=self.verbose)
        self._dkl.model.eval()

    def select_from_menu_df(self, menu_df, target_df, predictor_cols, type_col,
        target_ab_ids=None, lincoeffs_target_means_by_ab=None, cost_dict=None,**kwargs):
        
        # 1. Combine menu df with target_df .. is this necessary? or do we just assume that 
        # we have that? need inputs for evaluate function
        start_rows, end_rows, rows, target_rows, pam30_penalty_values  = self.preprocess_menu_rows(menu_df, target_df, target_ab_ids=target_ab_ids)

        # 2 compute / or recieve pam30 or other annotations to the sequence
        pred_x = torch.tensor(
        pd.concat(rows, axis=0, sort=False)[
            predictor_cols].values,
        dtype=torch.float
        )
        pred_i = torch.tensor(
            pd.concat(rows, axis=0, sort=False)[
                type_col].values.astype(np.double),
            dtype=torch.long
        )
        # TODO: DEFINE COST HERE?
        # TODO: k = 1 always since batching isn't done here
        # min_cost_rows = np.repeat(1, len(joint_mvn_mean))
        sel_idx, pred_mean, pred_std, covar = self.select(pred_x, pred_i, 1, **kwargs)
        del covar # we don't need this. though saving it could be a good sanity check
        # 4. add penalties # TODO: CANNOT ADD PEN
        # for i, p30i in enumerate(pam30_penalty_values):
        #     if not np.isnan(p30i):
        #         scores[i] = scores[i] + p30i
        fake_scores = np.ones(shape=len(pred_mean))
        fake_scores[sel_idx] = -100 # NOTE: subtract initialization points fromm selected index
        return fake_scores, pred_mean, pred_std
        



# class LargeFeatureExtractor(torch.nn.Sequential):
#     def __init__(self, data_dim, low_dim, add_spectrum_norm):
        
#         super(LargeFeatureExtractor, self).__init__()
#         self.add_module('linear1', add_spectrum_norm(torch.nn.Linear(data_dim, 1000)))
#         self.add_module('relu1', torch.nn.ReLU())
#         self.add_module('linear2',  add_spectrum_norm(torch.nn.Linear(1000, 500)))
#         self.add_module('relu2', torch.nn.ReLU())
#         self.add_module('linear3',  add_spectrum_norm(torch.nn.Linear(500, 50)))
#         # test if using higher dimensions could be better
#         if low_dim:
#             self.add_module('relu3', torch.nn.ReLU())
#             self.add_module('linear4',  add_spectrum_norm(torch.nn.Linear(50, 1)))
#         else:
#             self.add_module('relu3', torch.nn.ReLU())
#             self.add_module('linear4',  add_spectrum_norm(torch.nn.Linear(50, 10)))

# class GPRegressionModel(gpytorch.models.ExactGP):
#         def __init__(self, train_x, train_y, gp_likelihood, gp_feature_extractor, low_dim=True):
#             super(GPRegressionModel, self).__init__(train_x, train_y, gp_likelihood)
#             self.feature_extractor = gp_feature_extractor
#             self.mean_module = gpytorch.means.ConstantMean(constant_prior=train_y.mean())
#             if low_dim:
#                 self.covar_module = gpytorch.kernels.GridInterpolationKernel(
#                     gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=1), 
#                     outputscale_constraint=gpytorch.constraints.Interval(0.7,1.0)),
#                     num_dims=1, grid_size=100)
#             else:
#                 self.covar_module = gpytorch.kernels.LinearKernel(num_dims=10)

#             # This module will scale the NN features so that they're nice values
#             self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

#         def forward(self, x):
#             # We're first putting our data through a deep net (feature extractor)
#             self.projected_x = self.feature_extractor(x)
#             self.projected_x = self.scale_to_bounds(self.projected_x)  # Make the NN values "nice"

#             mean_x = self.mean_module(self.projected_x)
#             covar_x = self.covar_module(self.projected_x)
#             return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# class ExactGPRegressionModel(gpytorch.models.ExactGP):
#         def __init__(self, train_x, train_y, gp_likelihood, gp_feature_extractor, low_dim=True):
#             '''
#             Exact GP:
#             Leave placeholder for gp_feature_extractor
#             '''
#             super(ExactGPRegressionModel, self).__init__(train_x, train_y, gp_likelihood)
#             # self.mean_module = gpytorch.means.ZeroMean()
#             if low_dim:
#                 self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
#             else:
#                 self.covar_module = gpytorch.kernels.LinearKernel(num_dims=train_x.size(-1))
#             self.mean_module = gpytorch.means.ConstantMean(constant_prior=train_y.mean())

#             # This module will scale the NN features so that they're nice values
#             self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

#         def forward(self, x):
#             self.projected_x = x
#             # self.projected_x = self.scale_to_bounds(x)  # Make the values "nice"

#             mean_x = self.mean_module(self.projected_x)
#             covar_x = self.covar_module(self.projected_x)
#             return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# class DKL():

#     def __init__(self, train_x, train_y, n_iter=2, lr=1e-6, output_scale=.7, low_dim=False, 
#                  pretrained_nn=None, test_split=False, retrain_nn=True, spectrum_norm=False, exact_gp=False):
#         self.training_iterations = n_iter
#         self.lr = lr
#         self.train_x = train_x
#         self.train_y = train_y
#         self._y = train_y
#         self._x = train_x
#         self.data_dim = train_x.size(-1)
#         self.low_dim = low_dim
#         self.cuda = torch.cuda.is_available()
#         self.test_split = test_split
#         self.output_scale = output_scale
#         self.retrain_nn = retrain_nn
#         self.exact = exact_gp # exact GP overide
#         total_size = train_y.size(0)
#         if test_split:
#             test_ratio = 0.2
#             test_size = int(total_size * test_ratio)
#             shuffle_filter = np.random.choice(total_size, total_size, replace=False)
#             train_filter, test_filter = shuffle_filter[:-test_size], shuffle_filter[-test_size:]
#             self.train_x = train_x[train_filter]
#             self.train_y = train_y[train_filter]
#             self.test_x = train_x[test_filter]
#             self.test_y = train_y[test_filter]
#             self._x = self.test_x.clone()
#             self._y = self.test_y.clone()
#             self._x_train = self.train_x.clone()
#             self._y_train = self.train_y.clone()

#         def add_spectrum_norm(module, normalize=spectrum_norm):
#             if normalize:
#                 return torch.nn.utils.parametrizations.spectral_norm(module)
#             else:
#                 return module


#         self.feature_extractor = LargeFeatureExtractor(self.data_dim, self.low_dim, add_spectrum_norm=add_spectrum_norm)
#         if not (pretrained_nn is None):
#             self.feature_extractor.load_state_dict(pretrained_nn.encoder.state_dict(), strict=False)
#         self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

#         self.GPRegressionModel = GPRegressionModel if not self.exact else ExactGPRegressionModel
#         self.model = self.GPRegressionModel(self.train_x, self.train_y, self.likelihood, self.feature_extractor, self.low_dim)
#         if low_dim:
#             self.model.covar_module.base_kernel.outputscale = self.output_scale

#         if self.cuda:
#             self.model = self.model.cuda()
#             self.likelihood = self.likelihood.cuda()
#             self.train_x = self.train_x.cuda()
#             self.train_y = self.train_y.cuda()
    
#     # def train_model(self, loss_type="mse", verbose=False, **kwargs):
#     def train_model(self, loss_type="nll", verbose=False, **kwargs):
#         # Find optimal model hyperparameters
#         self.model.train()
#         self.likelihood.train()
#         record_mae = kwargs.get("record_mae", False) 
#         # Record MAE within a single training
#         if record_mae: 
#             record_interval = kwargs.get("record_interval", 1)
#             self.mae_record = np.zeros(self.training_iterations//record_interval)
#             self._train_mae_record = np.zeros(self.training_iterations//record_interval)

#         # Use the adam optimizer
#         if self.retrain_nn and not self.exact:
#             params = [
#                 {'params': self.model.feature_extractor.parameters()},
#                 {'params': self.model.covar_module.parameters()},
#                 {'params': self.model.mean_module.parameters()},
#                 {'params': self.model.likelihood.parameters()},
#             ]
#         else:
#             params = [
#                 {'params': self.model.covar_module.parameters()},
#                 {'params': self.model.mean_module.parameters()},
#                 {'params': self.model.likelihood.parameters()},
#             ]
#         # self.optimizer = torch.optim.Adam(params, lr=self.lr)
#         self.optimizer = SGLD(params, lr=self.lr)

#         # "Loss" for GPs - the marginal log likelihood
#         self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
#         if loss_type.lower() == "nll":
#             self.loss_func = lambda pred, y: -self.mll(pred, y)
#         elif loss_type.lower() == "mse":
#             tmp_loss = torch.nn.MSELoss()
#             self.loss_func = lambda pred, y: tmp_loss(pred.mean, y)
#         else:
#             raise NotImplementedError(f"{loss_type} not implemented")

#         def train(verbose=False):
#             iterator = tqdm.tqdm(range(self.training_iterations)) if verbose else range(self.training_iterations)
#             for i in iterator:
#                 # Zero backprop gradients
#                 self.optimizer.zero_grad()
#                 # Get output from model
#                 self.output = self.model(self.train_x)
#                 # Calc loss and backprop derivatives
#                 # print("loss", self.output.mean, self.train_y, self.loss_func)
#                 self.loss = self.loss_func(self.output, self.train_y)
#                 # print('backward')
#                 self.loss.backward()
#                 self.optimizer.step()
#                 # print("record")
#                 if record_mae and (i + 1) % record_interval == 0:
#                     self.mae_record[i//record_interval] = self.predict(self._x, self._y, verbose=False)
#                     if self.test_split:
#                         self._train_mae_record[i//record_interval] = self.predict(self._x_train, self._y_train, verbose=False)
#                     else:
#                         self._train_mae_record[i//record_interval] = self.mae_record[i//record_interval]
#                     if verbose:
#                         iterator.set_postfix({"NLL": self.loss.item(),
#                                         "Test MAE":self.mae_record[i//record_interval], 
#                                         "Train MAE": self._train_mae_record[i//record_interval] })
#                     self.model.train()
#                     self.likelihood.train()
#                 elif verbose:
#                     iterator.set_postfix(loss=self.loss.item())
#                 # iterator.set_description(f'ML (loss={self.loss:.4})')
#         train(verbose)

#     def predict(self, test_x, test_y, verbose=True, return_pred=False):
#         self.model.eval()
#         self.likelihood.eval()
#         if self.cuda:
#             test_x, test_y = test_x.cuda(), test_y.cuda()
#         with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
#             preds = self.model(test_x)
#         MAE = torch.mean(torch.abs(preds.mean - test_y))
#         if self.cuda:
#             MAE = MAE.cpu()


#         self.model.train()
#         self.likelihood.train()
#         if verbose:
#             print('Test MAE: {}'.format(MAE))
    
#         if return_pred:
#             return MAE, preds.mean.cpu() if self.cuda else preds.mean
#         else:
#             return MAE
    
#     def pure_predict(self, test_x,  return_pred=True):
#         '''
#         Different from predict, only predict value at given test_x and return the predicted mean by default
#         '''
#         self.model.eval()
#         self.likelihood.eval()
#         with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
#             preds = self.model(test_x)
#         self.model.train()
#         self.likelihood.train()

#         if return_pred:
#             return preds.mean.cpu()

#     def latent_space(self, input_x):
#         if self.exact:
#             return input_x.cpu() if self.cuda else input_x
#         if eval:
#             self.model.eval()
#             self.likelihood.eval()
#         with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
#             latent_x = self.model.feature_extractor(input_x)
#         return latent_x.cpu() if self.cuda else latent_x

#     def train_model_kneighbor_collision(self, n_neighbors:int=10, Lambda=1e-1, rho=1e-4, 
#                                         regularize=True, dynamic_weight=False, return_record=False, verbose=False):
#         # Find optimal model hyperparameters
#         torch.autograd.set_detect_anomaly(True)
#         self.model.train()
#         self.likelihood.train()
#         self.NNeighbors = NearestNeighbors(n_neighbors=n_neighbors)
#         self.softmax = torch.nn.Softmax(dim=1)
#         self.penalty_record = torch.zeros(self.training_iterations)
#         self.mse_record = torch.zeros(self.training_iterations)
#         self.nll_record = torch.zeros(self.training_iterations)

#         # Use the adam optimizer

#         if self.exact:
#             _parameters = [
#                 {'params': self.model.covar_module.parameters()},
#                 {'params': self.model.mean_module.parameters()},
#                 {'params': self.model.likelihood.parameters()},
#             ]
#         else:
#             _parameters = [
#                 {'params': self.model.feature_extractor.parameters()},
#                 {'params': self.model.covar_module.parameters()},
#                 {'params': self.model.mean_module.parameters()},
#                 {'params': self.model.likelihood.parameters()},
#             ]
#         self.optimizer = torch.optim.Adam(_parameters, lr=self.lr)

#         # "Loss" for GPs - the marginal log likelihood
#         self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
#         self.penalty = 0

#         def train():
#             iterator = tqdm.tqdm(range(self.training_iterations)) if verbose else range(self.training_iterations)
#             for i in iterator:
#                 # Zero backprop gradients
#                 self.optimizer.zero_grad()
                
#                 # Get output from model
#                 self.output = self.model(self.train_x)


#                 # collision penalty
#                 self.latent_variable = self.model.projected_x
#                 latent_variable = self.latent_variable.cpu() if self.cuda else self.latent_variable
#                 zero_var = torch.zeros([1,1]).cuda() if self.cuda else torch.zeros([1,1])
#                 self.NNeighbors.fit(latent_variable.detach().numpy())
#                 tmp_penalty = torch.zeros(self.latent_variable.size(0)).cuda() if self.cuda else torch.zeros(self.latent_variable.size(0))
#                 # tmp_weights = torch.zeros(self.latent_variable.size(0)).cuda() if self.cuda else torch.zeros(self.latent_variable.size(0))
#                 tmp_weights = self.train_y.detach()
#                 # for label, var in zip(self.train_y, self.latent_variable):
#                 for var_id, label in enumerate(self.train_y):
#                     var_val = self.latent_variable[var_id].cpu() if self.cuda else self.latent_variable[var_id]
#                     # tmp_weights[var_id] = label.detach().cpu().numpy() if self.cuda else label.detach().numpy()
#                     tmp_weights[var_id] = label.detach()
#                     neighbors_dists, neighbors_indices = self.NNeighbors.kneighbors(var_val.reshape([1,-1]).detach().numpy(), n_neighbors)
#                     # for nei_dist, nei_idx  in zip(neighbors_dists[0], neighbors_indices[0]):
#                     for nei_dist, nei_idx  in zip(neighbors_dists, neighbors_indices):
#                         # z_dist = torch.norm(torch.abs(var   - self.latent_variable[nei_idx])) # will cause double backward
#                         z_dist = np.linalg.norm(nei_dist)
#                         y_dist = torch.norm(torch.abs(label - self.train_y[nei_idx])) * Lambda
#                         assert torch.sum(self.train_y.cpu() - self._y)== 0
#                         tmp = torch.maximum(zero_var,  y_dist - z_dist) ** 2
#                         tmp_penalty[var_id] = tmp.reshape([1,1]).cuda() if self.cuda else tmp.reshape([1,1])   
#                         # self.penalty += tmp

#                 if dynamic_weight:
#                     self.penalty_weights = self.softmax(tmp_weights.reshape([1,-1]))
#                     tmp_penalty = torch.sum(self.penalty_weights * tmp_penalty)
#                 self.penalty = sum(tmp_penalty / self.output.variance)
#                 self.penalty = self.penalty * rho
                            

#                 # Calc loss and backprop derivatives
#                 nll = -self.mll(self.output, self.train_y)
#                 penalty = self.penalty
#                 if verbose:
#                     print(f"nll {nll}, penalty {penalty}")
#                 # loss
#                 if regularize:
#                     self.loss = nll + penalty
#                 else:
#                     self.loss = nll
#                 # self.loss = penalty
#                 self.loss.backward(retain_graph=True)
#                 if verbose:
#                     iterator.set_postfix(loss=self.loss.item())
#                 self.optimizer.step()
#                 # iterator.set_description(f'ML (loss={self.loss:.4})')

#                 # keep records
#                 if return_record:
#                     pred_mean = self.output.mean.cpu() if self.cuda else self.output.mean
#                     mse = torch.sum((pred_mean - self._y) ** 2)
#                     self.mse_record[i] = mse.detach()
#                     self.nll_record[i] = nll.detach()
#                     self.penalty_record[i] = penalty

#         # %time train()
#         train()
#         if return_record:
#             return self.penalty_record, self.mse_record, self.nll_record

#     def next_point(self, test_x, acq="ts", method="love", return_idx=False, beta=2):
#         """
#         Maximize acquisition function to find next point to query
#         """
#         # print("enter next point")
#         # clear cache
#         self.model.train()
#         self.likelihood.train()

#         # Set into eval mode
#         self.model.eval()
#         self.likelihood.eval()
#         test_x_selection = None
#         if self.exact and test_x.size(0) > 1000:
#             test_x_selection = np.random.choice(test_x.size(0), 1000)
#             test_x = test_x[test_x_selection]

#         if self.cuda:
#           test_x = test_x.cuda()

#         if acq.lower() in ["ts", 'qei']:
#             '''
#             Either Thompson Sampling or Monte-Carlo EI
#             '''
#             _num_sample = 100 if acq.lower() == 'qei' else 1 
#             if method.lower() == "love":
#                 with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.max_root_decomposition_size(200):
#                     # NEW FLAG FOR SAMPLING
#                     with gpytorch.settings.fast_pred_samples():
#                         # start_time = time.time()
#                         samples = self.model(test_x).rsample(torch.Size([_num_sample]))
#                         # fast_sample_time_no_cache = time.time() - start_time
#             elif method.lower() == "ciq":
#                 with torch.no_grad(), gpytorch.settings.ciq_samples(True), gpytorch.settings.num_contour_quadrature(10), gpytorch.settings.minres_tolerance(1e-4):
#                         # start_time = time.time()
#                         samples = self.likelihood(self.model(test_x)).rsample(torch.Size([_num_sample]))
#                         # fast_sample_time_no_cache = time.time() - start_time
#             else:
#                 raise NotImplementedError(f"sampling method {method} not implemented")
            
#             if acq.lower() == 'ts':
#                 self.acq_val = samples.T.squeeze()
#             elif acq.lower() == 'qei':
#                 # _best_y = self.train_y.max().squeeze()
#                 _best_y = samples.T.mean(dim=-1).max()
#                 self.acq_val = (samples.T - _best_y).clamp(min=0).mean(dim=-1)
#                 assert max(self.acq_val) > 0


#         elif acq.lower() in ["ucb", 'ci', 'lcb', 'rci']:
#             with torch.no_grad(), gpytorch.settings.fast_pred_var():
#                 observed_pred = self.likelihood(self.model(test_x))
#                 lower, upper = observed_pred.confidence_region()
#                 lower, upper = beta_CI(lower, upper, beta)

#             if acq.lower() == 'ucb':
#                 self.acq_val = upper
#             elif acq.lower() in ['ci', 'rci']:
#                 self.acq_val = upper - lower
#                 if acq.lower == 'rci':
#                     self.acq_val[upper < lower.max()] = self.acq_val.min()
#             elif acq.lower() == 'lcb':
#                 self.acq_val = -lower            
#         elif acq.lower() == 'truvar':
#             # print("enter next point truvar")
#             with torch.no_grad(), gpytorch.settings.fast_pred_var():
#                 observed_pred = self.likelihood(self.model(test_x))
#                 lower, upper = observed_pred.confidence_region()
#                 lower, upper = beta_CI(lower, upper, beta)
#                 pred_mean = (lower + upper) / 2

#             self.acq_val = torch.zeros(test_x.size(0))
#             for idx in range(test_x.size(0)):
#                 tmp_x = torch.cat([self.train_x, test_x[idx].reshape([1,-1])], dim=0)
#                 tmp_y = torch.cat([self.train_y, pred_mean[idx].reshape([1,])])
#                 _tmp_model = self.GPRegressionModel(tmp_x, tmp_y, self.likelihood, self.feature_extractor).eval()
#                 with torch.no_grad(), gpytorch.settings.fast_pred_var():
#                     _tmp_observed_pred = self.likelihood(_tmp_model(test_x))
#                     _tmp_lower, _tmp_upper = _tmp_observed_pred.confidence_region()
#                     _tmp_lower, _tmp_upper = beta_CI(_tmp_lower, _tmp_upper, beta)
#                 self.acq_val[idx] = torch.sum(_tmp_lower - lower) + torch.sum(upper - _tmp_upper)
#             # print("acq value", self.acq_val.size(), self.acq_val)
    

#         else:
#             raise NotImplementedError(f"acq {acq} not implemented")

#         max_pts = torch.argmax(self.acq_val)
#         candidate = test_x[max_pts]
#         if return_idx:
#             if test_x_selection is None:
#                 return max_pts
#             else:
#                 return test_x_selection[max_pts]
#         else:
#             return candidate

#     def CI(self, test_x,):
#         """
#         Return LCB and UCB
#         """

#         # clear cache
#         self.model.train()
#         self.likelihood.train()

#         # Set into eval mode
#         self.model.eval()
#         self.likelihood.eval()

#         if self.cuda:
#           test_x = test_x.cuda()

#         with torch.no_grad(), gpytorch.settings.fast_pred_var():
#             observed_pred = self.likelihood(self.model(test_x))
#             lower, upper = observed_pred.confidence_region()
        
#         return lower, upper

#     def intersect_CI_next_point(self, test_x, max_test_x_lcb, min_test_x_ucb, acq="ci", beta=2, return_idx=False):
#         """
#         Maximize acquisition function to find next point to query in the intersection of historical CI
#         """

#         # clear cache
#         self.model.train()
#         self.likelihood.train()

#         # Set into eval mode
#         self.model.eval()
#         self.likelihood.eval()

#         if self.cuda:
#           test_x = test_x.cuda()

#         if acq.lower() in ["ucb", 'ci', 'lcb', 'ucb-debug']:
#             with torch.no_grad(), gpytorch.settings.fast_pred_var():
#                 observed_pred = self.likelihood(self.model(test_x))
#                 lower, upper = observed_pred.confidence_region()
#                 # intersection
#                 lower, upper = beta_CI(lower, upper, beta)
#                 assert lower.shape == max_test_x_lcb.shape and upper.shape == min_test_x_ucb.shape
#                 lower = torch.max(lower.to("cpu"), max_test_x_lcb.to("cpu"))
#                 upper = torch.min(upper.to("cpu"), min_test_x_ucb.to("cpu"))
#             if acq.lower() == 'ucb-debug':
#                 self.acq_val = observed_pred.confidence_region()[1]

#             if acq.lower() == 'ucb':
#                 self.acq_val = upper
#             elif acq.lower() == 'ci':
#                 self.acq_val = upper - lower
#             elif acq.lower() == 'lcb':
#                 self.acq_val = -lower              
#         else:
#             raise NotImplementedError(f"acq {acq} not implemented")

#         max_pts = torch.argmax(self.acq_val)
#         candidate = test_x[max_pts]

#         # Store the intersection
#         self.max_test_x_lcb, self.min_test_x_ucb = lower, upper

#         if return_idx:
#             return max_pts
#         else:
#             return candidate

