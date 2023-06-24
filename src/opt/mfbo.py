import gpytorch
import os
import random
import torch
import tqdm

import numpy as np
import matplotlib.pyplot as plt
from types import Any, List

from ..models import DKL
from sklearn.preprocessing import StandardScaler, RobustScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DK_BO_AE_MF():
    """
    Initialize the network with auto-encoder, multi-fidelity version of DKBO
    """
    def __init__(self, 
                train_x:torch.Tensor,
                train_y:torch.Tensor, 
                fid_cost:torch.Tensor,
                n_init:int=10, 
                lr:float=1e-6, 
                train_iter:int=10, 
                spectrum_norm:bool=False,
                verbose:bool=False, 
                robust_scaling:bool=True, 
                pretrained_nn:torch.nn.Sequential=None, 
                low_dim:bool=True,
                retrain_nn:bool=True,
                exact_gp:bool=False,
                noisy_observation:bool=False,
                noise_constraint=None,
                **kwargs:Any)->None:
        '''
        Init the DK_BO_AE_MF
        Input:
            @train_x: all discretization of search space, with the last dimension as fidelity number
            @train_y: all corresponding labels
            @fid_cost: fidelity cost
            @n_init: number of initial data points for GP training
            @lr: learning rate
            @spectrum_norm: if add spectrum normalization to the feature extractor
            @verbose: if printing verbose information
            @robust_scaling: if applying robust scaling to the search space
            @pretrained_nn: if initialize the feature extractor with pretrained encoder
            @low_dim: if using low_dim version of the Deep Kernel
            @retrain_nn: if retrain the neural network during the periodical retrain of the kernel
            @exact_gp: if using exact_gp or deep kernel
            @noise_constraints: constrain the noise of the likelihood
        Return:
            None
        '''
        # scale input
        ScalerClass = RobustScaler if robust_scaling else StandardScaler
        self.scaler = ScalerClass().fit(train_x)
        train_x = self.scaler.transform(train_x)
        # init vars
        self.lr = lr
        self.low_dim = low_dim
        self.verbose = verbose
        self.n_init = n_init
        self.n_neighbors = min(self.n_init, 10)
        self.Lambda = 1
        self.train_x = torch.from_numpy(train_x).float()
        self.train_y = train_y
        self.fid_cost = fid_cost
        self.train_iter = train_iter
        self.retrain_nn = retrain_nn
        self.maximum = torch.max(self.train_y)
        self.spectrum_norm = spectrum_norm
        self.exact = exact_gp # exact GP overide
        self.noise_constraint = noise_constraint
        self.observed = torch.zeros(self.train_x.size(0))
        self.pretrained_nn = pretrained_nn
        self._noisy_observation = noisy_observation
        self.init_x = kwargs.get("init_x", self.train_x[:n_init])
        self.init_y = kwargs.get("init_y", self.train_y[:n_init])
        if self.noise_constraint: # perturbed with normal noise
            self.init_y = torch.normal(mean=self.init_y, std=torch.ones(self.init_y.size(0)))
        self._model = DKL(self.init_x, self.init_y.squeeze(), 
                          n_iter=self.train_iter, lr= self.lr, 
                          low_dim=self.low_dim, pretrained_nn=self.pretrained_nn, 
                          retrain_nn=retrain_nn, spectrum_norm=spectrum_norm, 
                          exact_gp=exact_gp, noise_constraint=self.noise_constraint)

        self.cuda = torch.cuda.is_available()
        self.train()
    
    def train(self, verbose:bool=False)->None:
        '''
        Train the kernel
        '''
        self._model.train_model(verbose=verbose)

        
    def query(self, n_iter:int=10, acq:str="ts", retrain_interval:int=1, **kwargs:Any)->torch.Tensor:
        '''
        Query the oracle
        '''
        self.regret = np.zeros(n_iter)
        self.interval = np.zeros(n_iter)

        if_tqdm = kwargs.get("if_tqdm", False)
        early_stop = kwargs.get("early_stop", True)
        iterator = tqdm.tqdm(range(n_iter)) if if_tqdm else range(n_iter)
        ci_intersection = kwargs.get("ci_intersection", False)
        max_test_x_lcb = kwargs.get("max_test_x_lcb", None)
        min_test_x_ucb = kwargs.get("min_test_x_ucb", None)
        beta = kwargs.get("beta", 1)
        _delta = kwargs.get("delta", .2)
        real_beta = beta <= 0 # if using analytic beta
        _candidate_idx_list = np.zeros(n_iter)
        _low_fid_budget_ratio 

        for i in iterator:

            test_x = self.train_x
            test_y = self.train_y
            util_array = torch.arange(self.train_x.size(0))
            if real_beta:
                beta = (2 * np.log((self.train_x.shape[0] * (np.pi * (self.init_x.shape[0] + 1)) ** 2) /(6 * _delta))) ** 0.5
            # Calculate Acquisition
            if ci_intersection:
                assert not( max_test_x_lcb is None or min_test_x_ucb is None)
                candidate_idx = self._model.intersect_CI_next_point(test_x, 
                    max_test_x_lcb=max_test_x_lcb, min_test_x_ucb=min_test_x_ucb, acq=acq, beta=beta, return_idx=True)
            else:
                candidate_idx = self._model.next_point(test_x, acq, "love", return_idx=True, beta=beta,)

            _candidate_idx_list[i] = candidate_idx
            self.init_x = torch.cat([self.init_x, test_x[candidate_idx].reshape(1,-1)], dim=0)
            new_y = test_y[candidate_idx] if not self._noisy_observation else torch.normal(mean=test_y[candidate_idx], std=torch.ones(1))
            self.init_y = torch.cat([self.init_y, new_y.reshape(1,-1)])
            self.observed[util_array[candidate_idx]] = 1

            # retrain
            if i % retrain_interval != 0 and self.low_dim: # allow skipping retrain in low-dim setting
                self._state_dict_record = self._model.feature_extractor.state_dict()
                self._output_scale_record = self._model.model.covar_module.base_kernel.outputscale
                self._length_scale_record = self._model.model.covar_module.base_kernel.base_kernel.lengthscale
            
            self._model = DKL(self.init_x, self.init_y.squeeze(),
                              n_iter=self.train_iter, lr= self.lr,
                              low_dim=self.low_dim, pretrained_nn=self.pretrained_nn, 
                              retrain_nn=self.retrain_nn, spectrum_norm=self.spectrum_norm,
                              exact_gp=self.exact, noise_constraint=self.noise_constraint)
            
            if i % retrain_interval != 0 and self.low_dim:
                self._model.feature_extractor.load_state_dict(self._state_dict_record, strict=False)
                self._model.model.covar_module.base_kernel.outputscale = self._output_scale_record
                self._model.model.covar_module.base_kernel.base_kernel.lengthscale = self._length_scale_record
            else:
                self.train()

            # regret & interval
            self.regret[i] = self.maximum - torch.max(self.init_y)
            if self.regret[i] < 1e-10 and early_stop:
                break
            if if_tqdm:
                iterator.set_postfix({"regret":self.regret[i], "Internal_beta": beta})

            _lcb, _ucb = self._model.CI(self.train_x)
            self.interval[i] = (_ucb.max() -_lcb.max()).numpy()
