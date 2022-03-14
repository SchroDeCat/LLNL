"""
Utilities to support partition BO
"""

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

def LinearClustering(X, y, n_clusters=2, n_iter=-1):
    '''
    Implementation of 
    [Spath, H. Correction to algorithm 39: clusterwise linear ̈regression. Computing 1981](https://link.springer.com/content/pdf/10.1007/BF02265317.pdf)
    Improve the linear regression by partitioning.
    Input: 
    - X: [n * dim] embeddings
    - y: [n] lables
    - n_cluster: number of clusters
    - n_iters: number of iterations, if -1 --> iterate until converge
    Return:
    - total loss
    - cluster id: [n]
    '''
    
    THRESHOLD = 1

    def minimize_mse_one_step(X, y, cluster_id, n_clusters):
        """
        Minimize the MSE for one step, return minimized MSE, new cluster id
        """
        ## init reg models
        reg_models = [None for _ in range(n_clusters)]
        for idx in range(n_clusters):
            cluster_filter = cluster_id == idx
            reg_models[idx] = LinearRegression().fit(X[cluster_filter], y[cluster_filter])
        
        ## improve the clustering
        total_losses = np.zeros(X.shape[0])
        for idx, point in enumerate(X):
            # tmp_model = reg_models[0]
            # tmp_point = point[np.newaxis, :]
            # tmp_pred = tmp_model.predict(tmp_point)
            losses = np.array([ (y[idx] - reg_models[model_id].predict(point[np.newaxis,:]).squeeze() )**2 for model_id in range(n_clusters)])
            cluster_id[idx] = np.argmin(losses)
            total_losses[idx] = np.min(losses)
        
        return np.mean(total_losses), cluster_id


    # initialize with a kmeans
    cluster_id = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(X)

    # iterate to improve the regression
    if n_iter == -1:
        loss = np.float("inf")
        while(True):
            next_loss, cluster_id = minimize_mse_one_step(X, y, cluster_id, n_clusters)
            if np.abs(next_loss - loss) < THRESHOLD:
                return cluster_id
            else:
                loss = next_loss
    else:
        for iter in range(n_iter):
            loss, cluster_id = minimize_mse_one_step(X, y, cluster_id, n_clusters)
        return cluster_id

    # ensure return
    return cluster_id


def KernelRidgeClustering(X, y, n_clusters=2, n_iter=-1):
    '''
    Implementation of 
    [Spath, H. Correction to algorithm 39: clusterwise linear ̈regression. Computing 1981](https://link.springer.com/content/pdf/10.1007/BF02265317.pdf)
    Improve the linear regression by partitioning.
    Input: 
    - X: [n * dim] embeddings
    - y: [n] lables
    - n_cluster: number of clusters
    - n_iters: number of iterations, if -1 --> iterate until converge
    Return:
    - total loss
    - cluster id: [n]
    '''
    
    THRESHOLD = 1

    def minimize_mse_one_step(X, y, cluster_id, n_clusters):
        """
        Minimize the MSE for one step, return minimized MSE, new cluster id
        """
        ## init reg models
        reg_models = [None for _ in range(n_clusters)]
        for idx in range(n_clusters):
            cluster_filter = cluster_id == idx
            reg_models[idx] = KernelRidge().fit(X[cluster_filter], y[cluster_filter])
        
        ## improve the clustering
        total_losses = np.zeros(X.shape[0])
        for idx, point in enumerate(X):
            # tmp_model = reg_models[0]
            # tmp_point = point[np.newaxis, :]
            # tmp_pred = tmp_model.predict(tmp_point)
            losses = np.array([ (y[idx] - reg_models[model_id].predict(point[np.newaxis,:]).squeeze() )**2 for model_id in range(n_clusters)])
            cluster_id[idx] = np.argmin(losses)
            total_losses[idx] = np.min(losses)
        
        return np.mean(total_losses), cluster_id


    # initialize with a kmeans
    cluster_id = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(X)

    # iterate to improve the regression
    if n_iter == -1:
        loss = np.float("inf")
        while(True):
            next_loss, cluster_id = minimize_mse_one_step(X, y, cluster_id, n_clusters)
            if np.abs(next_loss - loss) < THRESHOLD:
                return cluster_id
            else:
                loss = next_loss
    else:
        for iter in range(n_iter):
            loss, cluster_id = minimize_mse_one_step(X, y, cluster_id, n_clusters)
        return cluster_id

    # ensure return
    return cluster_id
