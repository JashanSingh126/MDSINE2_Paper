'''Utility functions for the posterior
'''
import logging
import time
import collections
import itertools
import sys
import os
import h5py
import copy
import psutil
# import ray

import numpy as np
import numba
import scipy.sparse
# import torch
import numpy.random as npr
import scipy.stats
import scipy
import math
import random

import matplotlib.pyplot as plt

import pylab as pl

sys.path.append('..')
from names import STRNAMES, REPRNAMES

# @numba.jit(nopython=True, fastmath=True, cache=True)
def negbin_loglikelihood(k,m,dispersion):
    '''Loglikelihood - with parameterization in [1]

    Parameters
    ----------
    k : int
        Observed counts
    m : int
        Mean
    phi : float
        Dispersion

    Returns
    -------
    float
        Negative Binomial Log Likelihood

    References
    ----------
    [1] TE Gibson, GK Gerber. Robust and Scalable Models of Microbiome Dynamics. ICML (2018)
    '''
    r = 1/dispersion
    return math.lgamma(k+r) - math.lgamma(k+1) - math.lgamma(r) \
            + r * (math.log(r) - math.log(r+m)) + k * (math.log(m) - math.log(r+m))

@numba.jit(nopython=True, fastmath=True, cache=False)
def negbin_loglikelihood_MH_condensed(k,m,dispersion):
        '''
        Loglikelihood - with parameterization in [1] - but condensed (do not calculate stuff
        we do not have to)

        Parameters
        ----------
        k : int
            Observed counts
        m : int
            Mean
        phi : float
            Dispersion

        Returns
        -------
        float
            Negative Binomial Log Likelihood

        References
        ----------
        [1] TE Gibson, GK Gerber. Robust and Scalable Models of Microbiome Dynamics. ICML (2018)
        '''
        r = 1/dispersion
        rm = r+m
        return math.lgamma(k+r) - math.lgamma(r) \
            + r * (math.log(r) - math.log(rm)) + k * (math.log(m) - math.log(rm))

def negbin_loglikelihood_MH_condensed_not_fast(k,m,dispersion):
        '''
        Loglikelihood - with parameterization in [1] - but condensed (do not calculate stuff
        we do not have to). We use this function if `negbin_loglikelihood_MH_condensed` fails to
        compile, which can happen when doing jobs on the cluster

        Parameters
        ----------
        k : int
            Observed counts
        m : int
            Mean
        phi : float
            Dispersion

        Returns
        -------
        float
            Negative Binomial Log Likelihood

        References
        ----------
        [1] TE Gibson, GK Gerber. Robust and Scalable Models of Microbiome Dynamics. ICML (2018)
        '''
        r = 1/dispersion
        rm = r+m
        return math.lgamma(k+r) - math.lgamma(r) \
            + r * (math.log(r) - math.log(rm)) + k * (math.log(m) - math.log(rm))

def expected_n_clusters(G):
    '''Calculate the expected number of clusters given the number of ASVs

    Parameters
    ----------
    G : pl.Graph
        Graph object

    Returns
    -------
    int
        Expected number of clusters
    '''
    conc = G[STRNAMES.CONCENTRATION].prior.mean()
    return conc * np.log((G.data.n_asvs + conc) / conc)

def build_prior_covariance(G, cov, order, sparse=True, diag=False, cuda=False):
    '''Build basic prior covariance or precision for the variables
    specified in `order`

    Parameters
    ----------
    G : pylab.graph.Graph
        Graph to get the variables from
    cov : bool
        If True, build the covariance. If False, build the precision
    order : list(str)
        Which parameters to get the priors of
    sparse : bool
        If True, return as a sparse matrix
    diag : bool
        If True, returns the diagonal of the matrix. If this is True, it
        overwhelms the flag `sparse`
    cuda : bool
        If True, returns the array/matrix on the gpu (if there is one). Will not return 
        in sparse form - only dense.

    Returns
    -------
    arr : np.ndarray, scipy.sparse.dia_matrix, torch.DoubleTensor
        Prior covariance or precision matrix in either dense (np.ndarray) or
        sparse (scipy.sparse.dia_matrix) form
    '''
    n_asvs = G.data.n_asvs
    a = []
    for reprname in order:
        if reprname == REPRNAMES.GROWTH_VALUE:
            a.append(np.full(n_asvs, G[REPRNAMES.PRIOR_VAR_GROWTH].value))

        elif reprname == REPRNAMES.SELF_INTERACTION_VALUE:
            a.append(np.full(n_asvs, G[REPRNAMES.PRIOR_VAR_SELF_INTERACTIONS].value))

        elif reprname == REPRNAMES.CLUSTER_INTERACTION_VALUE:
            n_interactions = G[REPRNAMES.CLUSTER_INTERACTION_INDICATOR].num_pos_indicators
            a.append(np.full(n_interactions, G[REPRNAMES.PRIOR_VAR_INTERACTIONS].value))

        elif reprname == REPRNAMES.PERT_VALUE:
            for perturbation in G.perturbations:
                num_on = perturbation.indicator.num_on_clusters()
                a.append(np.full(
                    num_on,
                    perturbation.magnitude.prior.var.value))

        else:
            raise ValueError('reprname ({}) not recognized'.format(reprname))

    if len(a) == 1:
        arr = np.asarray(a[0])
    else:
        arr = np.asarray(list(itertools.chain.from_iterable(a)))
    if not cov:
        arr = 1/arr
    # if cuda:
    #     arr = torch.DoubleTensor(arr).to(_COMPUTE_DEVICE)
    if diag:
        return arr
    # if cuda:
    #     return torch.diag(arr)
    if sparse:
        return scipy.sparse.dia_matrix((arr,[0]), shape=(len(arr),len(arr))).tocsc()
    else:
        return np.diag(arr)

def build_prior_mean(G, order, shape=None, cuda=False):
    '''Builds the prior mean vector for all the variables in `order`.

    Parameters
    ----------
    G : pylab.grapg.Graph
        Graph to index the objects
    order : list
        list of objects to add the priors of. If the variable is the
        cluster interactions or cluster perturbations, then we assume the
        prior mean is a scalar and we set that value for every single value.
    shape : tuple, None
        Shape to cast the array into
    cuda : bool
        If True, returns the array/matrix on the gpu (if there is one)

    Returns
    -------
    np.ndarray, torch.DoubleTensor
    '''
    a = []
    for name in order:
        v = G[name]
        if v.id == REPRNAMES.GROWTH_VALUE:
            a.append(v.prior.mean.value * np.ones(G.data.n_asvs))
        elif v.id == REPRNAMES.SELF_INTERACTION_VALUE:
            a.append(v.prior.mean.value * np.ones(G.data.n_asvs))
        elif v.id == REPRNAMES.CLUSTER_INTERACTION_VALUE:
            a.append(
                np.full(
                    G[REPRNAMES.CLUSTER_INTERACTION_INDICATOR].num_pos_indicators,
                    v.prior.mean.value))
        elif v.id == REPRNAMES.PERT_VALUE:
            for perturbation in G.perturbations:
                a.append(np.full(
                    perturbation.indicator.num_on_clusters(),
                    perturbation.magnitude.prior.mean.value))
        else:
            raise ValueError('`name` ({}) not recognized'.format(name))
    if len(a) == 1:
        a = np.asarray(a[0])
    else:
        a = np.asarray(list(itertools.chain.from_iterable(a)))
    if shape is not None:
        a = a.reshape(*shape)
    # if cuda:
    #     a = torch.DoubleTensor(a).to(_COMPUTE_DEVICE)
    return a

def sample_categorical_log(log_p):
    '''Generate one sample from a categorical distribution with event
    probabilities provided in unnormalized log-space.

    Parameters
    ----------
    log_p : array_like
        logarithms of event probabilities, ***which need not be normalized***

    Returns
    -------
    int
        One sample from the categorical distribution, given as the index of that
        event from log_p.
    '''
    try:
        exp_sample = math.log(random.random())
        events = np.logaddexp.accumulate(np.hstack([[-np.inf], log_p]))
        events -= events[-1]
        return next(x[0]-1 for x in enumerate(events) if x[1] >= exp_sample)
    except:
        logging.critical('CRASHED IN `sample_categorical_log`:\nlog_p{}'.format(
            log_p))
        raise

def log_det(M, var):
    '''Computes pl.math.log_det but also saves the array if it crashes

    Parameters
    ----------
    M : nxn matrix (np.ndarray, scipy.sparse)
        Matrix to calculate the log determinant
    var : pl.variable.Variable subclass
        This is the variable that `log_det` was called from

    Returns
    -------
    np.ndarray
        Log determinant of matrix
    '''
    if scipy.sparse.issparse(M):
        M_ = np.zeros(shape=M.shape)
        M.toarray(out=M_)
        M = M_
    try:
        # if type(M) == torch.Tensor:
        #     return torch.inverse(M)
        # else:
        return pl.math.log_det(M)
    except:
        try:
            sample_iter = var.sample_iter
        except:
            sample_iter = None
        filename = 'crashes/logdet_error_iter{}_var{}pinv_{}.npy'.format(
            sample_iter, var.name, var.G.name)
        logging.critical('\n\n\n\n\n\n\n\nSaved array at "{}" - now crashing\n\n\n'.format(
                filename))
        os.makedirs('crashes/', exist_ok=True)
        np.save(filename, M)
        raise

def pinv(M, var):
    '''Computes np.linalg.pinv but it also saves the array that crashed it if
    it crashes.

    Parameters
    ----------
    M : nxn matrix (np.ndarray, scipy.sparse)
        Matrix to invert
    var : pl.variable.Variable subclass
        This is the variable that `pinv` was called from

    Returns
    -------
    np.ndarray
        Inverse of the matrix
    '''
    if scipy.sparse.issparse(M):
        M_ = np.zeros(shape=M.shape)
        M.toarray(out=M_)
        M = M_
    try:
        # if type(M) == torch.Tensor:
        #     return torch.inverse(M)
        # else:
        try:
            return np.linalg.pinv(M)
        except:
            try:
                return scipy.linalg.pinv(M)
            except:
                return scipy.linalg.inv(M)
    except:
        try:
            sample_iter = var.sample_iter
        except:
            sample_iter = None
        filename = 'crashes/pinv_error_iter{}_var{}pinv_{}.npy'.format(
            sample_iter, var.name, var.G.name)
        logging.critical('\n\n\n\n\n\n\n\nSaved array at "{}" - now crashing\n\n\n'.format(
                filename))
        os.makedirs('crashes/', exist_ok=True)
        np.save(filename, M)
        raise

# @numba.jit(nopython=True, fastmath=True, cache=True)
def prod_gaussians(means, variances):
    '''Product of Gaussians

    $\mu = [\mu_1, \mu_2, ..., \mu_n]$
    $\var = [\var_1, \var_2, ..., \var_3]$

    Means and variances must be in the same order.

    Parameters
    ----------
    means : np.ndarray
        All of the means
    variances : np.ndarray
        All of the means
    '''
    mu = means[0]
    var = variances[0]
    for i in range(1,len(means)):
        mu, var = _calc_params(mu1=mu, mu2=means[i], var1=var, var2=variances[i])
    return mu, var

# @numba.jit(nopython=True, fastmath=True, cache=True)
def _calc_params(mu1, mu2, var1, var2):
    v = var1+var2
    mu = ((var1*mu2) + (var2*mu1))/(v)
    var = (var1*var2)/v
    return mu,var

