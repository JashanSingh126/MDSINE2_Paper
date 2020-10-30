import logging
import numpy as np
import sys
import os
import os.path
import pickle
import pandas
import argparse

import matplotlib.pyplot as plt
import seaborn

import pylab as pl
import model
import synthetic
from names import STRNAMES, REPRNAMES
import preprocess_filtering as filtering
import data
import base
import config


class MLCRRWorker(pl.multiprocessing.PersistentWorker):

    def __init__(self, Xs, ys, G, h, n_asvs, n_perts, total):
        self.Xs = Xs
        self.ys = ys
        self.G = G
        self.h = h
        self.n_asvs = n_asvs
        self.n_perts = n_perts
        self.total = total

    def sim(self, gp, ip, pp, leave_out,i):
        '''Set the penalty for each of the parameters and specify which replicate to
        leave out

        Parameters
        ----------
        gp : float
            growth penalty
        ip : float
            interaction penalty (includes self-interactions)
        pp : float
            perturbation penalty
        leave_out : int
            Which replicate index to leave out

        Returns
        -------
        (leave_out, (gp, ip, pp), loss)
        '''

        try:
            iiii = i

            Xs = None
            ys = None
            if len(self.Xs) == 1:
                X = self.Xs[0]
                y = self.ys[0]
            else:
                for i in range(len(self.Xs)):
                    if i == leave_out:
                        continue
                    if Xs is None:
                        Xs = (self.Xs[i],)
                        ys = (self.ys[i],)
                    else:
                        Xs += (self.Xs[i],)
                        ys += (self.ys[i],)

                y = np.vstack(ys)
                X = np.vstack(Xs)
            
            D = np.append(np.full(self.n_asvs, gp), np.full(self.n_asvs**2, ip))
            if self.n_perts is not None:
                D = np.append(D, np.full(self.n_asvs*self.n_perts, pp))
            D = np.diag(D)
            
            theta = pl.inference.MLRR.run_MLCRR(X=X, y=y, D=D, G=self.G, 
                h=self.h, A=None, b=None).reshape(-1,1)

            loss = y - X @ theta
            rmse_loss = np.sqrt(np.sum(np.square(loss)))
            print('{}/{}: {:.3E}'.format(iiii, self.total, rmse_loss))

        except Exception as e:
            print(e)
            rmse_loss = np.nan

        return (gp, ip, pp, leave_out, rmse_loss)

def run(params, graph_name, subjset):
    '''Run MLCRR
    '''
    # Initialize graph and data
    if pl.isstr(params):
        params = config.MLCRRConfig.load(params)
    if pl.isstr(subjset):
        subjset = pl.base.SubjectSet.load(subjset)
    
    GRAPH = pl.graph.Graph(name=graph_name, seed=params.INIT_SEED)
    GRAPH.as_default()

    d = data.Data(asvs=subjset.asvs, subjects=subjset, 
        min_rel_abund=False, data_logscale=True, G=GRAPH, add_eps_to_data=True)
    n_asvs = len(subjset.asvs)

    # Initialize the variables
    growth = pl.variables.Variable(
        name=STRNAMES.GROWTH_VALUE, G=GRAPH, shape=(n_asvs,))
    self_interactions = pl.variables.Variable(
        name=STRNAMES.SELF_INTERACTION_VALUE, shape=(n_asvs, ), G=GRAPH)
    interactions = pl.variables.Variable(name=STRNAMES.CLUSTER_INTERACTION_VALUE, 
        shape=(n_asvs * (n_asvs - 1), ), G=GRAPH)
    if subjset.perturbations is not None:
        for pidx, subj_pert in enumerate(subjset.perturbations):
            perturbation = pl.contrib.Perturbation(start=subj_pert.start, 
                end=subj_pert.end, asvs=subjset.asvs, G=GRAPH, name=subj_pert.name)
        perturbations = pl.variables.Variable(name=STRNAMES.PERT_VALUE, 
            shape=(len(subjset.perturbations)*n_asvs, ), G=GRAPH)

    # Set up the name spaces
    REPRNAMES.set(G=GRAPH)

    # Set up inference and theta parameter order
    mlcrr = pl.inference.MLRR(constrain=True, graph=GRAPH)
    order = [
        STRNAMES.GROWTH_VALUE,
        STRNAMES.SELF_INTERACTION_VALUE,
        STRNAMES.CLUSTER_INTERACTION_VALUE]
    if subjset.perturbations is not None:
        order.append(STRNAMES.PERT_VALUE)
    mlcrr.set_parameter_order(order=order)

    # Set up the constraints
    # We want the constraints so that growth is always positive and self-interactions
    # are always positive. interactions and perturbations are unbounded
    diag = np.append(np.full(2*n_asvs, -1), np.ones(n_asvs*(n_asvs-1)))
    if subjset.perturbations is not None:
        diag = np.append(diag, np.ones(n_asvs * len(subjset.perturbations)))
    G = np.diag(diag)
    h = np.append(
        np.zeros(2*n_asvs), 
        np.full(G.shape[0]-2*n_asvs, float('inf'))).reshape(-1,1)
    mlcrr.set_constraints(G=G, h=h)

    # Set up penalties
    mlcrr.set_penalty(parameter=growth.id, penalty=1e-3)
    mlcrr.set_penalty(parameter=self_interactions.id, penalty=1e-3)
    mlcrr.set_penalty(parameter=interactions.id, penalty=1e-3)
    if subjset.perturbations is not None:
        mlcrr.set_penalty(parameter=perturbations.id, penalty=1e-3)

    # Set up the design matrices
    growthDM = data.GrowthDesignMatrix(G=GRAPH, name='growth_design_matrix', 
        data_logscale=True, perturbations_additive=True)
    selfinteractionsDM = data.SelfInteractionDesignMatrix(G=GRAPH,
        name='self_interactions_design_matrix',
        data_logscale=True)
    interactionsDM = data.InteractionsBaseDesignMatrix(G=GRAPH, data_logscale=True,
        name=STRNAMES.CLUSTER_INTERACTION_VALUE)
    if subjset.perturbations is not None:
        perturbationsDM = data.PerturbationBaseDesignMatrix(data_logscale=True, additive=True, 
            name=STRNAMES.PERT_VALUE, G=GRAPH)
        perturbationsDM.build()

    growthDM.build()
    selfinteractionsDM.build()
    interactionsDM.build()

    # Set up observation matrices
    lhs = data.LHSVector(G=GRAPH, data_logscale=True, name='lhs_vector')
    lhs.build()

    y = d.construct_lhs()
    rhs = [
        REPRNAMES.GROWTH_VALUE, 
        REPRNAMES.SELF_INTERACTION_VALUE,
        REPRNAMES.CLUSTER_INTERACTION_VALUE]
    if subjset.perturbations is not None:
        rhs.append(REPRNAMES.PERT_VALUE)
    X = d.construct_rhs(keys=rhs, toarray=True)

    mlcrr.run_single(X=X, y=y)

    # Set interactions as a matrix
    value = np.zeros(shape=(n_asvs, n_asvs), dtype=float)
    iii = 0
    for i in range(n_asvs):
        for j in range(n_asvs):
            if i == j:
                value[i,j] = np.nan
                continue
            value[i,j] = interactions.value[iii]
            iii += 1
    interactions.value = value

    # Divide up the perturbations
    i = 0
    for perturbation in GRAPH.perturbations:
        perturbation.indicator.value[:] = True
        perturbation.magnitude.value = perturbations[i:i+n_asvs]
        i += n_asvs

    mlcrr.save(config.MLCRR_RESULTS_FILENAME)
    GRAPH.save(config.GRAPH_FILENAME)

    return mlcrr

def runCV(params, graph_name, subjset):
    '''Run MLCRR cross validation
    '''
    # Initialize graph and data
    if pl.isstr(params):
        params = config.MLCRRConfig.load(params)
    if pl.isstr(subjset):
        subjset = pl.base.SubjectSet.load(subjset)
    
    GRAPH = pl.graph.Graph(name=graph_name, seed=params.INIT_SEED)
    GRAPH.as_default()

    d = data.Data(asvs=subjset.asvs, subjects=subjset, 
        min_rel_abund=False, data_logscale=True, G=GRAPH)
    n_asvs = len(subjset.asvs)

    # Initialize the variables
    growth = pl.variables.Variable(
        name=STRNAMES.GROWTH_VALUE, G=GRAPH, shape=(n_asvs,))
    self_interactions = pl.variables.Variable(
        name=STRNAMES.SELF_INTERACTION_VALUE, shape=(n_asvs, ), G=GRAPH)
    interactions = pl.variables.Variable(name=STRNAMES.CLUSTER_INTERACTION_VALUE, 
        shape=(n_asvs * (n_asvs - 1), ), G=GRAPH)
    if subjset.perturbations is not None:
        for pidx, subj_pert in enumerate(subjset.perturbations):
            perturbation = pl.contrib.Perturbation(start=subj_pert.start, 
                end=subj_pert.end, asvs=subjset.asvs, G=GRAPH, name=subj_pert.name)
        perturbations = pl.variables.Variable(name=STRNAMES.PERT_VALUE, 
            shape=(len(subjset.perturbations)*n_asvs, ), G=GRAPH)

    # Set up the name spaces
    REPRNAMES.set(G=GRAPH)

    # Set up inference and theta parameter order
    mlcrr = pl.inference.MLRR(constrain=True, graph=GRAPH)
    order = [
        STRNAMES.GROWTH_VALUE,
        STRNAMES.SELF_INTERACTION_VALUE,
        STRNAMES.CLUSTER_INTERACTION_VALUE]
    if subjset.perturbations is not None:
        order.append(STRNAMES.PERT_VALUE)
    mlcrr.set_parameter_order(order=order)

    # Set up the constraints
    # We want the constraints so that growth is always positive and self-interactions
    # are always positive. interactions and perturbations are unbounded
    diag = np.append(np.full(2*n_asvs, -1), np.ones(n_asvs*(n_asvs-1)))
    if subjset.perturbations is not None:
        diag = np.append(diag, np.ones(n_asvs * len(subjset.perturbations)))
    G = np.diag(diag)
    h = np.append(
        np.zeros(2*n_asvs), 
        np.full(G.shape[0]-2*n_asvs, float('inf'))).reshape(-1,1)
    mlcrr.set_constraints(G=G, h=h)

    # Set up the design matrices
    growthDM = data.GrowthDesignMatrix(G=GRAPH, name='growth_design_matrix', 
        data_logscale=True, perturbations_additive=True)
    selfinteractionsDM = data.SelfInteractionDesignMatrix(G=GRAPH,
        name='self_interactions_design_matrix',
        data_logscale=True)
    interactionsDM = data.InteractionsBaseDesignMatrix(G=GRAPH, data_logscale=True,
        name=STRNAMES.CLUSTER_INTERACTION_VALUE)
    if subjset.perturbations is not None:
        perturbationsDM = data.PerturbationBaseDesignMatrix(data_logscale=True, 
            perturbations_additive=True, name=STRNAMES.PERT_VALUE, G=GRAPH)
        perturbationsDM.build()

    growthDM.build()
    selfinteractionsDM.build()
    interactionsDM.build()

    # Set up observation matrices
    lhs = data.LHSVector(G=GRAPH, data_logscale=True, name='lhs_vector')
    lhs.build()

    y = d.construct_lhs()
    rhs = [
        REPRNAMES.GROWTH_VALUE, 
        REPRNAMES.SELF_INTERACTION_VALUE,
        REPRNAMES.CLUSTER_INTERACTION_VALUE]
    if subjset.perturbations is not None:
        rhs.append(REPRNAMES.PERT_VALUE)
    X = d.construct_rhs(keys=rhs, toarray=True)

    # Index out each subject Xs
    Xs = []
    ys = []

    i = 0
    for ridx in range(d.n_replicates):
        n_dts = d.n_dts_for_replicate[ridx]
        l = n_dts * n_asvs
        X_temp = X[i:i+l,:]
        y_temp = y[i:i+l,:]

        rows_to_keep = []
        for rowidx in range(len(y_temp)):
            if np.all(np.isfinite(y_temp[rowidx])):
                rows_to_keep.append(rowidx)
        X_temp = X_temp[rows_to_keep, :]
        y_temp = y_temp[rows_to_keep, :]

        Xs.append(X_temp)
        ys.append(y_temp)
        i += l

    # Set up workers
    n_perts = None
    if subjset.perturbations is not None:
        n_perts = len(subjset.perturbations)
    
    # Set up penalty meshes:
    growth_penalty_mesh = np.logspace(start=params.CV_MAP_MIN, stop=params.CV_MAP_MAX, 
        num=params.CV_MAP_N)
    interaction_penalty_mesh = np.logspace(start=params.CV_MAP_MIN, stop=params.CV_MAP_MAX, 
        num=params.CV_MAP_N)

    total = params.CV_MAP_N * params.CV_MAP_N
    if subjset.perturbations is not None:
        perturbation_penalty_mesh = np.logspace(start=params.CV_MAP_MIN, stop=params.CV_MAP_MAX, 
            num=params.CV_MAP_N)
        total *= params.CV_MAP_N

    total *= d.n_replicates

    if params.N_CPUS is not None:
        pool = pl.multiprocessing.PersistentPool(ptype='dasw')
        for i in range(params.N_CPUS):
            pool.add_worker(MLCRRWorker(Xs=Xs, ys=ys, 
                G=G, h=h, n_asvs=n_asvs, n_perts=n_perts, total=total))
        pool.staged_map_start('sim')        

    # calculate
    reverse_index = {}
    iii = 0
    for i, gp in enumerate(growth_penalty_mesh):
        reverse_index[gp] = i
        for ip in interaction_penalty_mesh:
            for lo in range(d.n_replicates):

                if subjset.perturbations is not None:
                    for pp in perturbation_penalty_mesh:
                        kwargs = {'gp': gp, 'ip': ip, 'pp': pp, 'leave_out': lo, 'i': iii}
                        pool.staged_map_put(kwargs)
                        iii += 1
                else:
                    kwargs = {'gp': gp, 'ip': ip, 'pp': None, 'leave_out': lo, 'i': iii}
                    pool.staged_map_put(kwargs)
                    iii += 1

    ret = pool.staged_map_get()
    pool.kill()

    # Find the minimum - take the geometric mean of the leave outs
    if subjset.perturbations is not None:
        arr = np.ones(shape=(params.CV_MAP_N,params.CV_MAP_N,params.CV_MAP_N, d.n_replicates))*np.nan
        for (gp, ip, pp, iii, rmse_loss) in ret:
            i = reverse_index[gp]
            j = reverse_index[ip]
            k = reverse_index[pp]
            arr[i,j,k, iii] = rmse_loss
    else:
        arr = np.ones(shape=(params.CV_MAP_N,params.CV_MAP_N, d.n_replicates))*np.nan
        for (gp, ip, pp, iii, rmse_loss) in ret:
            i = reverse_index[gp]
            j = reverse_index[ip]
            arr[i,j, iii] = rmse_loss

    # Multiply over the last axis
    arr = np.nanprod(arr, axis=-1)

    # Get the index of the N dimensional arr of the lowest
    ind = np.unravel_index(np.nanargmin(arr, axis=None), arr.shape)
    gp = growth_penalty_mesh[ind[0]]
    ip = interaction_penalty_mesh[ind[1]]

    D = np.append(np.full(n_asvs, gp), np.full(n_asvs**2, ip))
    if subjset.perturbations is not None:
        pp = perturbation_penalty_mesh[ind[2]]
        D = np.append(D, np.full(n_perts*n_asvs, pp))

    # Get the parameters
    D = np.diag(D)
    y = np.vstack(ys)
    X = np.vstack(Xs)

    mlcrr.run_single(X=X, y=y, D=D)

    # Set interactions as a matrix
    value = np.zeros(shape=(n_asvs, n_asvs), dtype=float)
    iii = 0
    for i in range(n_asvs):
        for j in range(n_asvs):
            if i == j:
                value[i,j] = np.nan
                continue
            value[i,j] = interactions.value[iii]
            iii += 1
    interactions.value = value

    # Divide up the perturbations
    if subjset.perturbations is not None:
        i = 0
        for perturbation in GRAPH.perturbations:
            perturbation.indicator.value[:] = True
            perturbation.magnitude.value = perturbations[i:i+n_asvs]
            i += n_asvs

    mlcrr.save(config.MLCRR_RESULTS_FILENAME)
    GRAPH.save(config.GRAPH_FILENAME)
    return mlcrr
