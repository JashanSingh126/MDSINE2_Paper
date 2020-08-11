'''Functions to build and plot the model
'''
# Base
import logging
import random
import time
import sys
import os
import shutil
import h5py
import numba
import warnings
import pickle
import numpy as np
import pandas as pd
import json

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines

# Network
import networkx as nx
from py2cytoscape.util import from_networkx
import ete3

# Custom modules
import posterior
import data
import pylab as pl
import config
import synthetic
import model as model_module
import preprocess_filtering as filtering
import metrics
from names import STRNAMES, LATEXNAMES, REPRNAMES
import analyze_clusters

AUX_COLOR = 'green'
LATENT_COLOR = 'red'
MIN_TRAJ_COLOR = 'purple'

def run(params, graph_name, data_filename, tracer_filename, 
    mcmc_filename, graph_filename, hdf5_filename, 
    checkpoint_iter, crash_if_error, asv_strname='name',
    continue_inference=None):
    '''This is the method used to run the model specified by "Robust and Scalable
    Models of Microbiome Dynamics" by Gibson and Gerber (2018)

    Parameters
    ----------
    params : config.ModelConfig
        This class specifies all of the parameters of the model
    graph_name : str
        Name of the graph
    asv_strname : str
        This is the format of how to print the ASVs whenyou are printing the clusters
    data_filename : str
        This is the location where the SubjsetSet object with all the data is saved
    tracer_filename, mcmc_filename, graph_filename: str
        This is where we save the tracer, mcmc, and graph objects
    hdf5_filename : str
        This is where to save the hdf5 object of the trace
    checkpoint_iter : int
        How often to save the temparary trace
    continue_inference : int, None
        If this is not None, then we are continuing the chain at Gibb step
        `continue_inference` - else we are creating a new chain at Gibb step 0

    Returns
    -------
    pl.inference.BaseMCMC
        Inference chain
    '''
    # Type check
    if not config.isModelConfig(params):
        raise TypeError('`params` ({}) needs to be a config.ModelConfig object'.format(type(params)))
    if not np.all(pl.itercheck([data_filename, tracer_filename, hdf5_filename,
        graph_filename, mcmc_filename, asv_strname, graph_name], pl.isstr)):
        raise TypeError('Must be strs')
    if not pl.isint(checkpoint_iter):
        raise TypeError('Must be int ({})'.format(type(checkpoint_iter)))
    
    if continue_inference is not None:
        GRAPH = pl.graph.Graph.load(graph_filename)
    else: 
        GRAPH = pl.graph.Graph(name=graph_name, seed=params.INIT_SEED)
    GRAPH.as_default()

    # If we are continuing the chain, then we dont have to do the initialization, we can
    # just load the chain and then set the inference starting
    if continue_inference is not None:
        if not pl.isint(continue_inference):
            raise TypeError('`continue_inference` ({}) must be None or an int'.format(
                type(continue_inference)))
            if continue_inference < 0:
                raise ValueError('`continue_inference` ({}) must be  > 0'.format(continue_inference))
        REPRNAMES.set(G=GRAPH)
        logging.info('Continuing inference at Gibb step {}'.format(continue_inference))
        mcmc = pl.inference.BaseMCMC.load(mcmc_filename)
        mcmc.continue_inference(gibbs_step_start=continue_inference)

        # Set the items to the values they were at that gibbs step

        # Set cluster assignments
        cluster_assignments = mcmc.graph[REPRNAMES.CLUSTERING]
        clustering = mcmc.graph[REPRNAMES.CLUSTERING_OBJ]
        if params.LEARN[STRNAMES.CLUSTERING]:
            coclusters = clustering.coclusters.get_trace_from_disk(section='entire')
            toarray = pl.cluster.toarray_from_cocluster(coclusters[-1])
            clustering.from_array(toarray)
        # Else do nothing because it was fixed.

        # Set concentration
        concentration = mcmc.grpah[REPRNAMES.CONCENTRATION]
        if params.LEARN[STRNAMES.CONCENTRATION]:
            trace = concentration.get_trace_from_disk(section='entire')
            concentration.value = trace[-1]

        # Set interactions

        # Set perturbations

        # Set growth

        # Set self interactions

        # Set process variance

        # Set filtering

        # Build the design matrices from scratch (all of them)





        filtering = mcmc.graph[REPRNAMES.FILTERING]
        Z = mcmc.graph[REPRNAMES.CLUSTER_INTERACTION_INDICATOR]
        subjset = mcmc.graph.data.subjects

        return run_inference(mcmc=mcmc, crash_if_error=crash_if_error, 
            cluster_assignments=cluster_assignments, filtering=filtering, Z=Z,
            subjset=subjset, data_filename=data_filename)


    params.INITIALIZATION_KWARGS[STRNAMES.FILTERING]['h5py_filename'] = hdf5_filename

    ######################################################################
    # Read in data, assume that it has already been filtered, clustering
    # Set qpcr max value if necessary
    logging.info('Loading from subjset')
    subjset = pl.SubjectSet.load(filename=data_filename)
    asvs = subjset.asvs
    d = data.Data(asvs=subjset.asvs, subjects=subjset, 
        min_rel_abund=params.ADD_MIN_REL_ABUNDANCE,
        data_logscale=params.DATA_LOGSCALE, G=GRAPH)
    clustering = pl.cluster.Clustering(clusters=None, items=asvs, G=GRAPH, 
        name=STRNAMES.CLUSTERING_OBJ)

    ######################################################################
    # Instantiate variables
    # Interaction Value
    prior_var_interactions = posterior.PriorVarInteractions(
        prior=pl.variables.SICS(
            dof=pl.variables.Constant(None, G=GRAPH),
            scale=pl.variables.Constant(None, G=GRAPH),
            G=GRAPH), G=GRAPH)
    prior_mean_interactions = posterior.PriorMeanInteractions(
        prior=pl.variables.Normal(
            mean=pl.variables.Constant(None, G=GRAPH),
            var=pl.variables.Constant(None, G=GRAPH),
            G=GRAPH), G=GRAPH)
    interaction_value_prior = pl.variables.Normal(
        mean=prior_mean_interactions,
        var=prior_var_interactions, G=GRAPH)
    interaction_indicator_prior = pl.variables.Beta(
        a=pl.variables.Constant(None, G=GRAPH),
        b=pl.variables.Constant(None, G=GRAPH),
        G=GRAPH)
    pi_z = posterior.ClusterInteractionIndicatorProbability(
        prior=interaction_indicator_prior,G=GRAPH)
    interactions = posterior.ClusterInteractionValue(
        prior=interaction_value_prior,clustering=clustering, G=GRAPH,
        perturbations_additive=params.PERTURBATIONS_ADDITIVE)
    # Interaction indicators
    Z = posterior.ClusterInteractionIndicators(
        prior=pi_z, mp=params.MP_INDICATORS, G=GRAPH,
        relative=params.RELATIVE_LOG_MARGINAL_INDICATORS)
    
    # Growth
    prior_var_growth = posterior.PriorVarMH(
        prior=pl.variables.SICS(
            dof=pl.variables.Constant(None, G=GRAPH),
            scale=pl.variables.Constant(None, G=GRAPH),
            G=GRAPH),
        child_name=STRNAMES.GROWTH_VALUE,
        G=GRAPH)
    prior_mean_growth = posterior.PriorMeanMH(
        prior=pl.variables.TruncatedNormal(
            mean=pl.variables.Constant(None, G=GRAPH),
            var=pl.variables.Constant(None, G=GRAPH),
            G=GRAPH), child_name=STRNAMES.GROWTH_VALUE, G=GRAPH)
    prior_growth = pl.variables.Normal(
        mean=prior_mean_growth,
        var=prior_var_growth,
        name='prior_{}'.format(STRNAMES.GROWTH_VALUE),
        G=GRAPH)
    growth = posterior.Growth(prior=prior_growth, G=GRAPH, 
        perturbations_additive=params.PERTURBATIONS_ADDITIVE)

    # Self Interactions
    prior_var_self_interactions = posterior.PriorVarMH(
        prior=pl.variables.SICS(
            dof=pl.variables.Constant(None, G=GRAPH),
            scale=pl.variables.Constant(None, G=GRAPH),
            G=GRAPH),
        child_name=STRNAMES.SELF_INTERACTION_VALUE,
        G=GRAPH)
    prior_mean_self_interactions = posterior.PriorMeanMH(
        prior=pl.variables.TruncatedNormal(
            mean=pl.variables.Constant(None, G=GRAPH),
            var=pl.variables.Constant(None, G=GRAPH),
            G=GRAPH), child_name=STRNAMES.SELF_INTERACTION_VALUE, G=GRAPH)
    prior_self_interactions = pl.variables.Normal(
        mean=prior_mean_self_interactions,
        var=prior_var_self_interactions,
        name='prior_{}'.format(STRNAMES.SELF_INTERACTION_VALUE),
        G=GRAPH)
    self_interactions = posterior.SelfInteractions(
        perturbations_additive=params.PERTURBATIONS_ADDITIVE,
        prior=prior_self_interactions, G=GRAPH)

    # Process variance
    if params.DATA_LOGSCALE:
        prior_process_var = pl.variables.SICS( 
            dof=pl.variables.Constant(None, G=GRAPH),
            scale=pl.variables.Constant(None, G=GRAPH),
            G=GRAPH)
        process_var = posterior.ProcessVarGlobal(
            G=GRAPH, c_m=params.C_M, prior=prior_process_var,
            perturbations_additive=params.PERTURBATIONS_ADDITIVE)
    else:
        process_var = posterior.ProcessVarHeteroscedasticGlobal(G=GRAPH, c_m=params.C_M, 
            data_logscale=params.DATA_LOGSCALE)

    # Concentration
    prior_concentration = pl.variables.Gamma(
        shape=pl.variables.Constant(None, G=GRAPH),
        scale=pl.variables.Constant(None, G=GRAPH),
        G=GRAPH)
    concentration = posterior.Concentration(
        prior=prior_concentration,
        G=GRAPH)

    # clustering
    cluster_assignments = posterior.ClusterAssignments(
        clustering=clustering, concentration=concentration,
        G=GRAPH, mp=params.MP_CLUSTERING,
        relative=params.RELATIVE_LOG_MARGINAL_CLUSTERING)

    # Filtering and zero inflation
    if params.DATA_LOGSCALE:
        filtering = posterior.FilteringLogMP(G=GRAPH, mp=params.MP_FILTERING, 
            perturbations_additive=params.PERTURBATIONS_ADDITIVE)
    else:
        filtering = posterior.FilteringMP(G=GRAPH, mp=params.MP_FILTERING)
    zero_inflation = posterior.ZeroInflation(G=GRAPH, mp=params.MP_ZERO_INFLATION)

    # Perturbations - first initialize theperturbation objects and then the
    # regression aggregation objects
    if subjset.perturbations is not None:
        for pidx, subj_pert in enumerate(subjset.perturbations):
            pert_start = subj_pert.start
            pert_end = subj_pert.end
            if subj_pert.name is None:
                name = STRNAMES.PERTURBATIONS + str(pidx)
            else:
                name = subj_pert.name
            perturbation = pl.contrib.ClusterPerturbation(
                start=pert_start, end=pert_end, probability=pl.variables.Beta(
                    name=name + '_probability', G=GRAPH, value=None, a=None, b=None),
                clustering=clustering, G=GRAPH, name=name,
                signal_when_clusters_change=False, signal_when_item_assignment_changes=False)

            prior_var_magn = posterior.PriorVarPerturbationSingle(prior=pl.variables.SICS(
                dof=pl.variables.Constant(None, G=GRAPH),
                scale=pl.variables.Constant(None, G=GRAPH),
                G=GRAPH), perturbation=perturbation, G=GRAPH)
            prior_mean_magn = posterior.PriorMeanPerturbationSingle(
                prior=pl.variables.Normal(
                    mean=pl.variables.Constant(None, G=GRAPH),
                    var=pl.variables.Constant(None, G=GRAPH),
                    G=GRAPH), perturbation=perturbation, G=GRAPH)
            prior_magn = pl.variables.Normal(G=GRAPH,
                mean=prior_mean_magn,
                var=prior_var_magn)
            perturbation.magnitude.add_prior(prior_magn)

            prior_prob = pl.variables.Beta(
                a=pl.variables.Constant(None, G=GRAPH),
                b=pl.variables.Constant(None, G=GRAPH),
                G=GRAPH)
            perturbation.probability.add_prior(prior_prob)

        prior_var_magns = posterior.PriorVarPerturbations(G=GRAPH)
        prior_mean_magns = posterior.PriorMeanPerturbations(G=GRAPH)
        pert_values = posterior.PerturbationMagnitudes(G=GRAPH, 
            perturbations_additive=params.PERTURBATIONS_ADDITIVE)
        pert_ind = posterior.PerturbationIndicators(G=GRAPH,
            need_to_trace=False, relative=params.RELATIVE_LOG_MARGINAL_PERT_INDICATORS)
        pert_ind_prob = posterior.PerturbationProbabilities(G=GRAPH)
    else:
        pert_values = None
        pert_ind = None
        pert_ind_prob = None    

    beta = posterior.RegressCoeff(
        growth=growth,
        self_interactions=self_interactions,
        interactions=interactions,
        pert_mag=pert_values,
        perturbations_additive=params.PERTURBATIONS_ADDITIVE,
        G=GRAPH)

    # Set qPCR variance priors
    qpcr_variances = posterior.qPCRVariances(G=GRAPH, L=params.N_QPCR_BUCKETS)
    qpcr_dofs = posterior.qPCRDegsOfFreedoms(G=GRAPH, L=params.N_QPCR_BUCKETS)
    qpcr_scales = posterior.qPCRScales(G=GRAPH, L=params.N_QPCR_BUCKETS)

    # Set priors
    for l in range(params.N_QPCR_BUCKETS):
        dof = qpcr_dofs.value[l]
        scale = qpcr_scales.value[l]

        qpcr_scale_prior = pl.variables.SICS( 
            dof=pl.variables.Constant(None, G=GRAPH),
            scale=pl.variables.Constant(None, G=GRAPH),
            name='prior_' + STRNAMES.QPCR_SCALES + '_{}'.format(l),
            G=GRAPH)
        qpcr_dof_prior = pl.variables.Uniform(
            low=pl.variables.Constant(None, G=GRAPH),
            high=pl.variables.Constant(None, G=GRAPH),
            name='prior_' + STRNAMES.QPCR_DOFS + '_{}'.format(l),
            G=GRAPH)
        
        # add priors
        dof.add_prior(qpcr_dof_prior)
        scale.add_prior(qpcr_scale_prior)

    ######################################################################
    # Assign qPCR measurements to each variance bucket
    mean_log_measurements = []
    indices = []
    for ridx in range(d.n_replicates):
        for tidx,t in enumerate(d.given_timepoints[ridx]):
            mean_log_measurements.append(np.mean(d.qpcr[ridx][t].log_data))
            indices.append((ridx, tidx))

    idxs = np.argsort(mean_log_measurements)
    l_len = int(len(mean_log_measurements)/params.N_QPCR_BUCKETS)
    logging.info('There are {} qPCR measurements for {} buckets. Each bucket is' \
        ' {} measurements long'.format(len(indices), params.N_QPCR_BUCKETS, l_len))
    
    iii = 0
    for l in range(params.N_QPCR_BUCKETS):
        # If it is the last bucket, assign the rest of the elements to it
        if l == params.N_QPCR_BUCKETS - 1:
            l_len = len(mean_log_measurements) - iii
        for i in range(l_len):
            idx = idxs[iii]
            ridx,tidx = indices[idx]
            qpcr_variances.add_qpcr_measurement(ridx=ridx, tidx=tidx, l=l)
            qpcr_dofs.add_qpcr_measurement(ridx=ridx, tidx=tidx, l=l)
            qpcr_scales.add_qpcr_measurement(ridx=ridx, tidx=tidx, l=l)
            iii += 1

    qpcr_dofs.set_shape()
    qpcr_scales.set_shape()

    # Set the IDs of the variables in the graph
    REPRNAMES.set(G=GRAPH)

    ######################################################################
    # Set up inference and inference order.
    mcmc = pl.inference.BaseMCMC(
        burnin=params.BURNIN,
        n_samples=params.N_SAMPLES,
        graph=GRAPH)
    order = []
    for name in params.INFERENCE_ORDER:
        if params.LEARN[name]:
            if not STRNAMES.is_perturbation_param(name):
                order.append(name)
            elif subjset.perturbations is not None:
                order.append(name)
    mcmc.set_inference_order(order)

    ######################################################################
    # Initialize
    for name in params.INITIALIZATION_ORDER:
        logging.info('Initializing {}'.format(name))
        if STRNAMES.is_perturbation_param(name) and subjset.perturbations is None:
            logging.info('Skipping over {} because it is a perturbation parameter ' \
                'and there are no perturbations'.format(name))
            continue
        
        # Call `initialize`
        try:
            GRAPH[name].initialize(**params.INITIALIZATION_KWARGS[name])
        except Exception as error:
            logging.critical('Initialization in `{}` failed with the parameters: {}'.format(
                name, params.INITIALIZATION_KWARGS[name]) + ' with the follwing error:\n{}'.format(
                    error))
            for a in GRAPH._persistent_pntr:
                a.kill()
            raise

        # Initialize data matrices if necessary
        if name == STRNAMES.ZERO_INFLATION:
            # Initialize the basic data matrices after initializing filtering
            lhs = data.LHSVector(G=GRAPH, name='lhs_vector', data_logscale=params.DATA_LOGSCALE)
            lhs.build()
            growthDM = data.GrowthDesignMatrix(G=GRAPH, name='growth_design_matrix',
                data_logscale=params.DATA_LOGSCALE, perturbations_additive=params.PERTURBATIONS_ADDITIVE)
            growthDM.build_without_perturbations()
            selfinteractionsDM = data.SelfInteractionDesignMatrix(G=GRAPH,
                name='self_interactions_design_matrix',
                data_logscale=params.DATA_LOGSCALE)
            selfinteractionsDM.build()
        if name == STRNAMES.CLUSTER_INTERACTION_INDICATOR:
            # Initialize the interactions data matrices after initializing the interactions
            interactionsDM = data.InteractionsDesignMatrix(G=GRAPH, data_logscale=params.DATA_LOGSCALE)
            interactionsDM.build()
        if name == STRNAMES.PERT_INDICATOR and subjset.perturbations is not None:
            # Initialize the perturbation data matrices after initializing the perturbations
            perturbationsDM = data.PerturbationDesignMatrix(G=GRAPH, data_logscale=params.DATA_LOGSCALE,
                perturbations_additive=params.PERTURBATIONS_ADDITIVE)
            perturbationsDM.base.build()
            perturbationsDM.M.build()
        if name == STRNAMES.PERT_VALUE and subjset.perturbations is not None and not params.PERTURBATIONS_ADDITIVE:
            d.design_matrices[REPRNAMES.GROWTH_VALUE].build_with_perturbations()

    logging.info('\n\n\n')
    logging.info('Initialization Values:')
    logging.info('Growth')
    logging.info('\tprior.mean: {}'.format(GRAPH[STRNAMES.GROWTH_VALUE].prior.mean.value))
    logging.info('\tprior.var: {}'.format(GRAPH[STRNAMES.GROWTH_VALUE].prior.var.value))
    logging.info('\tvalue: {}'.format(GRAPH[STRNAMES.GROWTH_VALUE].value.flatten()))

    logging.info('Self-Interactions')
    logging.info('\tprior.mean: {}'.format(GRAPH[STRNAMES.SELF_INTERACTION_VALUE].prior.mean.value))
    logging.info('\tprior.var: {}'.format(GRAPH[STRNAMES.SELF_INTERACTION_VALUE].prior.var.value))
    logging.info('\tvalue: {}'.format(GRAPH[STRNAMES.SELF_INTERACTION_VALUE].value.flatten()))

    logging.info('Prior Variance Growth')
    logging.info('\tprior.dof: {}'.format(GRAPH[STRNAMES.PRIOR_VAR_GROWTH].prior.dof.value))
    logging.info('\tprior.scale: {}'.format(GRAPH[STRNAMES.PRIOR_VAR_GROWTH].prior.scale.value))
    logging.info('\tvalue: {}'.format(GRAPH[STRNAMES.PRIOR_VAR_GROWTH].value))

    logging.info('Prior Variance Self-Interactions')
    logging.info('\tprior.dof: {}'.format(GRAPH[STRNAMES.PRIOR_VAR_SELF_INTERACTIONS].prior.dof.value))
    logging.info('\tprior.scale: {}'.format(GRAPH[STRNAMES.PRIOR_VAR_SELF_INTERACTIONS].prior.scale.value))
    logging.info('\tvalue: {}'.format(GRAPH[STRNAMES.PRIOR_VAR_SELF_INTERACTIONS].value))

    logging.info('Prior Variance Interactions')
    logging.info('\tprior.dof: {}'.format(GRAPH[STRNAMES.PRIOR_VAR_INTERACTIONS].prior.dof.value))
    logging.info('\tprior.scale: {}'.format(GRAPH[STRNAMES.PRIOR_VAR_INTERACTIONS].prior.scale.value))
    logging.info('\tvalue: {}'.format(GRAPH[STRNAMES.PRIOR_VAR_INTERACTIONS].value))

    logging.info('Process Variance')
    if params.DATA_LOGSCALE:
        logging.info('\tprior.dof: {}'.format(GRAPH[STRNAMES.PROCESSVAR].prior.dof.value))
        logging.info('\tprior.scale: {}'.format(GRAPH[STRNAMES.PROCESSVAR].prior.scale.value))
        logging.info('\tprior mean: {}'.format(GRAPH[STRNAMES.PROCESSVAR].prior.mean()))
    else:
        logging.info('\tv1: {}'.format(GRAPH[STRNAMES.PROCESSVAR].v1))
        logging.info('\tv2: {}'.format(GRAPH[STRNAMES.PROCESSVAR].v2))

    logging.info('Concentration')
    logging.info('\tprior.shape: {}'.format(GRAPH[STRNAMES.CONCENTRATION].prior.shape.value))
    logging.info('\tprior.scale: {}'.format(GRAPH[STRNAMES.CONCENTRATION].prior.scale.value))
    logging.info('\tvalue: {}'.format(GRAPH[STRNAMES.CONCENTRATION].value))

    logging.info('Indicator probability')
    logging.info('\tprior.a: {}'.format(GRAPH[STRNAMES.INDICATOR_PROB].prior.a.value))
    logging.info('\tprior.b: {}'.format(GRAPH[STRNAMES.INDICATOR_PROB].prior.b.value))
    logging.info('\tvalue: {}'.format(GRAPH[STRNAMES.INDICATOR_PROB].value))

    if subjset.perturbations is not None:
        logging.info('Perturbation values:')
        for perturbation in GRAPH.perturbations:
            logging.info('\tperturbation {}'.format(perturbation.name))
            logging.info('\t\tvalue: {}'.format(perturbation.magnitude.value))
            logging.info('\t\tprior.mean: {}'.format(perturbation.magnitude.prior.mean.value))
        logging.info('Perturbation prior variances:')
        for perturbation in GRAPH.perturbations:
            logging.info('\t\tdof: {}'.format(perturbation.magnitude.prior.var.prior.dof.value))
            logging.info('\t\tscale: {}'.format(perturbation.magnitude.prior.var.prior.scale.value))
            logging.info('\t\tvalue: {}'.format(perturbation.magnitude.prior.var.value))
        logging.info('Perturbation indicators:')
        for perturbation in GRAPH.perturbations:
            logging.info('\tperturbation {}: {}'.format(perturbation.name,
                perturbation.indicator.cluster_array()))
        logging.info('Perturbation indicator probability:')
        for perturbation in GRAPH.perturbations:
            logging.info('\tperturbation {}'.format(perturbation.name))
            logging.info('\t\tvalue: {}'.format(perturbation.probability.value))
            logging.info('\t\tprior.a: {}'.format(perturbation.probability.prior.a.value))
            logging.info('\t\tprior.b: {}'.format(perturbation.probability.prior.b.value))

    logging.info('\n\n\n')

    ######################################################################
    # Set up diagnostic variables
    if params.DIAGNOSTIC_VARIABLES is not None:
        diag_vars = []
        for diag_var in params.DIAGNOSTIC_VARIABLES:
            if diag_var == 'n_clusters':
                diag_vars.append(GRAPH[
                    STRNAMES.CLUSTER_INTERACTION_VALUE].clustering.n_clusters)
            elif diag_var in GRAPH:
                diag_vars.append(GRAPH[diag_var])
            else:
                cluster_assignments.kill()
                raise ValueError('Diagnostic variable `{}` was not `n_clusters` or ' \
                    'in {}'.format(diag_var,list(GRAPH.name2id.keys())))
        mcmc.set_diagnostic_variables(vars=diag_vars)

    ######################################################################
    # Setup tracer, filenames and run
    mcmc.set_tracer(filename=hdf5_filename, ckpt=checkpoint_iter)
    mcmc.tracer.set_save_location(tracer_filename)
    mcmc.set_save_location(mcmc_filename)
    GRAPH.set_save_location(graph_filename)

    return run_inference(mcmc=mcmc, crash_if_error=crash_if_error, 
        cluster_assignments=cluster_assignments, filtering=filtering, Z=Z,
        subjset=subjset, data_filename=data_filename)

def run_inference(mcmc, crash_if_error, cluster_assignments, filtering, Z, 
    subjset, data_filename):
    try:
        mcmc = mcmc.run(log_every=1)
    except Exception as e:
        logging.critical('CHAIN `{}` CRASHED'.format(mcmc.graph.name))
        logging.critical('Error: {}'.format(e))
        if crash_if_error:
            raise
    cluster_assignments.kill()
    filtering.kill()
    Z.kill()

    ######################################################################
    # Unnormalize the learned parameters if necessary
    if subjset.qpcr_normalization_factor is not None:
        subjset, mcmc = unnormalize_parameters(subjset=subjset, mcmc=mcmc)
        subjset.save(data_filename)

    return mcmc

def unnormalize_parameters(subjset, mcmc):
    '''Unnormalize the parameters if they are still normalized
    '''
    GRAPH = mcmc.graph
    if subjset.qpcr_normalization_factor is not None:
        logging.info('Denormalizing the parameters')
        # Because loading the entire thing is slow - do it piece by piece
        f = h5py.File(GRAPH.tracer.filename, 'r+', libver='latest')
        ckpt = GRAPH.tracer.ckpt

        GRAPH[STRNAMES.PROCESSVAR].c_m /= subjset.qpcr_normalization_factor
        GRAPH[STRNAMES.FILTERING].v2 /= subjset.qpcr_normalization_factor

        # Adjust the self interactions if necessary
        if STRNAMES.SELF_INTERACTION_VALUE in mcmc.tracer.being_traced:
            dset = f[STRNAMES.SELF_INTERACTION_VALUE]
            dset[:,:] = dset[:,:] * subjset.qpcr_normalization_factor

        if STRNAMES.PRIOR_MEAN_SELF_INTERACTIONS in mcmc.tracer.being_traced:
            dset = f[STRNAMES.PRIOR_MEAN_SELF_INTERACTIONS]
            dset[:] = dset[:] * subjset.qpcr_normalization_factor

        if mcmc.is_in_inference_order(STRNAMES.PRIOR_VAR_SELF_INTERACTIONS):
            dset = f[STRNAMES.PRIOR_VAR_SELF_INTERACTIONS]
            dset[:] = dset[:] * (subjset.qpcr_normalization_factor**2)

        if mcmc.is_in_inference_order(STRNAMES.QPCR_VARIANCES):
            vs = GRAPH[STRNAMES.QPCR_VARIANCES]
            for l in range(vs.L):
                dset = f[STRNAMES.QPCR_VARIANCES + '_{}'.format(l)]
                dset[:] = dset[:] * (subjset.qpcr_normalization_factor**2)

        if mcmc.is_in_inference_order(STRNAMES.QPCR_SCALES):
            vs = GRAPH[STRNAMES.QPCR_SCALES]
            for l in range(vs.L):
                dset = f[STRNAMES.QPCR_SCALES + '_{}'.format(l)]
                dset[:] = dset[:] * (subjset.qpcr_normalization_factor**2)

        # Adjust the interactions if necessary
        if mcmc.tracer.is_being_traced(STRNAMES.INTERACTIONS_OBJ):
            dset = f[STRNAMES.INTERACTIONS_OBJ]
            total_samples = dset.attrs['end_iter']
            i = 0
            while (i * ckpt) < total_samples:
                start_idx = int(i * ckpt)
                end_idx = int((i+1) * ckpt)

                if end_idx > total_samples:
                    end_idx = total_samples
                dset[start_idx: end_idx] = dset[start_idx: end_idx] * subjset.qpcr_normalization_factor
                i += 1
        
        if mcmc.is_in_inference_order(STRNAMES.PRIOR_MEAN_INTERACTIONS):
            dset = f[STRNAMES.PRIOR_MEAN_INTERACTIONS]
            dset[:] = dset[:] * subjset.qpcr_normalization_factor

        if mcmc.is_in_inference_order(STRNAMES.PRIOR_VAR_INTERACTIONS):
            dset = f[STRNAMES.PRIOR_VAR_INTERACTIONS]
            dset[:] = dset[:] * (subjset.qpcr_normalization_factor**2)

        if mcmc.is_in_inference_order(STRNAMES.FILTERING):
            for ridx in range(len(subjset)):
                for dset_name in [STRNAMES.LATENT_TRAJECTORY, STRNAMES.AUX_TRAJECTORY]:
                    name = dset_name + '_ridx{}'.format(ridx)
                    if name not in f:
                        continue
                    dset = f[name]
                    total_samples = dset.attrs['end_iter']
                    i = 0
                    while (i * ckpt) < total_samples:
                        start_idx = int(i * ckpt)
                        end_idx = int((i+1) * ckpt)

                        if end_idx > total_samples:
                            end_idx = total_samples
                        dset[start_idx: end_idx] = dset[start_idx: end_idx]/subjset.qpcr_normalization_factor
                        i += 1

        f.close()
        # Denormalize
        subjset.denormalize_qpcr()
    else:
        logging.info('Data already denormalized')
    return subjset, mcmc

def normalize_parameters(subjset, mcmc):
    '''Unnormalize the parameters if they are still normalized
    '''
    GRAPH = mcmc.graph
    normalization_factor = subjset.qpcr_normalization_factor
    if subjset.qpcr_normalization_factor is not None:
        # Because loading the entire thing is slow - do it piece by piece
        f = h5py.File(GRAPH.tracer.filename, 'r+', libver='latest')
        ckpt = GRAPH.tracer.ckpt

        GRAPH[STRNAMES.PROCESSVAR].c_m *= subjset.qpcr_normalization_factor
        GRAPH[STRNAMES.FILTERING].v2 *= subjset.qpcr_normalization_factor

        # Adjust the self interactions if necessary
        if STRNAMES.SELF_INTERACTION_VALUE in mcmc.tracer.being_traced:
            dset = f[STRNAMES.SELF_INTERACTION_VALUE]
            dset[:,:] = dset[:,:] / subjset.qpcr_normalization_factor

        if STRNAMES.PRIOR_MEAN_SELF_INTERACTIONS in mcmc.tracer.being_traced:
            dset = f[STRNAMES.PRIOR_MEAN_SELF_INTERACTIONS]
            dset[:] = dset[:] / subjset.qpcr_normalization_factor

        if mcmc.is_in_inference_order(STRNAMES.PRIOR_VAR_SELF_INTERACTIONS):
            dset = f[STRNAMES.PRIOR_VAR_SELF_INTERACTIONS]
            dset[:] = dset[:] / (subjset.qpcr_normalization_factor**2)

        if mcmc.is_in_inference_order(STRNAMES.QPCR_VARIANCES):
            vs = GRAPH[STRNAMES.QPCR_VARIANCES]
            for l in range(vs.L):
                dset = f[STRNAMES.QPCR_VARIANCES + '_{}'.format(l)]
                dset[:] = dset[:] / (subjset.qpcr_normalization_factor**2)

        if mcmc.is_in_inference_order(STRNAMES.QPCR_SCALES):
            vs = GRAPH[STRNAMES.QPCR_SCALES]
            for l in range(vs.L):
                dset = f[STRNAMES.QPCR_SCALES + '_{}'.format(l)]
                dset[:] = dset[:] / (subjset.qpcr_normalization_factor**2)

        # Adjust the interactions if necessary
        if mcmc.tracer.is_being_traced(STRNAMES.INTERACTIONS_OBJ):
            dset = f[STRNAMES.INTERACTIONS_OBJ]
            total_samples = dset.attrs['end_iter']
            i = 0
            while (i * ckpt) < total_samples:
                start_idx = int(i * ckpt)
                end_idx = int((i+1) * ckpt)

                if end_idx > total_samples:
                    end_idx = total_samples
                dset[start_idx: end_idx] = dset[start_idx: end_idx] / subjset.qpcr_normalization_factor
                i += 1
        
        if mcmc.is_in_inference_order(STRNAMES.PRIOR_MEAN_INTERACTIONS):
            dset = f[STRNAMES.PRIOR_MEAN_INTERACTIONS]
            dset[:] = dset[:] / subjset.qpcr_normalization_factor

        if mcmc.is_in_inference_order(STRNAMES.PRIOR_VAR_INTERACTIONS):
            dset = f[STRNAMES.PRIOR_VAR_INTERACTIONS]
            dset[:] = dset[:] / (subjset.qpcr_normalization_factor**2)

        if mcmc.is_in_inference_order(STRNAMES.FILTERING):
            for ridx in range(len(subjset)):
                for dset_name in [STRNAMES.LATENT_TRAJECTORY, STRNAMES.AUX_TRAJECTORY]:
                    name = dset_name + '_ridx{}'.format(ridx)
                    if name not in f:
                        continue
                    dset = f[name]
                    total_samples = dset.attrs['end_iter']
                    i = 0
                    while (i * ckpt) < total_samples:
                        start_idx = int(i * ckpt)
                        end_idx = int((i+1) * ckpt)

                        if end_idx > total_samples:
                            end_idx = total_samples
                        dset[start_idx: end_idx] = dset[start_idx: end_idx]*subjset.qpcr_normalization_factor
                        i += 1

        f.close()
        # Denormalize
        # subjset.denormalize_qpcr()
    return subjset, mcmc

def copy_basepath(basepath, copy_path, copy_dirs=False):
    '''Copies the folder and its contents into the path `copy_path`.
    This will only copy the nested directories inside of `basepath` if `copy_dirs` is `True`.

    Parameters
    ----------
    basepath : str
        This is the folder that you want to copy.
    copy_path : str
        This is the path to copy everything to.
    copy_dirs : bool
        If True, it will copy the nested folders inside of basepath - if not 
        it will not copy them and skip over
    '''
    if not pl.isstr(basepath):
        raise TypeError('`basepath` ({}) must be a str'.format(type(basepath)))
    if not os.path.isdir(basepath):
        raise ValueError('`basepath` ({}) does not exist'.format(basepath))
    if not pl.isstr(copy_path):
        raise TypeError('`copy_path` ({}) must be a str'.format(type(copy_path)))
    if not os.path.isdir(copy_path):
        os.makedirs(copy_path, exist_ok=True)
    else:
        warnings.warn('`copy_path` ({}) already exists'.format(copy_path))
    if not pl.isbool(copy_dirs):
        raise TypeError('`copy_dirs` ({}) must be a bool'.format(type(copy_dirs)))

    if copy_path[-1] != '/':
        copy_path = copy_path + '/'
    if basepath[-1] != '/':
        basepath = basepath + '/'

    for filename in os.listdir(basepath):
        logging.debug('Copying file {}'.format(filename))
        if os.path.isdir(basepath + filename):
            if copy_dirs:
                copy_basepath(
                    basepath=basepath+filename, 
                    copy_path=copy_path+filename, 
                    copy_dirs=True)
        else:
            temp_src = basepath + filename
            temp_dst = copy_path + filename
            shutil.copyfile(temp_src, temp_dst)

    return

# @profile
def readify_chain(src_basepath, params, dst_basepath=None, plot_diagnostic_variables=True,
    asv_prefix_formatter='%(index)s %(name)s ', yticklabels='%(name)s %(index)s',
    xticklabels='%(index)s', yscale_log=True, center_color_for_strength=True, percentile = 5.,
    run_on_copy=True, plot_name_filtering=None, sort_interactions_by_cocluster=False,
    plot_gif_filtering=False, plot_filtering_thresh=True, exact_filename=None,
    syndata=None, calculate_keystoneness=True):
    '''Makes human readable printout of the graph in the chain as well
    as creating figures of the diagnostic variables and the learned variables.

    If the chain crashed before the posterior, then we calculate the statistics
    based on the burnin.

    Parameters
    ----------
    src_basepath : str
        Folder where all the files are that you are plotting
    dst_basepath : str, None
        This is the folder that you want the posterior. If this is None, then 
        `dst_basepath = src_basepath + 'posterior'`.
    plot_diagnostic_variables : bool, Optional
        If True, plot the diagnostic variables
    asv_prefix_formatter: str, Optional
        This is a formatter for the prefix for each ASV. For variables that
        are per ASV (like growth parameters) it is nice to get the
        attributes of the ASV like it's genus or the ASV name given by DADA
        instead of just an index. This formatter lets you do this. The format
        will be passed into `pl.asvname_formatter` and the output of
        this will be prepended to the value. Look at the documentation for
        that function to get the valid formatting options.
        If you do not want one then set as `None`.
    yticklabels, xticklabels : str, Optional
        Formatter for doing the ytick labels in plots
    yscale_log : bool
        If True, plot everything in log-scale
    center_color_for_strength : bool
        If True, center the colors for the strangeth matrix
    percentile : numeric, positive
        This is the percentile to plot the filtering parameters at
        Example:
            percentile = 5.
                Plot the 5th and 95th percentile with the plot
    run_on_copy : bool
        If True, it will copy the src data before it will plot it
    plot_name_filtering : str, None
        If True, creates a figure title of the name of the ASV with the specified format.
        If None, there is no plotting.,
    sort_interactions_by_cocluster : bool
        If True, it will use the order specified by how strong the coclustering proportions
        are instead of the automatic ordering by the sorting 
    plot_filtering_thresh : bool
        If False, do not plot the filtering threshold when plotting the filtering
    syndata : synthetic.SyntheticData, str
        This is an optional synthetic data object if we want to compare the learned parameters
        to the true value. If it is a string then this si the filename location to load it from
    '''
    logging.basicConfig(level=logging.INFO)
    if not pl.isstr(src_basepath):
        raise TypeError('`src_basepath` ({}) must be a str'.format(type(src_basepath)))
    if not os.path.isdir(src_basepath):
        raise ValueError('`src_basepath` ({}) does not exist')
    if src_basepath[-1] != '/':
            src_basepath = src_basepath + '/'
    if dst_basepath is None:
        dst_basepath = src_basepath + 'posterior/'
    else:
        if not pl.isstr(dst_basepath):
            raise TypeError('`dst_basepath` ({}) must be a str'.format(type(dst_basepath)))
        os.makedirs(dst_basepath, exist_ok=True)
    if dst_basepath[-1] != '/':
        dst_basepath = dst_basepath + '/'
    if not np.all(pl.itercheck([
        plot_diagnostic_variables, 
        yscale_log, 
        center_color_for_strength,
        run_on_copy], pl.isbool)):
        raise TypeError('`plot_diagnostic_variables` ({}) `yscale_log` ({}) `center_color_for_strength` ({}), '\
            'and `run_on_copy` ({}) must be bools'.format(type(plot_diagnostic_variables), type(yscale_log), 
            type(center_color_for_strength), type(run_on_copy)))
    if asv_prefix_formatter is None:
        asv_prefix_formatter = ''
    if not pl.isstr(asv_prefix_formatter):
        raise TypeError('`asv_prefix_formatter` ({}) must be a str'.format(type(asv_prefix_formatter)))
    
    if run_on_copy:
        logging.info('Copying basepath')
        cpy_basepath = src_basepath[:-1] + '_copy/'
        copy_basepath(basepath=src_basepath, copy_path=cpy_basepath, copy_dirs=False)
        src_basepath = cpy_basepath

    basepath = dst_basepath
    chain = pl.inference.BaseMCMC.load(src_basepath + config.MCMC_FILENAME)
    try:
        fparams = config.FilteringConfig.load(src_basepath + config.FPARAMS_FILENAME)
    except:
        fparams = None
    subjset = pl.SubjectSet.load(src_basepath + config.SUBJSET_FILENAME)

    # Unnormalize if necessary
    subjset, chain = unnormalize_parameters(subjset=subjset, mcmc=chain)
    subjset.save(src_basepath + config.SUBJSET_FILENAME)
    # print('normalize1')
    # subjset,chain = normalize_parameters(subjset=subjset, mcmc=chain)
    # print('normalize2')
    # subjset,chain = normalize_parameters(subjset=subjset, mcmc=chain)
    # subjset.denormalize_qpcr()
    

    if exact_filename is not None:
        exact_subjset = pl.base.SubjectSet.load(exact_filename)
    if syndata is not None:
        if pl.isstr(syndata):
            syndata = synthetic.SyntheticData.load(syndata)
        if not synthetic.issynthetic(syndata):
            raise TypeError('`syndata` ({}) must be type synthetic.SyntheticData'.format(type(syndata)))

    # # Check if we have an entire chain, partial train, or only burnin
    if chain.sample_iter < 100:
        logging.critical('There are too few samples to find the posterior ({} samples)'.format(
            chain.sample_iter))
        return
    if chain.sample_iter > chain.burnin:
        SECTION = 'posterior'
        LEN_POSTERIOR = chain.sample_iter+1 - chain.burnin
    elif chain.sample_iter <= chain.burnin:
        SECTION = 'burnin'
        LEN_POSTERIOR = chain.sample_iter + 1

    # if the directory already exists, delete it and make a new one
    if os.path.isdir(basepath):
        shutil.rmtree(basepath, ignore_errors=True)
    os.makedirs(basepath, exist_ok=True)

    # If we learned clusters, put them together in plotting
    CLUSTERING = chain.graph[STRNAMES.CLUSTERING_OBJ]
    ASV_ORDER = None
    ASVS = CLUSTERING.items
    try:
        CLUSTERING.generate_cluster_assignments_posthoc(n_clusters='mode', set_as_value=True)
        asvorder = []
        for cluster in CLUSTERING:
            for oidx in cluster.members:
                asvorder.append(oidx)
        ASV_ORDER = asvorder
    except:
        # Do nothing because we did not learn clustering
        pass

    REGRESS_COEFF = chain.graph[STRNAMES.REGRESSCOEFF]

    overviewpath = basepath + 'overview.txt'
    f = open(overviewpath, 'w')
    f.write('###################################\n')
    f.write('Output from chain `{}`\n'.format(chain.graph.name))
    f.write('###################################\n')
    f.write('Seed: {}\n'.format(chain.graph.seed))
    f.write('Total number of samples: {}\n'.format(chain.n_samples))
    f.write('Burnin: {}\n'.format(chain.burnin))
    f.write('Number of samples actually done: {}\n'.format(chain.sample_iter))
    f.write('Number of ASVs: {}\n'.format(chain.graph.data.asvs.n_asvs))
    f.write('Inference Order:\n')
    for i,ele in enumerate(chain.inf_order):
        f.write('\t{}\n'.format(str(chain.graph[ele].name)))
    f.write('Learned Variables:\n')
    for ele in chain.tracer.being_traced:
        f.write('\t{}\n'.format(str(ele)))
    f.close()

    # Calculate the stability - save as a npy array
    logging.info('Calculating the stability')
    growth = chain.graph[STRNAMES.GROWTH_VALUE]
    if chain.tracer.is_being_traced(STRNAMES.GROWTH_VALUE):
        growth_values = growth.get_trace_from_disk(section=SECTION)
    else:
        growth_values = growth.value
        growth_values = np.zeros(shape=(LEN_POSTERIOR, len(ASVS))) + growth_values.reshape(-1,1)

    si = chain.graph[STRNAMES.SELF_INTERACTION_VALUE]
    if chain.tracer.is_being_traced(STRNAMES.SELF_INTERACTION_VALUE):
        si_values = si.get_trace_from_disk(section=SECTION)
    else:
        si_values = si.value
        si_values = np.zeros(shape=(LEN_POSTERIOR, len(ASVS))) + si_values.reshape(-1,1)

    interactions = chain.graph[STRNAMES.INTERACTIONS_OBJ]
    if chain.tracer.is_being_traced(STRNAMES.INTERACTIONS_OBJ):
        interaction_values = interactions.get_trace_from_disk(section=SECTION)
        interaction_values[np.isnan(interaction_values)] = 0
    else:
        interactions = interactions.get_datalevel_value_matrix(set_neg_indicators_to_nan=False)
        interaction_values = np.zeros(shape=(LEN_POSTERIOR, len(ASVS), len(ASVS))) + interactions

    stabil = np.zeros(shape=(LEN_POSTERIOR, len(ASVS), len(ASVS)))
    stabil = calc_eigan_over_gibbs(ret=stabil, growth=growth_values, si=si_values, 
        interactions=interaction_values)
    np.save(basepath + 'stability.npy', stabil)

    # Plot growth parameters
    growthpath = basepath + 'growth/'
    fgrowthpath = growthpath + 'output.txt'
    os.makedirs(growthpath, exist_ok=True)
    f = open(fgrowthpath, 'w')
    f.close()

    if chain.tracer.is_being_traced(STRNAMES.PRIOR_MEAN_GROWTH):
        growth_mean = chain.graph[STRNAMES.PRIOR_MEAN_GROWTH]
        f = open(fgrowthpath, 'a')

        summary = pl.variables.summary(growth_mean, section=SECTION)
        f.write('###################################\n')
        f.write('ASV Growth Mean\n')
        f.write('###################################\n')
        for key,ele in summary.items():
            f.write('{}: {}\n'.format(key,ele))

        # Show prior on axis 1, show acceptance rate on axis 2
        ax1, ax2 = pl.visualization.render_trace(var=growth_mean, plt_type='both', 
            section=SECTION, include_burnin=True, rasterized=True)
        l,h = ax1.get_xlim()
        xs = np.arange(l,h,step=(h-l)/1000) 
        ys = []
        for x in xs:
            ys.append(growth_mean.prior.pdf(value=x))
        ax1.plot(xs, ys, label='prior', alpha=0.5, color='red', rasterized=True)
        ax1.legend()

        ax3 = ax2.twinx()
        ax3 = pl.visualization.render_acceptance_rate_trace(var=growth_mean, ax=ax3, 
            label='Acceptance Rate', color='red', scatter=False, rasterized=True)
        ax3.legend()

        fig = plt.gcf()
        fig.tight_layout()
        fig.suptitle('Growth Mean, {}'.format(LATEXNAMES.PRIOR_MEAN_GROWTH))
        plt.savefig(growthpath + 'mean.pdf')
        plt.close()
        f.close()
    else:
        # Specify what the mean was
        growth_mean = chain.graph[STRNAMES.PRIOR_MEAN_GROWTH]
        f = open(fgrowthpath, 'a')
        f.write('###################################\n')
        f.write('ASV Growth Mean\n')
        f.write('###################################\n')
        f.write('Not learned. Value: {}\n'.format(growth_mean.value))
        f.close()

    if chain.is_in_inference_order(STRNAMES.PRIOR_VAR_GROWTH):
        f = open(fgrowthpath, 'a')
        pv = chain.graph[STRNAMES.PRIOR_VAR_GROWTH]
        summary = pl.variables.summary(pv, section=SECTION)
        f.write('\n\n###################################\n')
        f.write('Prior Variance Growth\n')
        f.write('###################################\n')
        for key,ele in summary.items():
            f.write('{}: {}\n'.format(key,ele))
        ax1, ax2 = pl.visualization.render_trace(var=pv, plt_type='both', 
            section=SECTION, include_burnin=True, log_scale=True, rasterized=True)

        l,h = ax1.get_xlim()
        xs = np.arange(l,h,step=(h-l)/100) 
        ys = []
        for x in xs:
            ys.append(pl.random.sics.pdf(value=x, 
                dof=pv.prior.dof.value,
                scale=pv.prior.scale.value))
        ax1.plot(xs, ys, label='prior', alpha=0.5, color='red', rasterized=True)
        ax1.legend()

        ax3 = ax2.twinx()
        ax3 = pl.visualization.render_acceptance_rate_trace(var=pv, ax=ax3, 
            label='Acceptance Rate', color='red', scatter=False, rasterized=True)
        ax3.legend()

        fig = plt.gcf()
        fig.tight_layout()
        fig.suptitle('Prior Variance Growth, {}'.format(LATEXNAMES.PRIOR_VAR_GROWTH))
        plt.savefig(growthpath + 'var.pdf')
        plt.close()
        f.close()
    else:
        # Specify what the mean was
        pv = chain.graph[STRNAMES.PRIOR_VAR_GROWTH]
        f = open(fgrowthpath, 'a')
        f.write('\n\n###################################\n')
        f.write('Prior Variance Growth\n')
        f.write('###################################\n')
        f.write('Not learned. Value: {}\n'.format(pv.value))
        f.close()

    if chain.tracer.is_being_traced(STRNAMES.GROWTH_VALUE):
        growth = chain.graph[STRNAMES.GROWTH_VALUE]

        # write summary statistics to main regress_coeff file
        f = open(fgrowthpath, 'a')
        f.write('\n\n###################################\n')
        f.write('ASV Growth Values\n')
        f.write('###################################\n')
        summary = pl.variables.summary(growth, section=SECTION)
        for key,arr in summary.items():
            f.write('{}\n'.format(key))
            for idx,ele in enumerate(arr):
                prefix = ''
                if asv_prefix_formatter is not None:
                    prefix = pl.asvname_formatter(
                        format=asv_prefix_formatter,
                        asv=ASVS.names.order[idx],
                        asvs=ASVS)
                f.write('\t' + prefix + '{}\n'.format(ele))        

        # Plot the prior on top of the posterior
        if chain.tracer.is_being_traced(STRNAMES.PRIOR_MEAN_GROWTH):
            prior_mean_trace =chain.graph[STRNAMES.PRIOR_MEAN_GROWTH].get_trace_from_disk(
                    section=SECTION)
        else:
            prior_mean_trace = growth.prior.mean.value * np.ones(
                LEN_POSTERIOR, dtype=float)
        if chain.tracer.is_being_traced(STRNAMES.PRIOR_VAR_GROWTH):
            prior_std_trace = np.sqrt(
                chain.graph[STRNAMES.PRIOR_VAR_GROWTH].get_trace_from_disk(
                    section=SECTION))
        else:
            prior_std_trace = np.sqrt(growth.prior.var.value) * np.ones(
                LEN_POSTERIOR, dtype=float)

        for idx in range(ASVS.n_asvs):
            fig = plt.figure()
            ax_posterior = fig.add_subplot(1,2,1)
            pl.visualization.render_trace(var=growth, idx=idx, plt_type='hist',
                label=SECTION, color='blue', ax=ax_posterior, section=SECTION,
                include_burnin=True, rasterized=True)

            # Get the limits and only look at the posterior within 20% range +- of
            # this number
            low_x, high_x = ax_posterior.get_xlim()

            arr = np.zeros(len(prior_std_trace), dtype=float)
            for i in range(len(prior_std_trace)):
                arr[i] = pl.random.truncnormal.sample(mean=prior_mean_trace[i], std=prior_std_trace[i], 
                    low=growth.low, high=growth.high)
            pl.visualization.render_trace(var=arr, plt_type='hist', 
                label='prior', color='red', ax=ax_posterior, rasterized=True)

            if syndata is not None:
                ax_posterior.axvline(x=syndata.dynamics.growth[idx], color='red', alpha=0.65, 
                    label='True Value')

            ax_posterior.legend()
            ax_posterior.set_xlim(left=low_x*.8, right=high_x*1.2)

            # plot the trace
            ax_trace = fig.add_subplot(1,2,2)
            pl.visualization.render_trace(var=growth, idx=idx, plt_type='trace', 
                ax=ax_trace, section=SECTION, include_burnin=True, rasterized=True)

            if syndata is not None:
                ax_trace.axhline(y=syndata.dynamics.growth[idx], color='red', alpha=0.65, 
                    label='True Value')
                ax_trace.legend()

            if asv_prefix_formatter is not None:
                asvname = pl.asvname_formatter(
                    format=asv_prefix_formatter,
                    asv=ASVS.names.order[idx],
                    asvs=ASVS)
            else:
                asvname = ASVS.names.order[idx]
            latexname = pl.asvname_formatter(format=LATEXNAMES.GROWTH_VALUE,
                asv=idx, asvs=ASVS)
            asvname = asvname.replace('/', '_').replace(' ', '_')

            if REGRESS_COEFF.update_jointly_growth_si:
                ax3 = ax_trace.twinx()
                ax3 = pl.visualization.render_acceptance_rate_trace(var=growth, ax=ax3, 
                    label='Acceptance Rate', color='red', scatter=False, idx=idx, rasterized=True)
                ax3.legend()

            fig.suptitle('Growth {}, {}'.format(asvname, latexname))
            fig.tight_layout()
            fig.subplots_adjust(top=0.85)
            plt.savefig(growthpath + '{}.pdf'.format(ASVS.names.order[idx]))
            plt.close()

        f.close()

    # Plot self-interaction parameters
    sipath = basepath + 'self_interactions/'
    fsipath = sipath + 'output.txt'
    os.makedirs(sipath, exist_ok=True)
    f = open(fsipath, 'w')
    f.close()

    if chain.tracer.is_being_traced(STRNAMES.PRIOR_MEAN_SELF_INTERACTIONS):
        si_mean = chain.graph[STRNAMES.PRIOR_MEAN_SELF_INTERACTIONS]
        f = open(fsipath, 'a')

        summary = pl.variables.summary(si_mean, section=SECTION)
        f.write('###################################\n')
        f.write('ASV Self-interaction Mean\n')
        f.write('###################################\n')
        for key,ele in summary.items():
            f.write('{}: {}\n'.format(key,ele))

        ax1, ax2 = pl.visualization.render_trace(var=si_mean, plt_type='both', log_scale=yscale_log, 
            section=SECTION, include_burnin=True, rasterized=True)
        l,h = ax1.get_xlim()
        xs = np.arange(l,h,step=(h-l)/1000) 
        ys = []
        for x in xs:
            ys.append(si_mean.prior.pdf(value=x))
        ax1.plot(xs, ys, label='prior', alpha=0.5, color='red')
        ax1.legend()

        ax3 = ax2.twinx()
        ax3 = pl.visualization.render_acceptance_rate_trace(var=growth_mean, ax=ax3, 
            label='Acceptance Rate', color='red', scatter=False, rasterized=True)
        ax3.legend()

        fig = plt.gcf()
        fig.tight_layout()
        fig.suptitle('Self-interaction Mean, {}'.format(LATEXNAMES.PRIOR_MEAN_SELF_INTERACTIONS))
        plt.savefig(sipath + 'mean.pdf')
        plt.close()
        f.close()
    else:
        # Specify what the mean was
        si_mean = chain.graph[STRNAMES.PRIOR_MEAN_GROWTH]
        f = open(fsipath, 'a')
        f.write('###################################\n')
        f.write('ASV Self-interaction Mean\n')
        f.write('###################################\n')
        f.write('Not learned. Value: {}\n'.format(si_mean.value))
        f.close()

    if chain.is_in_inference_order(STRNAMES.PRIOR_VAR_SELF_INTERACTIONS):
        f = open(fsipath, 'a')
        pv = chain.graph[STRNAMES.PRIOR_VAR_SELF_INTERACTIONS]
        summary = pl.variables.summary(pv, section=SECTION)
        f.write('\n\n###################################\n')
        f.write('Prior Variance Self-Interactions\n')
        f.write('###################################\n')
        for key,ele in summary.items():
            f.write('{}: {}\n'.format(key,ele))
        ax1, ax2 = pl.visualization.render_trace(var=pv, plt_type='both', log_scale=yscale_log, 
            section=SECTION, include_burnin=True, rasterized=True)

        l,h = ax1.get_xlim()
        xs = np.arange(l,h,step=(h-l)/100) 
        ys = []
        for x in xs:
            ys.append(pl.random.sics.pdf(value=x, 
                dof=pv.prior.dof.value,
                scale=pv.prior.scale.value))
        ax1.plot(xs, ys, label='prior', alpha=0.5, color='red')
        ax1.legend()

        ax3 = ax2.twinx()
        ax3 = pl.visualization.render_acceptance_rate_trace(var=growth_mean, ax=ax3, 
            label='Acceptance Rate', color='red', scatter=False, rasterized=True)
        ax3.legend()

        fig = plt.gcf()
        fig.tight_layout()
        fig.suptitle('Prior Variance Self-Interactions, {}'.format(LATEXNAMES.PRIOR_VAR_SELF_INTERACTIONS))
        plt.savefig(sipath + 'var.pdf')
        plt.close()
        f.close()
    else:
        # Specify what the mean was
        pv = chain.graph[STRNAMES.PRIOR_VAR_SELF_INTERACTIONS]
        f = open(fsipath, 'a')
        f.write('\n\n###################################\n')
        f.write('Prior Variance Self-Interactions\n')
        f.write('###################################\n')
        f.write('Not learned. Value: {}\n'.format(pv.value))
        f.close()

    if chain.tracer.is_being_traced(STRNAMES.SELF_INTERACTION_VALUE):
        f = open(fsipath, 'a')
        self_interactions = chain.graph[STRNAMES.SELF_INTERACTION_VALUE]
        
        f.write('\n\n###################################\n')
        f.write('ASV Self Interaction Values\n')
        f.write('###################################\n')

        summary = pl.variables.summary(self_interactions, section=SECTION)
        for key,arr in summary.items():
            f.write('{}\n'.format(key))
            for idx,ele in enumerate(arr):
                prefix = ''
                if asv_prefix_formatter is not None:
                    prefix = pl.asvname_formatter(
                        format=asv_prefix_formatter,
                        asv=ASVS.names.order[idx],
                        asvs=ASVS)
                f.write('\t' + prefix + '{}\n'.format(ele))

        # Plot the prior on top of the posterior
        if chain.tracer.is_being_traced(STRNAMES.PRIOR_MEAN_SELF_INTERACTIONS):
            prior_mean_trace =chain.graph[STRNAMES.PRIOR_MEAN_SELF_INTERACTIONS].get_trace_from_disk(
                    section=SECTION)
        else:
            prior_mean_trace = self_interactions.prior.mean.value * np.ones(
                LEN_POSTERIOR, dtype=float)
        if chain.tracer.is_being_traced(STRNAMES.PRIOR_VAR_SELF_INTERACTIONS):
            prior_std_trace = np.sqrt(
                chain.graph[STRNAMES.PRIOR_VAR_SELF_INTERACTIONS].get_trace_from_disk(
                    section=SECTION))
        else:
            prior_std_trace = np.sqrt(self_interactions.prior.var.value) * np.ones(
                LEN_POSTERIOR, dtype=float)

        for idx in range(ASVS.n_asvs):
            fig = plt.figure()
            ax_posterior = fig.add_subplot(1,2,1)
            pl.visualization.render_trace(var=self_interactions, idx=idx, plt_type='hist',
                label='posterior', color='blue', ax=ax_posterior, log_scale=yscale_log,
                section=SECTION, include_burnin=True, rasterized=True)
            
            # Get the limits and only look at the posterior within 20% range +- of
            # this number
            low_x, high_x = ax_posterior.get_xlim()

            arr = np.zeros(len(prior_std_trace), dtype=float)
            for i in range(len(prior_std_trace)):
                arr[i] = pl.random.truncnormal.sample(mean=prior_mean_trace[i], std=prior_std_trace[i], 
                    low=self_interactions.low, high=self_interactions.high)
            pl.visualization.render_trace(var=arr, plt_type='hist', 
                label='prior', color='red', ax=ax_posterior, log_scale=yscale_log, rasterized=True)

            if syndata is not None:
                ax_posterior.axvline(x=syndata.dynamics.self_interactions[idx], color='red', 
                    alpha=0.65, label='True Value')

            ax_posterior.legend()
            ax_posterior.set_xlim(left=low_x*.8, right=high_x*1.2)

            # plot the trace
            ax_trace = fig.add_subplot(1,2,2)
            pl.visualization.render_trace(var=self_interactions, idx=idx, plt_type='trace', ax=ax_trace,
                log_scale=yscale_log, section=SECTION, include_burnin=True, rasterized=True)

            if syndata is not None:
                ax_trace.axhline(y=syndata.dynamics.self_interactions[idx], color='red', 
                    alpha=0.65, label='True Value')
                ax_trace.legend()

            if REGRESS_COEFF.update_jointly_growth_si:
                ax3 = ax_trace.twinx()
                ax3 = pl.visualization.render_acceptance_rate_trace(var=self_interactions, ax=ax3, 
                    label='Acceptance Rate', color='red', scatter=False, idx=idx, rasterized=True)
                ax3.legend()
            
            if asv_prefix_formatter is not None:
                asvname = pl.asvname_formatter(
                    format=asv_prefix_formatter,
                    asv=ASVS.names.order[idx],
                    asvs=ASVS)
            else:
                asvname = ASVS.names.order[idx]
            latexname = pl.asvname_formatter(
                format=LATEXNAMES.SELF_INTERACTION_VALUE,
                asv = ASVS[idx],
                asvs = ASVS)
            asvname = asvname.replace('/', '_').replace(' ', '_')
            fig.suptitle('Self-interactions {}, {}'.format(asvname, latexname))
            fig.tight_layout()
            fig.subplots_adjust(top=0.85)
            plt.savefig(sipath + '{}.pdf'.format(ASVS.names.order[idx]))
            plt.close()
        f.close()

    # Plot clustering parameters
    clusteringpath = basepath + 'clustering/'
    clustertrajpath = clusteringpath + 'trajectories/'
    fclusteringpath = clusteringpath + 'output.txt'
    os.makedirs(clusteringpath, exist_ok=True)
    os.makedirs(clustertrajpath, exist_ok=True)
    f = open(fclusteringpath, 'w')
    f.close()

    if chain.is_in_inference_order(STRNAMES.CONCENTRATION):
        f = open(fclusteringpath, 'a')
        concentration = chain.graph[STRNAMES.CONCENTRATION]
        summary = pl.variables.summary(concentration, section=SECTION)

        f.write('###################################\n')
        f.write('Concentration\n')
        f.write('###################################\n')
        for key,ele in summary.items():
            f.write('{}: {}\n'.format(key,ele))
        ax1, _ = pl.visualization.render_trace(var=concentration, plt_type='both', 
            section=SECTION, include_burnin=True, log_scale=True, rasterized=True)

        l,h = ax1.get_xlim()
        xs = np.arange(l,h,step=(h-l)/1000) 
        ys = []
        for x in xs:
            ys.append(concentration.prior.pdf(value=x))
        ax1.plot(xs, ys, label='prior', alpha=0.5, color='red')
        ax1.legend()

        fig = plt.gcf()
        fig.suptitle('Concentration, {}'.format(LATEXNAMES.CONCENTRATION))
        plt.savefig(clusteringpath + 'concentration.pdf')
        plt.close()
        f.close()
    else:
        f = open(fclusteringpath, 'a')
        f.write('\n\nNo cluster concentration learned')
        f.close()

    if chain.is_in_inference_order(STRNAMES.CLUSTERING):
        f = open(fclusteringpath, 'a')

        cocluster_trace = CLUSTERING.coclusters.get_trace_from_disk(section=SECTION)
        coclusters = pl.variables.summary(cocluster_trace, section=SECTION)['mean']
        for i in range(coclusters.shape[0]):
            coclusters[i,i] = np.nan

        pl.visualization.render_cocluster_proportions(
            coclusters=coclusters, asvs=ASVS, clustering=CLUSTERING,
            yticklabels=yticklabels, include_tick_marks=False, xticklabels=xticklabels,
            title='Cluster Assignments, {}'.format(LATEXNAMES.CLUSTERING), order=ASV_ORDER)
        fig = plt.gcf()
        fig.tight_layout()
        plt.savefig(clusteringpath + 'coclusters.pdf')
        plt.close()

        pl.visualization.render_trace(var=CLUSTERING.n_clusters, plt_type='both', 
            section=SECTION, include_burnin=True, rasterized=True)
        fig = plt.gcf()
        fig.suptitle('Number of Clusters')
        plt.savefig(clusteringpath + 'n_clusters.pdf')
        plt.close()

        # print out which ASV belongs in which cluster
        f.write('\n\n###################################\n')
        f.write('Clustering Assignments\n')
        f.write('###################################\n')
        ca = CLUSTERING.generate_cluster_assignments_posthoc(n_clusters='mean', section=SECTION)
        cluster_assignments = {}
        for idx, assignment in enumerate(ca):
            if assignment in cluster_assignments:
                cluster_assignments[assignment].append(idx)
            else:
                cluster_assignments[assignment] = [idx]
        f.write('Mean number of clusters: {}\n'.format(len(cluster_assignments)))
        for idx,lst in enumerate(cluster_assignments.values()):
            f.write('Cluster {} - Size {}\n'.format(idx, len(lst)))
            for oidx in lst:
                # Get rid of index because that does not really make sense here
                label = pl.asvname_formatter(
                    format=asv_prefix_formatter.replace('%(index)s',''),
                    asv=ASVS.index[oidx],
                    asvs=ASVS)
                f.write('\t- {}\n'.format(label))
        f.close()
    else:
        f = open(fclusteringpath, 'a')
        f.write('\n\nNo clustering learned - This was the set clustering:')

        for cidx, cluster in enumerate(CLUSTERING):
            f.write('Cluster {}\n'.format(cidx))
            for aidx in cluster:
                asv = ASVS.index[aidx]
                f.write('\t- {}\n'.format(pl.asvname_formatter(format=
                    asv_prefix_formatter.replace('%(index)s', ''),
                    asv=asv, asvs=ASVS)))

        f.close()
    
    # Plot the trajectories for each cluster
    logging.info('Plotting cluster trajectories')
    subjset = chain.graph.data.subjects
    df_total = subjset.df(dtype='abs', agg='mean', times='union')
    M_total = df_total.to_numpy()
    times = np.array(list(df_total.columns))
    for cidx, cluster in enumerate(CLUSTERING):
        aidxs_master = np.asarray(list(cluster.members))
        names_master = np.asarray([
            pl.asvname_formatter(format='%(name)s %(genus)s %(species)s',
                asv=asv, asvs=subjset.asvs) for asv in aidxs_master])

        n_subplots = 1 + (len(aidxs_master)//10)
        fig = plt.figure(figsize=(15, 5*n_subplots))
        
        # Plot the asvs in sets of 10 so that there are no color overlaps
        iii = 0
        subplot_idx = 1
        while iii < len(aidxs_master):
            ax = fig.add_subplot(n_subplots, 1, subplot_idx)

            # Get the indexes youre working on
            iii_end = np.min([iii+10, len(aidxs_master)])

            aidxs = aidxs_master[iii:iii_end]
            trajs = M_total[aidxs, :]
            names = names_master[iii:iii_end]

            for i in range(len(names)):
                ax.plot(times, trajs[i], label=names[i], marker='o')

            # Add in perturbations
            ax = pl.visualization.shade_in_perturbations(ax, 
                perturbations=subjset.perturbations)

            ax.set_yscale('log')
            ax.legend(bbox_to_anchor=(1.05,1))
            ax.set_ylabel('CFUs/g')
            ax.set_xlabel('Time (days)')

            subplot_idx += 1
            iii = iii_end

        fig.suptitle('Cluster {} Trajectories, Mean Abundances Over Subjects'.format(cidx))
        plt.subplots_adjust(right=0.75)
        plt.savefig(clustertrajpath + 'cluster{}.pdf'.format(cidx))
        plt.close()

    # # Calculate keystoneness
    # if calculate_keystoneness:
    #     # Get the growth rates
    #     growth_values = pl.variables.summary(
    #         chain.graph[STRNAMES.GROWTH_VALUE],
    #         only='mean', section=SECTION)['mean']
    #     si_values = pl.variables.summary(
    #         chain.graph[STRNAMES.GROWTH_VALUE],
    #         only='mean', section=SECTION)['mean']
    #     A_values = pl.variables.summary(
    #         chain.graph[STRNAMES.INTERACTIONS_OBJ], set_nan_to_0=True,
    #         section=SECTION, only='mean')['mean']
        





    #Plot interaction parameters
    interactionspath = basepath + 'interactions/'
    finteractionspath = interactionspath + 'output.txt'
    os.makedirs(interactionspath, exist_ok=True)
    f = open(finteractionspath, 'w')
    f.close()

    if chain.is_in_inference_order(STRNAMES.PRIOR_VAR_INTERACTIONS):
        f = open(finteractionspath, 'a')
        pv = chain.graph[STRNAMES.PRIOR_VAR_INTERACTIONS]
        summary = pl.variables.summary(pv, section=SECTION)
        f.write('\n\n###################################\n')
        f.write('Prior Variance Interactions\n')
        f.write('###################################\n')
        for key,ele in summary.items():
            f.write('{}: {}\n'.format(key,ele))
        ax1, _ = pl.visualization.render_trace(var=pv, plt_type='both', log_scale=True, 
            section=SECTION, include_burnin=True, rasterized=True)

        l,h = ax1.get_xlim()
        xs = np.arange(l,h,step=(h-l)/100) 
        ys = []
        for x in xs:
            ys.append(pl.random.sics.pdf(value=x, 
                dof=pv.prior.dof.value,
                scale=pv.prior.scale.value))
        ax1.plot(xs, ys, label='prior', alpha=0.5, color='red')
        ax1.legend()

        fig = plt.gcf()
        fig.suptitle('Prior Variance Interactions, {}'.format(
            LATEXNAMES.PRIOR_VAR_INTERACTIONS))
        plt.savefig(interactionspath + 'var.pdf')
        plt.close()
        f.close()

    if chain.is_in_inference_order(STRNAMES.PRIOR_MEAN_INTERACTIONS):
        f = open(finteractionspath, 'a')
        mean = chain.graph[STRNAMES.PRIOR_MEAN_INTERACTIONS]
        summary = pl.variables.summary(mean, section=SECTION)
        f.write('\n\n###################################\n')
        f.write('Prior Mean Interactions\n')
        f.write('###################################\n')
        for key,ele in summary.items():
            f.write('{}: {}\n'.format(key,ele))
        ax1, _ = pl.visualization.render_trace(var=mean, plt_type='both', 
            log_scale=True, section=SECTION, include_burnin=True, rasterized=True)

        l,h = ax1.get_xlim()
        xs = np.arange(l,h,step=(h-l)/100) 
        ys = []
        for x in xs:
            ys.append(pl.random.normal.pdf(value=x, 
                mean=mean.prior.mean.value,
                std=np.sqrt(mean.prior.var.value)))
        ax1.plot(xs, ys, label='prior', alpha=0.5, color='red')
        ax1.legend()

        fig = plt.gcf()
        fig.suptitle('Prior Mean Interactions, {}'.format(
            LATEXNAMES.PRIOR_MEAN_INTERACTIONS))
        plt.savefig(interactionspath + 'mean.pdf')
        plt.close()
        f.close()

    if chain.tracer.is_being_traced(STRNAMES.INTERACTIONS_OBJ):
        interactions = chain.graph[STRNAMES.INTERACTIONS_OBJ]
        f = open(finteractionspath, 'a')

        # Plot the interactions
        summary = pl.variables.summary(interactions, set_nan_to_0=True, section=SECTION)
        for key, arr in summary.items():
            try:
                pl.visualization.render_interaction_strength(
                    interaction_matrix=arr, log_scale=yscale_log, asvs=ASVS,
                    clustering=CLUSTERING, yticklabels=yticklabels, include_tick_marks=False,
                    xticklabels=xticklabels, include_colorbar=True, center_colors=center_color_for_strength,
                    title='{} {}'.format(key.capitalize(), LATEXNAMES.CLUSTER_INTERACTION_VALUE),
                    order=ASV_ORDER)
                fig = plt.gcf()
                fig.tight_layout()
                plt.savefig(interactionspath + '{}_matrix.pdf'.format(key.replace(' ','_')))
                plt.close()
            except:
                logging.warning('Failed plotting {}'.format(key))

        bayes_factors = interactions.generate_bayes_factors_posthoc(
            prior=chain.graph[STRNAMES.CLUSTER_INTERACTION_INDICATOR].prior, section=SECTION)
        try:
            pl.visualization.render_bayes_factors(
                bayes_factors=bayes_factors, asvs=ASVS, clustering=CLUSTERING, 
                xticklabels=xticklabels, max_value=25, yticklabels=yticklabels, 
                include_tick_marks=False, order=ASV_ORDER)
        except:
            logging.critical('Failed plotting Bayes factor')
        fig = plt.gcf()
        fig.tight_layout()
        plt.savefig(interactionspath+'bayes_factors.pdf')
        plt.close()
        
        # Calculate the in/out degree for each ASV
        inout_dict = interactions.generate_in_out_degree_posthoc(section=SECTION)
        f.write('\nIn degree for each ASV\n')
        f.write(  '----------------------\n')
        summary = pl.variables.summary(inout_dict['in'])
        for key,arr in summary.items():
            f.write('{}\n'.format(key))
            for idx,ele in enumerate(arr):
                prefix = ''
                if asv_prefix_formatter is not None:
                    prefix = pl.asvname_formatter(
                        format=asv_prefix_formatter,
                        asv=ASVS.names.order[idx],
                        asvs=ASVS)
                f.write('\t' + prefix + '{}\n'.format(ele))

        f.write('\nOut degree for each ASV\n')
        f.write(  '-----------------------\n')
        summary = pl.variables.summary(inout_dict['out'])
        for key,arr in summary.items():
            f.write('{}\n'.format(key))
            for idx,ele in enumerate(arr):
                prefix = ''
                if asv_prefix_formatter is not None:
                    prefix = pl.asvname_formatter(
                        format=asv_prefix_formatter,
                        asv=ASVS.names.order[idx],
                        asvs=ASVS)
                f.write('\t' + prefix + '{}\n'.format(ele))
        f.close()

        # Plot the in degree
        inout_dict = interactions.generate_in_out_degree_posthoc(section='entire')
        inpath = interactionspath + 'in_degree/'
        os.makedirs(inpath, exist_ok=True)
        in_arr = inout_dict['in']
        for oidx in range(len(ASVS)):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            if SECTION == 'burnin':
                n_burnin = len(in_arr)
            else:
                n_burnin = chain.burnin
            ax = pl.visualization.render_trace(var=in_arr, idx=oidx, ax=ax, 
                plt_type='trace', include_burnin=True, scatter=True, n_burnin=n_burnin, 
                alpha=0.5, title='In-degree\n{}'.format(pl.asvname_formatter(
                    format=asv_prefix_formatter,
                    asv=ASVS.names.order[oidx],
                    asvs=ASVS)), rasterized=True)
            plt.savefig(inpath + '{}.pdf'.format(ASVS[oidx].name))
            plt.close()

        mean_in = np.mean(a=in_arr, axis=1)
        # mean_low = np.quantile(a=in_arr, axis=1, q=.25)
        # mean_high = np.quantile(a=in_arr, axis=1, q=.75)
        # xs = np.arange(len(mean_high)) - chain.burnin

        fig = plt.figure()
        ax = fig.add_subplot(111)
        if SECTION == 'burnin':
            n_burnin = len(mean_in)
        else:
            n_burnin = chain.burnin
        ax = pl.visualization.render_trace(var=mean_in, ax=ax, 
                plt_type='trace', include_burnin=True, scatter=True, n_burnin=n_burnin, 
                alpha=0.90, title='Mean In-Degree', rasterized=True)
        # ax.fill_between(xs, y1=mean_low, y2=mean_high, alpha=0.15, color='blue')
        plt.savefig(inpath + 'mean.pdf')
        plt.close()

        # Plot out degree
        inout_dict = interactions.generate_in_out_degree_posthoc(section='entire')
        outpath = interactionspath + 'out_degree/'
        os.makedirs(outpath, exist_ok=True)
        out_arr = inout_dict['out']
        for oidx in range(len(ASVS)):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            if SECTION == 'burnin':
                n_burnin = len(out_arr)
            else:
                n_burnin = chain.burnin
            ax = pl.visualization.render_trace(var=out_arr, idx=oidx, ax=ax, 
                plt_type='trace', include_burnin=True, scatter=True, n_burnin=n_burnin, 
                alpha=0.5, title='Out-degree\n{}'.format(pl.asvname_formatter(
                    format=asv_prefix_formatter,
                    asv=ASVS.names.order[oidx],
                    asvs=ASVS)), rasterized=True)
            plt.savefig(outpath + '{}.pdf'.format(ASVS[oidx].name))
            plt.close()

        mean_out = np.mean(a=out_arr, axis=1)
        # mean_low = np.quantile(a=out_arr, axis=1, q=.25)
        # mean_high = np.quantile(a=out_arr, axis=1, q=.75)
        # xs = np.arange(len(mean_high)) - chain.burnin

        fig = plt.figure()
        ax = fig.add_subplot(111)
        if SECTION == 'burnin':
            n_burnin = len(mean_out)
        else:
            n_burnin = chain.burnin
        ax = pl.visualization.render_trace(var=mean_out, ax=ax, 
                plt_type='trace', include_burnin=True, scatter=True, n_burnin=n_burnin, 
                alpha=0.90, title='Mean Out-Degree', rasterized=True)
        # ax.fill_between(xs, y1=mean_low, y2=mean_high, alpha=0.15, color='blue')
        plt.savefig(outpath + 'mean.pdf')
        plt.close()

    if chain.is_in_inference_order(STRNAMES.INDICATOR_PROB):
        pi_z = chain.graph[STRNAMES.INDICATOR_PROB]
        f = open(finteractionspath, 'a')

        f.write('\n\n###################################\n')
        f.write('Indicator Probability\n')
        f.write('###################################\n')
        summary = pl.variables.summary(pi_z, section=SECTION)
        for key,val in summary.items():
            f.write('{}: {}\n'.format(key,val))
        pl.visualization.render_trace(var=pi_z, plt_type='both',
            section=SECTION, include_burnin=True, rasterized=True)
        fig = plt.gcf()
        fig.suptitle('Probability of an Interaction, {}'.format(
            LATEXNAMES.INDICATOR_PROB))
        plt.savefig(interactionspath + 'indicator_prob_trace.pdf')
        plt.close()
        f.close()

    if chain.is_in_inference_order(STRNAMES.PROCESSVAR):
        if params.DATA_LOGSCALE:
            f = open(overviewpath, 'a')
            pv = chain.graph[STRNAMES.PROCESSVAR]
            summary = pl.variables.summary(pv, section=SECTION)
            f.write('\n\n###################################\n')
            f.write('Process Variance\n')
            f.write('###################################\n')
            for key,ele in summary.items():
                f.write('{}: {}\n'.format(key,ele))
            ax1, _ = pl.visualization.render_trace(var=pv, plt_type='both',
                section=SECTION, include_burnin=True, log_scale=True, rasterized=True)

            l,h = ax1.get_xlim()
            try:
                xs = np.arange(l,h,step=(h-l)/100) 
                ys = []
                for x in xs:
                    # This might throw an overflow error
                    # print('\nvalue: {}\ndof: {}\nscale: {}'.format( 
                    #     x, pv.prior.dof.value,
                    #     pv.prior.scale.value))
                    ys.append(pl.random.sics.pdf(value=x, 
                        dof=pv.prior.dof.value,
                        scale=pv.prior.scale.value))
                ax1.plot(xs, ys, label='prior', alpha=0.5, color='red')
                ax1.legend()
            except OverflowError:
                logging.critical('OverflowError while plotting prior for ' \
                    'process variance')
            

            fig = plt.gcf()
            fig.suptitle('Process Variance, {}'.format(LATEXNAMES.PROCESSVAR))
            plt.savefig(basepath + 'processvar.pdf')
            plt.close()
            f.close()
        else:
            logging.info('NOT IMPLEMENTED FOR DATA NOT LOGSCALE')

    # Perturbations
    if STRNAMES.PERT_VALUE in chain.graph:
        for pidx in range(len(subjset.perturbations)):
            pert = subjset.perturbations[pidx]
            if pert.name is None:
                pname = 'pert{}'.format(pidx)
            else:
                pname = pert.name

            perturbation_path = basepath + '{} perturbation/'.format(pname)
            os.makedirs(perturbation_path, exist_ok=True)
            f = open(perturbation_path + 'output.txt', 'w')

            perturbation = chain.graph.perturbations[pidx]

            f.write('################################\n')
            f.write('Perturbation {}\n'.format(pidx))
            f.write('\tStart: {}\n'.format(perturbation.start))
            f.write('\tEnd: {}\n'.format(perturbation.end))
            f.write('\tLearn magnitude? {}\n'.format(
                chain.is_in_inference_order(STRNAMES.PERT_VALUE)))
            f.write('\tLearn indicator? {}\n'.format(
                chain.is_in_inference_order(STRNAMES.PERT_INDICATOR)))
            f.write('\tLearn probability? {}\n'.format(
                chain.is_in_inference_order(STRNAMES.PERT_INDICATOR_PROB)))
            f.write('\tLearn magnitude prior variance? {}\n'.format( 
                chain.is_in_inference_order(STRNAMES.PRIOR_VAR_PERT)))
            f.write('\tLearn magnitude prior mean? {}\n'.format( 
                chain.is_in_inference_order(STRNAMES.PRIOR_MEAN_PERT)))
            f.write('################################\n')

            if chain.is_in_inference_order(STRNAMES.PERT_INDICATOR_PROB):
                
                f.write('\n\nProbability\n')
                prob_sum = pl.variables.summary(
                    perturbation.probability.get_trace_from_disk())
                for key,val in prob_sum.items():
                    f.write('\t{}: {}\n'.format(key,val))

                ax1, _ = pl.visualization.render_trace(
                    var=perturbation.probability, plt_type='both', rasterized=True)
                fig = plt.gcf()
                fig.suptitle('{} perturbation probability'.format(pname))
                plt.savefig(perturbation_path+'probability.pdf')
                plt.close()

            if chain.is_in_inference_order(STRNAMES.PRIOR_VAR_PERT):
                f.write('\n\nMagnitude Prior Variance\n')
                var_sum = pl.variables.summary( 
                    perturbation.magnitude.prior.var, section=SECTION)
                for key,val in var_sum.items():
                    f.write('\t{}: {}\n'.format(key,val))
                
                ax1, _ = pl.visualization.render_trace( 
                    var=perturbation.magnitude.prior.var, plt_type='both',
                        section=SECTION, include_burnin=True, log_scale=True, rasterized=True)
                fig = plt.gcf()
                fig.suptitle('{} perturbation magnitude prior variance'.format(pname))
                plt.savefig(perturbation_path+'prior_var.pdf')
                plt.close()

                prior_std_trace = np.sqrt(perturbation.magnitude.prior.var.get_trace_from_disk( 
                    section=SECTION))
            else:
                prior_std_trace = None

            if chain.is_in_inference_order(STRNAMES.PRIOR_MEAN_PERT):
                f.write('\n\nMagnitude Prior Mean\n')
                var_sum = pl.variables.summary( 
                    perturbation.magnitude.prior.mean, section=SECTION)
                for key,val in var_sum.items():
                    f.write('\t{}: {}\n'.format(key,val))
                
                ax1, _ = pl.visualization.render_trace( 
                    var=perturbation.magnitude.prior.mean, plt_type='both',
                    section=SECTION, include_burnin=True, rasterized=True)
                fig = plt.gcf()
                fig.suptitle('{} perturbation magnitude prior mean'.format(pname))
                plt.savefig(perturbation_path+'prior_mean.pdf')
                plt.close()

                prior_mean_trace = np.sqrt(perturbation.magnitude.prior.mean.get_trace_from_disk( 
                    section=SECTION))
            else:
                prior_mean_trace = None

            # Create the histogram for the prior if any of the priors were learned
            if prior_std_trace is not None or prior_mean_trace is not None:
                if prior_std_trace is None:
                    prior_std_trace = np.sqrt(perturbation.magnitude.prior.var.value) * \
                        np.ones(LEN_POSTERIOR, dtype=float)
                if prior_mean_trace is None:
                    prior_mean_trace = perturbation.magnitude.prior.mean.value * \
                        np.ones(LEN_POSTERIOR, dtype=float)

                prior_hist = np.zeros(len(prior_std_trace), dtype=float)
                for i in range(len(prior_hist)):
                    prior_hist[i] = pl.random.normal.sample(mean=prior_mean_trace[i], std=prior_std_trace[i])
            else:
                prior_hist = None
            

            perturbation_trace = perturbation.get_trace_from_disk(section=SECTION)
            for oidx in range(len(ASVS)):
                f.write('\n\nASV - {}:\n'.format(oidx))
                f.write('---------------\n')

                try:
                    # This will fail if it was never turned on (always np.nan)
                    ax_posterior, ax_trace = pl.visualization.render_trace(
                        var=perturbation, idx=oidx, plt_type='both', section=SECTION,
                        include_burnin=True, rasterized=True)
                    left,right = ax_posterior.get_xlim()

                    if ax_posterior is not None:
                        # Plot the prior
                        if prior_hist is not None:
                            pl.visualization.render_trace(var=prior_hist, plt_type='hist',
                                label='prior', color='red', alpha=0.5, ax=ax_posterior, rasterized=True)
                        else:
                            l,h = ax_posterior.get_xlim()
                            xs = np.arange(l,h,step=(h-l)/100)
                            prior = perturbation.magnitude.prior
                            ys = []
                            for x in xs:
                                ys.append(prior.pdf(value=x))
                            ax_posterior.plot(xs, ys, label='prior', alpha=0.5, color='red')

                        if syndata is not None:
                            true_pert = syndata.dynamics.perturbations[pidx].item_array(only_pos_ind=False)[oidx]
                            ax_posterior.axvline(x=true_pert, color='red', alpha=0.65, label='True Value')
                        ax_posterior.legend()

                        if syndata is not None:
                            true_pert = syndata.dynamics.perturbations[pidx].item_array(only_pos_ind=False)[oidx]
                            ax_trace.axhline(y=true_pert, color='red', alpha=0.65, label='True Value')
                            ax_trace.legend()
                    
                    ax_posterior.set_xlim(left=left*0.8, right=right*1.2)
                except:
                    logging.critical('Perturbation `{}` for {} was empty (all np.nan or 0s). ' \
                        'Skipping'.format(pname, ASVS[oidx].name))

                fig = plt.gcf()
                if asv_prefix_formatter is not None:
                    asvname = pl.asvname_formatter(
                        format=asv_prefix_formatter,
                        asv=ASVS.names.order[oidx],
                        asvs=ASVS)
                else:
                    asvname = ASVS.names.order[oidx]
                fig.suptitle('{} perturbation magnitude\n{}'.format(
                    pname, asvname))
                plt.savefig(perturbation_path+'{}.pdf'.format(ASVS[oidx].name))
                plt.close()
                pert_sum = pl.variables.summary(perturbation_trace[:,oidx], 
                    set_nan_to_0=True)

                for key,val in pert_sum.items():
                    f.write('\t{}: {}\n'.format(key,val))

                if chain.is_in_inference_order(STRNAMES.PERT_INDICATOR):
                    # Calculate bayes factor
                    try:
                        ind_sum = perturbation_bayes_factor(perturbation, oidx)
                        f.write('\tbayes factor: {}\n'.format(ind_sum))
                    except:
                        logging.critical('Cannot calculate a bayes factor for perturbation without ' \
                            'a prior on the probability ')
                        f.write('\tbayes factor: NA\n')
            f.close()

    if chain.is_in_inference_order(STRNAMES.FILTERING):
        # plot the latent and auxiliary trajectory with the data
        # Put in the ground truth data if it is available
        filtering_path = basepath + 'filtering/'
        os.makedirs(filtering_path, exist_ok=True)

        for ridx in range(chain.graph.data.n_replicates):
            latent_name = STRNAMES.LATENT_TRAJECTORY + '_ridx{}'.format(ridx)
            aux_name = STRNAMES.AUX_TRAJECTORY + '_ridx{}'.format(ridx)

            if exact_filename is not None:
                M_truth = exact_subjset.iloc(ridx).matrix()['abs']

            given_data = subjset.iloc(ridx).matrix()['abs']
            subj = subjset.iloc(ridx)
            

            replicate_path = filtering_path + 'subject{}/'.format(subj.name)
            os.makedirs(replicate_path, exist_ok=True)
            f = open(replicate_path + 'output_subject{}.txt'.format(subj.name), 'w')
            f_accept = open(replicate_path + 'accept_reject{}.txt'.format(subj.name), 'w')

            f.write('#########################\n')
            f.write('Subject {}\n'.format(subj.name))
            f.write('#########################\n')

            try:
                aux = chain.graph[aux_name]
            except:
                aux = None
            latent = chain.graph[latent_name]

            # init_aux = np.array(aux.initialization_value)
            # init_latent = np.array(latent.initialization_value)

            if aux is not None:
                aux_trace_master = aux.get_trace_from_disk(section=SECTION)
            latent_trace_master = latent.get_trace_from_disk(section=SECTION)

            times = chain.graph.data.times[ridx]
            given_times = chain.graph.data.given_timepoints[ridx]

            # Get the minimum trajectory
            if fparams is not None:
                f_dtype = fparams.DTYPE
                f_threshold = fparams.THRESHOLD
                min_traj = []
            qpcr_means = []
            read_depths = []
            for t in given_times:
                Q_mean = subjset.iloc(ridx).qpcr[t].mean()
                read_depth = np.sum(subjset.iloc(ridx).reads[t])

                qpcr_means.append(Q_mean)
                read_depths.append(read_depth)
                if fparams is not None:
                    if f_dtype == 'rel':
                        min_traj.append(Q_mean * f_threshold)
                    elif f_dtype == 'abs':
                        min_traj.append(f_threshold)
                    elif f_dtype == 'raw':
                        min_traj.append(Q_mean * f_threshold / read_depth)
                    else:
                        raise ValueError('`dtype` ({}) not recognized'.format(f_dtype))
            if fparams is not None:
                min_traj = np.asarray(min_traj)
                if not plot_filtering_thresh:
                    min_traj = None
            else:
                min_traj = None
            c_m = chain.graph[STRNAMES.PROCESSVAR].c_m

            qpcr_means = np.asarray(qpcr_means)

            # Plot the read depth and the qpcr means
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax2 = ax1.twinx()

            ln1 = ax1.plot(given_times, qpcr_means, marker='.', color='red', 
                label=r'$ \overline{Q} $')
            ln2 = ax2.plot(given_times, read_depths, marker='.', color='green', 
                label=r'$ \sum_i r_i $')
            fig.suptitle('qPCR and read Depth for subject {}, {}'.format(
                    subjset.iloc(ridx).name, SECTION))
            ax1.set_xlabel('Days')
            ax1.set_ylabel('CFUs/g')
            ax1.set_yscale('log')
            ax2.set_ylabel('Counts')

            pl.visualization.shade_in_perturbations(ax1, subjset.perturbations, 
                textcolor='grey', textsize=11)

            lns = [ln1[0], ln2[0]]
            labs = [l.get_label() for l in lns]
            ax1.legend(lns, labs, bbox_to_anchor=(1.1,1))
            fig.subplots_adjust(right=0.80)
            plt.savefig(replicate_path + 'subject_data.pdf')
            plt.close()

            for oidx in range(len(ASVS)):
                fig = plt.figure()

                if plot_name_filtering is not None:
                    # create the string of the taxonomy
                    title = pl.asvname_formatter(
                        format=plot_name_filtering,
                        asv=ASVS[oidx],
                        asvs=ASVS)
                    title += '\n'
                else:
                    title = ''
                title += 'Subject {}, {}'.format(subjset.iloc(ridx).name, ASVS[oidx].name)

                fig.suptitle(title)


                ax = fig.add_subplot(111)

                # get the necessary data
                data = given_data[oidx,:]
                if aux is not None:
                    aux_trace = aux_trace_master[:, oidx, :]
                else:
                    aux_trace = None
                latent_trace = latent_trace_master[:, oidx, :]

                # init_aux_traj = init_aux[oidx, :]
                # init_latent_traj = init_latent[oidx, :]

                kwargs = {'given_times':given_times, 'times':times, 'data':data, 'latent':latent_trace, 
                    'aux':aux_trace, 'yscale_log':yscale_log, 'percentile':percentile, 
                    'ax':ax, 'perturbations':chain.graph.perturbations, 'subjset': subjset, 'c_m': c_m,
                    'title':'Replicate {}, ASV index: {}'.format(ridx, oidx), 'min_traj': min_traj}
                    # 'init_aux_traj': init_aux_traj, 'init_latent_traj': init_latent_traj}
                if exact_filename is not None:
                    kwargs['truth'] = M_truth[oidx,:]
                else:
                    kwargs['truth'] = None
                plot_single_trajectory(**kwargs)

                # Color in perturbations, if necessary
                pl.visualization.shade_in_perturbations(ax, subjset.perturbations, 
                    textcolor='grey', textsize=9)

                fig.subplots_adjust(top=0.85, right=0.85)

                ax.legend(bbox_to_anchor=(1,1))
                if yscale_log:
                    bottom,_ = ax.get_ylim()
                    logging.info('BOTTOM {}'.format(bottom))
                    if bottom < 1e3:
                        ax.set_ylim(bottom=1e3)


                plt.savefig(replicate_path + '{}_traj.pdf'.format(ASVS[oidx].name))
                plt.close()

                if plot_gif_filtering:
                    plot_filtering_over_samples(chain=chain, 
                        ridx=ridx, 
                        oidx=oidx, 
                        basepath=replicate_path+'gifs/', 
                        slide=50, window=100,
                        logscale = yscale_log)


            # Record qPCR for the time point
            for tidx, timepoint in enumerate(times):
                given_tidx = chain.graph.data.data_timeindex2given_timeindex[(ridx,tidx)]

                f.write('\n\n--------------------------------\n')
                f.write('Timepoint: {}'.format(timepoint))

                f_accept.write('\n\n--------------------------------\n')
                f_accept.write('Timepoint: {}\n'.format(timepoint))

                if not np.isnan(given_tidx):
                    given_time = chain.graph.data.given_timepoints[ridx][given_tidx]
                    f.write('\tqPCR: mean {:.4E}, var: {:.4E}\n' \
                        '\tdata: {}\n'.format(
                        chain.graph.data.qpcr[ridx][given_time].mean(),
                        chain.graph.data.qpcr[ridx][given_time].var(),
                        chain.graph.data.qpcr[ridx][given_time].data))

                if aux is not None:
                    pred_qpcr_trace = np.sum(aux_trace_master[:, :, tidx], axis=1)
                else:
                    pred_qpcr_trace = np.sum(latent_trace_master[:, :, tidx], axis=1)
                s = pl.variables.summary(pred_qpcr_trace)
                for key, val in s.items():
                    f.write('\t{}: {:.4E}\n'.format(key,val))


                # record the trajectories for each ASV
                for oidx in range(len(ASVS)):

                    f.write('ASV: {}, time: {}, subject: {}\n'.format(ASVS.names.order[oidx],
                        timepoint, ridx))
                    if not np.isnan(given_tidx):
                        f.write('\tData: {}\n'.format(chain.graph.data.abs_data[ridx][oidx, given_tidx]))
                    else:
                        f.write('\tIntermediate time point\n')

                    s = pl.variables.summary(latent_trace_master[:,oidx, tidx])
                    f.write('\tLatent:\n')
                    for key,val in s.items():
                        f.write('\t\t{}: {}\n'.format(key,val))

                    if aux is not None:
                        s = pl.variables.summary(aux_trace_master[:, oidx, tidx])
                        f.write('\tAuxiliary:\n')
                        for key,val in s.items():
                            f.write('\t\t{}: {}\n'.format(key,val))
                        auxtracelocal = aux_trace_master[:, oidx, tidx]
                    
                    latenttracelocal = latent_trace_master[:, oidx, tidx]
                    acceptance_rate = pl.metropolis.acceptance_rate(
                        latenttracelocal, 0, latenttracelocal.shape[0])
                    f.write('\tAcceptance Rate: {}\n'.format(acceptance_rate))
                    # f.write('Covariance scaling: {}\n'.format())
                    f_accept.write('{} - {}\n'.format(ASVS.names.order[oidx], acceptance_rate))

            f.close()
            f_accept.close()

    else:
        logging.info('NO FILTERING')

    # qPCR parameters
    # ---------------
    qpcrbasepath = basepath + 'qpcr/'
    os.makedirs(qpcrbasepath, exist_ok=True)

    # qpcr hyperprior dof
    if chain.is_in_inference_order(STRNAMES.QPCR_DOFS):
        dofpath = qpcrbasepath + 'dofs/'
        os.makedirs(dofpath, exist_ok=True)
        dofs = chain.graph[STRNAMES.QPCR_DOFS]

        dof_traces = []

        f = open(dofpath + 'output.txt')
        for l, dof in enumerate(dofs.value):            

            summ = pl.variables.summary(dof, section=SECTION)
            dof_traces.append(dof.get_trace_from_disk())

            f.write('\n\n#########################\n')
            f.write('qPCR bucket {}\n'.format(l))
            f.write('#########################\n')
            f.write('Assigned timepoints and mean (log) abundances:\n')
            for ridx, tidx in dof.data_locs:
                t = chain.graph.data.given_timepoints[ridx][tidx]
                f.write('\t{}, {}: {}\n'.format( 
                    ridx, t, np.mean(chain.graph.data.qpcr[ridx][t].log_data)))
            f.write('Learned Degrees of Freedom:\n')
            for key,val in summ.items():
                f.write('\t{}: {}\n'.format(key,val[l]))

            ax1, ax2 = pl.visualization.render_trace(var=dof, idx=l, 
                plt_type='both', include_burnin=True, rasterized=True)

            # Plot the prior
            l,h = ax1.get_xlim()
            try:
                xs = np.arange(l,h,step=(h-l)/100) 
                ys = []
                for x in xs:
                    # This might throw an overflow error
                    # print('\nvalue: {}\ndof: {}\nscale: {}'.format( 
                    #     x, pv.prior.dof.value,
                    #     pv.prior.scale.value))
                    ys.append(pl.random.uniform.pdf(value=x, 
                        low=dof.prior.dof.value,
                        high=dof.prior.scale.value))
                ax1.plot(xs, ys, label='prior', alpha=0.5, color='red')
                ax1.legend()
            except OverflowError:
                logging.critical('OverflowError while plotting prior for ' \
                    'process variance')

            fig = plt.gcf()
            fig.suptitle('DOF of qPCR Prior\nSubset {}'.format(l))
            plt.savefig(dofpath + '{}.pdf'.format(l))
            plt.close()
        f.close()
    else:
        dof_traces = None

    # qpcr hyperprior scale
    if chain.is_in_inference_order(STRNAMES.QPCR_SCALES):
        scalepath = qpcrbasepath + 'scales/'
        os.makedirs(scalepath, exist_ok=True)
        scales = chain.graph[STRNAMES.QPCR_SCALES]

        scale_traces = []

        f = open(scalepath + 'output.txt')
        for l, scale in enumerate(scales.value):            

            summ = pl.variables.summary(scale, section=SECTION)
            scale_traces.append(scale.get_trace_from_disk())

            f.write('\n\n#########################\n')
            f.write('qPCR bucket {}\n'.format(l))
            f.write('#########################\n')
            f.write('Assigned timepoints and mean (log) abundances:\n')
            for ridx, tidx in scale.data_locs:
                t = chain.graph.data.given_timepoints[ridx][tidx]
                f.write('\t{}, {}: {}\n'.format( 
                    ridx, t, np.mean(chain.graph.data.qpcr[ridx][t].log_data)))
            f.write('Learned Scale:\n')
            for key,val in summ.items():
                f.write('\t{}: {}\n'.format(key,val[l]))

            ax1, ax2 = pl.visualization.render_trace(var=scale, idx=l, 
                plt_type='both', include_burnin=True, rasterized=True)

            # Plot the prior
            l,h = ax1.get_xlim()
            try:
                xs = np.arange(l,h,step=(h-l)/100) 
                ys = []
                for x in xs:
                    # This might throw an overflow error
                    # print('\nvalue: {}\ndof: {}\nscale: {}'.format( 
                    #     x, pv.prior.dof.value,
                    #     pv.prior.scale.value))
                    ys.append(pl.random.sics.pdf(value=x, 
                        dof=scale.prior.dof.value,
                        scale=scale.prior.scale.value))
                ax1.plot(xs, ys, label='prior', alpha=0.5, color='red')
                ax1.legend()
            except OverflowError:
                logging.critical('OverflowError while plotting prior for ' \
                    'process variance')

            fig = plt.gcf()
            fig.suptitle('Scale of qPCR Prior\nSubset {}'.format(l))
            plt.savefig(scalepath + '{}.pdf'.format(l))
            plt.close()
        f.close()
    else:
        scale_traces = None

    # qPCR measurements
    if chain.is_in_inference_order(STRNAMES.QPCR_VARIANCES):
        # Plot the variances of each qPCR measurement
        scalepath = qpcrbasepath + 'scales/'
        os.makedirs(scalepath, exist_ok=True)
        qpcrs = chain.graph[STRNAMES.QPCR_VARIANCES]

        f = open(scalepath + 'output_subject.txt')

        # Make the prior plots
        if scale_traces is not None and dof_traces is not None:
            priors = []
            for l in range(len(scale_traces)):
                dof = dof_traces[l]
                scale = scale_traces[l]
                a = np.zeros(len(dof), dtype=float)
                for i in range(len(a)):
                    a[i] = pl.random.sics.sample(dof=dof, scale=scale)
                priors.append(a)
        else:
            priors = None


        for ridx,qpcr in enumerate(qpcrs.value):

            subj = subjset.iloc(ridx)
            replicate_path = scalepath + 'subject{}/'.format(subj.name)
            os.makedirs(replicate_path, exist_ok=True)

            f.write('\n\n#########################\n')
            f.write('Subject {}\n'.format(subj.name))
            f.write('#########################\n')

            summ = pl.variables.summary(qpcr, section=SECTION)

            for tidx in range(len(chain.graph.data.given_timepoints[ridx])):
                t = chain.graph.data.given_timepoints[ridx][tidx]
                l = qpcr.priors_idx[tidx]
                f.write('\n--------------------------\n')
                f.write('Time index {}, time {}, subset {}:\nLearned scale:\n'.format(
                    tidx, t, l))
                for key,val in summ.items():
                    f.write('\t{}: {}\n'.format(key, val[tidx]))
                f.write('Given (log) data:\n\tMeasurements: ')
                for ddd_idx, ddd in enumerate(chain.graph.data.qpcr[ridx][t].log_data):
                    if ddd_idx > 0:
                        f.write(', ')
                    f.write('{:.3E}'.format(ddd))
                f.write('\n')
                ddd_summ = pl.variables.summary(chain.graph.data.qpcr[ridx][t].log_data)
                for key,val in ddd_summ.items():
                    f.write('\t{}: {:.3E}\n'.format(key,val))

                ax1, _ = pl.visualization.render_trace(var=qpcr, idx=tidx, 
                    plt_type='both', include_burnin=True, rasterized=True)
                
                if priors is not None:
                    pl.visualization.render_trace(var=priors[l], plt_type='hist',
                        label='prior', color='red', ax=ax1, rasterized=True)

                fig = plt.gcf()
                fig.suptitle('Scale of qPCR measurement\nSubject {}, time {}'.format(
                    subj.name, t))
                plt.savefig(replicate_path + '{}.pdf'.format(tidx))
                plt.close()
 
        f.close()

    # if plot_diagnostic_variables:
    #     if len(chain.diagnostic_variables) > 0:
    #         dvpath = basepath + 'diagnostic_variables/'
    #         os.makedirs(dvpath, exist_ok=True)

    #         for varname, var in chain.diagnostic_variables.items():
    #             pl.visualization.render_trace(var=var, plt_type='both', section=SECTION, rasterized=True)
    #             fig = plt.gcf()
    #             fig.suptitle(varname)
    #             plt.savefig(dvpath + '{}.pdf'.format(varname))
    #             plt.close()

    if run_on_copy:
        shutil.rmtree(src_basepath, ignore_errors=True)

def perturbation_bayes_factor(perturbation, oidx):
    ind_prior_factor = (perturbation.probability.prior.b.value + 1) / \
        (perturbation.probability.prior.a.value + 1)

    trace = ~ np.isnan(perturbation.get_trace_from_disk()[:,oidx])
    # trace = perturbation.get_trace_from_disk()[:,oidx]
    # print(trace)
    # sys.exit()

    ind_sum = pl.variables.summary(trace)['mean']
    ind_sum = ind_sum/(1. - ind_sum)
    ind_sum *= ind_prior_factor

    return ind_sum

def plot_single_trajectory(given_times, times, data, latent, aux, truth, min_traj, 
    percentile=5., ax=None, xlabel='days', ylabel='CFUs/g', #init_aux_traj, init_latent_traj, 
    title=None, perturbations=None, vmin=None, vmax=None, yscale_log=True, tight_layout=True, 
    subjset=None, c_m=None):
    '''Plots the trajectory with the real data and the auxiliary and latent
    trajectories learned from the error model. `percentile` is the
    interval to plot the errors at.

    Parameters
    ----------
    given_times : np.ndarray((n,))
        These are the time points to plot the given timepoints at (synthetic and data)
    times : np.ndarray((n,))
        These are the points to plot the auxiliary and latent trajectories at
    data : np.ndarray((n,))
        This is the real data
    latent, aux : np.ndarray(n_samples,n)
        These are the auxiliary and latent trajectories that we learned
        with the error model
    init_aux_traj, init_latent_traj : np.ndarray((n,))
        These are the initial values of the inference
    truth : np.ndarray((n,)), None
        These are the base, true dynamics without measurement noise
        If this is real data then this is None
    min_traj : np.ndarray((n,)), None
        This is the minimum abundance that is stated in the filtering
    c_m : numeric
        This is the level of reintroduction, the term we use in our process variance
    percentile : numeric [0,100], Optional
        This is the percentile to plot the errors at
        Defualt is 5
    ax : matplotlib.pyplot.Axes, Optional
        Axis we are plotting on. If nothing is provided then we will create
        a new one where it takes up a whole figure
    xlabel, ylabel, title : str, Optional
        Labels for the axis and the title of the axis
    '''
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)


    # plot the aux, green
    if aux is not None:
        q_low = np.nanpercentile(a=aux, q=percentile, axis=0)
        q_median = np.nanpercentile(a=aux, q=50, axis=0)
        q_high = np.nanpercentile(a=aux, q=100-percentile, axis=0)
        ax.plot(times, q_median, color=AUX_COLOR, label='$q_i$', marker='.')
        # ax.plot(times, init_aux_traj, color=AUX_COLOR, label='$q_i^{(0)}$', linestyle=':',
        #     marker='.', alpha=0.55)
        ax.fill_between(times, y1=q_low, y2=q_high, color=AUX_COLOR, alpha=0.15)

    # plot the latent, red
    if latent is not None:
        x_low = np.nanpercentile(a=latent, q=percentile, axis=0)
        x_median = np.nanpercentile(a=latent, q=50, axis=0)
        x_high = np.nanpercentile(a=latent, q=100-percentile, axis=0)
        ax.plot(times, x_median, color=LATENT_COLOR, label='$x_i$', marker='.')
        # ax.plot(times, init_latent_traj, color=LATENT_COLOR, label='$x_i^{(0)}$', linestyle=':',
        #     marker='.', alpha=0.55)
        ax.fill_between(times, y1=x_low, y2=x_high, color=LATENT_COLOR, alpha=0.15)

    # Plot the minimum abundance
    if min_traj is not None:
        ax.plot(given_times, min_traj, color=MIN_TRAJ_COLOR, marker='.', 
            label= 'Filtering Thresh', #r'$ \overline{Q} \frac{\hat{c}}{\sum r_{i}}$', 
            alpha=0.5)

    # ax.axhline(y=c_m, color='brown', label='$c_m$', alpha=0.5)

    # plot the data
    ax.plot(given_times, data, color='black', marker='x', linestyle=':', 
        label='data')
    if truth is not None:
        ax.plot(given_times, truth, color='blue', marker='x', linestyle=':', label='truth')

    if vmin is not None or vmax is not None:
        ax.set_ylim(vmin, vmax)
    if yscale_log:
        ax.set_yscale('log')
        
    # if title is not None:
    #     ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    # if tight_layout:
    #     fig = plt.gcf()
    #     fig.tight_layout()

    return ax

def plot_filtering_over_samples(chain, ridx, oidx, basepath, logscale, slide=50, window=100):
    '''Plot the filtering progressively over the samples over the inference.

    Parameters
    ----------


    Returns
    -------
    Files saved in `basepath`
    '''
    if pl.isstr(chain):
        chain = pl.inference.BaseMCMC.load(chain)
    if not pl.isstr(basepath):
        raise TypeError('`basepath` ({}) must be a str'.format(type(basepath)))

    os.makedirs(basepath, exist_ok=True)
    latent_name = STRNAMES.LATENT_TRAJECTORY + '_ridx{}'.format(ridx)

    latent = chain.graph[latent_name]
    master_latent_trace = latent.get_trace_from_disk(section='entire')
    master_latent_trace = master_latent_trace[:, oidx, :]

    asvname = chain.graph.data.asvs[oidx].name

    # init_aux = np.array(aux.initialization_value)[oidx,:]
    # init_latent = np.array(latent.initialization_value)[oidx,:]
    
    times = chain.graph.data.times[ridx]
    given_times = chain.graph.data.given_timepoints[ridx]
    data = chain.graph.data.subjects.iloc(ridx).matrix()['abs'][oidx, :]
    n_samples = master_latent_trace.shape[0]

    iii = 1
    start = 0
    end = window
    filenames = []
    while end < n_samples:
        latent_trace = master_latent_trace[start:end]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        percentile=5.

        plot_single_trajectory(given_times=given_times, times=times, data=data,
            latent=latent_trace, aux=None, truth=None,
            percentile=percentile, ax=ax, yscale_log=logscale, min_traj=None)

        pl.visualization.shade_in_perturbations(ax, chain.graph.data.subjects.perturbations, 
            textcolor='grey', textsize=10)
        fig.suptitle('{}_{}'.format(asvname, iii))

        filenames.append(basepath + '{}_{}.png'.format(asvname, iii))
        plt.savefig(filenames[-1])
        plt.close()
        iii += 1
        start += slide
        end += slide

    import imageio
    gif_path = basepath + '{}.gif'.format(asvname)
    with imageio.get_writer(gif_path, mode='I') as writer:
        for fname in filenames:
            image = imageio.imread(fname)
            writer.append_data(image)

    for fname in filenames:
        os.remove(fname)

def validate(src_basepath, model, forward_sims, 
    perturbations_additive, dst_basepath=None, uniform_sample_timepoints=False,
    xticklabels='%(index)s', yticklabels='(%(name)s) %(genus)s: %(index)s', 
    yscale_log=True, percentile=25, plot_consensus_clusters=True, 
    asv_prefix_formatter='%(index)s: (%(name)s) %(genus)s ', run_on_copy=True, 
    mp=None, sim_dt=0.001, output_dt=1/8, 
    bayes_factor_cutoff=10, comparison=None, traj_error_metric=None,
    growth_error_metric=None, si_error_metric=None, interaction_error_metric=None,
    pert_error_metric=None, clus_error_metric=None, traj_fillvalue=None, lookaheads=[1,2,3,7]):
    '''
    Plot the interaction matrix, data, and forward simulate the model. 
    Forward simulate the dynamics learned in `model` with the initial conditions
    given by `validate_subjset` and see if it can recapitulate the trajectories.

    Comparison
    ----------
    If `comparison` is specified, it is an inference object with variables we are comparing our learned
    model `model` to. It is assumed to be the same kind of inference (MCMC or Maximum Likelihood)
    and that the variable names are identical. Additionally, if there is data in the `data` parameter
    of the graph, then we assume that this is the true data.

    Parameters
    ----------
    src_basepath : str
        Folder where all of the files are
    model : pylab.inference.BaseMCMC, pylab.inference.MLRR
        Either a bayesian chain or an maximum lilelihood output
    dst_basepath : str, None
        This is the folder that you want the posterior. If this is None, then 
        `dst_basepath = src_basepath + 'validation'`.
    uniform_sample_timepoints : bool
        If True, timepoints are uniformly sampled. Else they are not.
    xticklabels, yticklabels : str
        Format to set the labels
    yscale_log : bool
        If Tue, set the yscale to log
    percentile : numeric
        Percentile to plot the trajectory
    plot_consensus_clusters : bool
        If this is True then we have a plot for each of the consensus clusters together.
        Only used if `model` is a bayesian chain
    title_format : str
        This is the format to plot each of the titles of the forward simulations
    asv_prefix_formatter: str, Optional
        This is a formatter for the prefix for each ASV. For variables that
        are per ASV (like growth parameters) it is nice to get the
        attributes of the ASV like it's genus or the ASV name given by DADA
        instead of just an index. This formatter lets you do this. The format
        will be passed into `pl.asvname_formatter` and the output of
        this will be prepended to the value. Look at the documentation for
        that function to get the valid formatting options. If you do not want one 
        then set as `None`.
    run_on_copy : bool
        If True, it will copy the src data before it will plot it
    mp : int, None
        How many workers to have during the forward simulation. If None then there
        is no parallelization
    sim_dt : float
        This is the time step that we integrate our system at.
    output_dt : numeric
        This is the step size for the output of the forward simulated trajectory
    bayes_factor_cutoff : float
        This is the cutoff bayes factor for plotting the interaction matrix. Any interaction
        that has a bayes factor less than this cutoff will not be plotted
    forward_sims : list, None
        These are the types of forward simulations to do:
        'sim-full'
            This does a full simulation from start to finish of the subject
        (start, n_days)
            if a tuple, then we assume that the start time is the first index and
            the number of days is the second index.
            `start` can be:
                'perts-start'
                    Start at the start of each perturbation
                'perts-end'
                    Start at the end of each perturbation
                array
                    These are an array of start times
                numeric
                    This is the start time
    comparison : pylab.inference.BaseModel, None
        This is the system we are comparing to. If nothing is specified then we do not do
        any comparisons. If this is specified, we assume this to be the true abundances 
        (no measurement noise). Additionally, we assume that the names of the subjects are 
        the same that are in the validation subjset.
    traj_fillvalue : float, None
        If not None, replace 0s in the trajectory with this number when doing metrics as to not have 0s 
        and not throw an error.
    *error_metric : callable, str
        These are the error metrics for different types:
            `traj_error_metric`: for the trajectories
            `growth_error_metric`: over the growth rates
            `si_error_metric`: over the self-interactions
            `interaction_error_metric`: over the interactions
            `pert_error_metric`: over the perturbations
            `clus_error_metric`: over the cluster assignments
    '''
    # Initialize paths
    if not pl.isbool(uniform_sample_timepoints):
        raise TypeError('`uniform_sample_timepoints` ({}) must be a bool'.format(
            type(uniform_sample_timepoints)))
    if not pl.isstr(src_basepath):
        raise TypeError('`src_basepath` ({}) must be a str'.format(type(src_basepath)))
    if not pl.ismodel(model):
        raise TypeError('`model` ({}) must be a pylab.inference.BaseModel'.format(type(model)))
    if not os.path.isdir(src_basepath):
        raise ValueError('`src_basepath` ({}) does not exist')
    if src_basepath[-1] != '/':
            src_basepath = src_basepath + '/'
    if dst_basepath is None:
        dst_basepath = src_basepath + 'validation/'
    else:
        if not pl.isstr(dst_basepath):
            raise TypeError('`dst_basepath` ({}) must be a str'.format(type(dst_basepath)))
        os.makedirs(dst_basepath, exist_ok=True)
    if dst_basepath[-1] != '/':
            dst_basepath = dst_basepath + '/'

    if comparison is not None:
        if not pl.ismodel(comparison):
            raise TypeError('`comparison` ({}) must be a pylab.inference.BaseModel'.format(
                type(comparison)))
        if type(comparison) != type(model):
            raise ValueError('`comparison` ({}) amd `model` ({}) must be the same inference type'.format(
                type(comparison), type(model)))
        comparison_results = {}

        # if comparison is specified, check the metrics
        if traj_error_metric is None:
            raise ValueError('if `comparison` is not None, then you need to specify the traj metric')
        if not callable(traj_error_metric):
            if not pl.isstr(traj_error_metric):
                raise TypeError('`traj_error_metric` ({}) must either be a callable or a str'.format(
                    type(traj_error_metric)))
            if traj_error_metric == 'pe':
                traj_error_metric = pl.metrics.PE
            elif traj_error_metric == 'rmse':
                traj_error_metric = pl.metrics.RMSE
            else:
                raise ValueError('`traj_error_metric` ({}) not recognized'.format(traj_error_metric))

        if growth_error_metric is None:
            raise ValueError('if `comparison` is not None, then you need to specify the param metric')
        if not callable(growth_error_metric):
            if not pl.isstr(growth_error_metric):
                raise TypeError('`growth_error_metric` ({}) must either be a callable or a str'.format(
                    type(growth_error_metric)))
            if growth_error_metric == 'pe':
                growth_error_metric = pl.metrics.PE
            elif growth_error_metric == 'rmse':
                growth_error_metric = pl.metrics.RMSE
            else:
                raise ValueError('`growth_error_metric` ({}) not recognized'.format(growth_error_metric))

        if si_error_metric is None:
            raise ValueError('if `comparison` is not None, then you need to specify the param metric')
        if not callable(si_error_metric):
            if not pl.isstr(si_error_metric):
                raise TypeError('`si_error_metric` ({}) must either be a callable or a str'.format(
                    type(si_error_metric)))
            if si_error_metric == 'pe':
                si_error_metric = pl.metrics.PE
            elif si_error_metric == 'rmse':
                si_error_metric = pl.metrics.RMSE
            else:
                raise ValueError('`si_error_metric` ({}) not recognized'.format(si_error_metric))

        if si_error_metric is None:
            raise ValueError('if `comparison` is not None, then you need to specify the param metric')
        if not callable(si_error_metric):
            if not pl.isstr(si_error_metric):
                raise TypeError('`si_error_metric` ({}) must either be a callable or a str'.format(
                    type(si_error_metric)))
            if si_error_metric == 'pe':
                si_error_metric = pl.metrics.PE
            elif si_error_metric == 'rmse':
                si_error_metric = pl.metrics.RMSE
            else:
                raise ValueError('`si_error_metric` ({}) not recognized'.format(si_error_metric))

        if interaction_error_metric is None:
            raise ValueError('if `comparison` is not None, then you need to specify the param metric')
        if not callable(interaction_error_metric):
            if not pl.isstr(interaction_error_metric):
                raise TypeError('`interaction_error_metric` ({}) must either be a callable or a str'.format(
                    type(interaction_error_metric)))
            if interaction_error_metric == 'pe':
                interaction_error_metric = pl.metrics.PE
            elif interaction_error_metric == 'rmse':
                interaction_error_metric = pl.metrics.RMSE
            else:
                raise ValueError('`interaction_error_metric` ({}) not recognized'.format(interaction_error_metric))

        if pert_error_metric is None:
            raise ValueError('if `comparison` is not None, then you need to specify the param metric')
        if not callable(pert_error_metric):
            if not pl.isstr(pert_error_metric):
                raise TypeError('`pert_error_metric` ({}) must either be a callable or a str'.format(
                    type(pert_error_metric)))
            if pert_error_metric == 'pe':
                pert_error_metric = pl.metrics.PE
            elif pert_error_metric == 'rmse':
                pert_error_metric = pl.metrics.RMSE
            else:
                raise ValueError('`pert_error_metric` ({}) not recognized'.format(pert_error_metric))

        if pl.isMCMC(model):
            if clus_error_metric is None:
                raise ValueError('if `comparison` is not None, then you need to specify the clustering metric')
            if not callable(clus_error_metric):
                if not pl.isstr(clus_error_metric):
                    raise TypeError('`clus_error_metric` ({}) must either be a callable or a str'.format(
                        type(clus_error_metric)))
                if clus_error_metric == 'vi':
                    clus_error_metric = pl.metrics.variation_of_information
                else:
                    raise ValueError('`clus_error_metric` ({}) not recognized'.format(clus_error_metric))

    if run_on_copy:
        cpy_basepath = src_basepath[:-1] + '_copy/'
        copy_basepath(basepath=src_basepath, copy_path=cpy_basepath, copy_dirs=False)
        src_basepath = cpy_basepath

    val_subjset = pl.SubjectSet.load(src_basepath + config.VALIDATION_SUBJSET_FILENAME)
    subjset = pl.SubjectSet.load(src_basepath + config.SUBJSET_FILENAME)
    try:
        fparams = config.FilteringConfig.load(src_basepath + config.FPARAMS_FILENAME)
    except:
        fparams = None
    basepath = dst_basepath
    results_filename = basepath + 'results.pkl'

    # if pl.isMCMC(model):
    #     if model.sample_iter < 100:
    #         logging.critical('There are too few samples ({} samples)'.format(
    #             model.sample_iter))
    #         return

    if os.path.isdir(basepath):
        shutil.rmtree(basepath, ignore_errors=True)
    os.makedirs(basepath, mode=0o777, exist_ok=True)

    if comparison is not None:
        errorpath = basepath + 'errors/'
        os.makedirs(errorpath, exist_ok=True)

    ASVS = subjset.asvs
    ASV_ORDER = None
    CLUSTERING = None
    # If clustering was learned, set the consensus clusters as the value
    if comparison is not None:
        if pl.isMCMC(comparison):
            clustering = comparison.graph[STRNAMES.CLUSTERING_OBJ]
            asvorder = []
            for cluster in clustering:
                for oidx in cluster.members:
                    asvorder.append(oidx)
        else:
            if comparison.graph.data.data is not None:
                asvorder = []
                for oname in comparison.graph.data.asvs.names:
                    asvorder.append(oname)
        ASV_ORDER = []
        for i in range(len(asvorder)):
            ASV_ORDER.append(ASVS[asvorder[i]].idx)

    if pl.isMCMC(model):
        CLUSTERING = model.graph[STRNAMES.CLUSTERING_OBJ]
        if comparison is None:
            try:
                CLUSTERING.generate_cluster_assignments_posthoc(n_clusters='mean', set_as_value=True)
                asvorder = []
                for cluster in CLUSTERING:
                    for oidx in cluster.members:
                        asvorder.append(oidx)
                ASV_ORDER = asvorder
            except:
                # Do nothing because we did not learn a clustering or it is a maximum likelihood model
                pass

    # Plot the data
    subjplotpath = basepath + 'base_data/'
    os.makedirs(subjplotpath, exist_ok=True)

    matrixes = [subj.matrix()['abs'] for subj in subjset]
    read_depthses = [subj.read_depth() for subj in subjset]
    qpcrses = [np.sum(subj.matrix()['abs'], axis=0) for subj in subjset]

    # Plot the subject data in a pool
    logging.info('Starting plotting base data')
    for asv in subjset.asvs:
        fig = plt.figure(figsize=(20,10))
        fig = filtering.plot_asv(
            subjset=subjset, asv=asv, fparams=fparams, fig=fig,
            legend=True, title_format='Subject %(sname)s',
            suptitle_format='%(name)s\n%(order)s, %(family)s, %(genus)s',
            yscale_log=True, matrixes=matrixes, read_depthses=read_depthses, 
            qpcrses=qpcrses)
        plt.savefig(subjplotpath + '{}.pdf'.format(asv.name))
        plt.close()

    # Plot the learned parameters
    logging.info('Start plotting parameters')
    if pl.isMCMC(model):
        growth = pl.variables.summary(
            model.graph[STRNAMES.GROWTH_VALUE])['mean']
        self_interactions = pl.variables.summary(
            model.graph[STRNAMES.SELF_INTERACTION_VALUE])['mean']
        interactions = pl.variables.summary(
            model.graph[STRNAMES.INTERACTIONS_OBJ], 
            set_nan_to_0=True)['mean']
        bayes_factors = model.graph[STRNAMES.INTERACTIONS_OBJ].generate_bayes_factors_posthoc(
            prior=model.graph[STRNAMES.CLUSTER_INTERACTION_INDICATOR].prior)
    else:
        growth = model.graph[STRNAMES.GROWTH_VALUE].value
        self_interactions = model.graph[STRNAMES.SELF_INTERACTION_VALUE].value
        interactions = model.graph[STRNAMES.CLUSTER_INTERACTION_VALUE].value
    growth = growth.ravel()
    self_interactions = self_interactions.ravel()

    if pl.isMCMC(model):
        if model.tracer.is_being_traced(STRNAMES.INTERACTIONS_OBJ):
            try:
                pl.visualization.render_bayes_factors(
                    bayes_factors=bayes_factors, asvs=ASVS, clustering=CLUSTERING,
                    max_value=25, yticklabels=yticklabels, xticklabels=xticklabels,
                    order=ASV_ORDER)
            except:
                logging.critical('Failed plotting Bayes factor')
            fig = plt.gcf()
            fig.tight_layout()
            plt.savefig(basepath + 'bayes_factors.pdf')
            plt.close()

        try:
            coclusters = pl.variables.summary(
                CLUSTERING.coclusters.get_trace_from_disk())['mean']
            for i in range(coclusters.shape[0]):
                coclusters[i,i] = np.nan
            pl.visualization.render_cocluster_proportions(
                coclusters=coclusters, asvs=ASVS, clustering=CLUSTERING,
                yticklabels=yticklabels, xticklabels=xticklabels, include_tick_marks=False,
                order=ASV_ORDER)
            fig = plt.gcf()
            fig.tight_layout()
            plt.savefig(basepath + 'coclusters.pdf')
            plt.close()

        # Get coclustering error
        except:
            logging.info('No clustering')

    if pl.isMCMC(model):
        title = 'Predicted Interaction Strength, {}\nbayes factor >= {}'.format(
            LATEXNAMES.CLUSTER_INTERACTION_VALUE, bayes_factor_cutoff)
    else:
        title = 'Predicted Interaction Strength, {}'.format( 
            LATEXNAMES.CLUSTER_INTERACTION_VALUE)
    A = np.zeros(shape=interactions.shape, dtype=float)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if i == j:
                #A[i,j] = self_interactions[i]
                continue
            else:
                if pl.isMCMC(model):
                    if bayes_factors[i,j] >= bayes_factor_cutoff:
                        A[i,j] = interactions[i,j]
                else:
                    A[i,j] = interactions[i,j]

    pl.visualization.render_interaction_strength(
        interaction_matrix=A, log_scale=yscale_log,
        asvs=ASVS, clustering=CLUSTERING, vmin=None, vmax=None,
        yticklabels=yticklabels, include_tick_marks=False, xticklabels=xticklabels,
        order=ASV_ORDER, title=title)
    fig = plt.gcf()
    fig.tight_layout()
    plt.savefig(basepath+'predicted_interaction_matrix.pdf')
    plt.close()

    f = open(basepath+'results.txt', 'w')
    if comparison is not None:

        f.write('\n\nExpected Interactions\n')
        f.write('=====================\n')

        # Interactions
        if pl.isMCMC(comparison):
            A_comparison = comparison.graph[STRNAMES.INTERACTIONS_OBJ].get_datalevel_value_matrix()
        else:
            A_comparison = comparison.graph[STRNAMES.CLUSTER_INTERACTION_VALUE].value

        # Only do the interactions
        for i in range(A_comparison.shape[0]):
            A_comparison[i,i] = 0
        
        pl.visualization.render_interaction_strength(
            interaction_matrix=A_comparison, log_scale=yscale_log,
            asvs=ASVS, vmin=None, vmax=None,
            yticklabels=yticklabels, include_tick_marks=False, xticklabels=xticklabels,
            order=ASV_ORDER, title='True Interaction Strength')
        fig = plt.gcf()
        fig.tight_layout()
        plt.savefig(basepath+'comparison_interaction_matrix.pdf')
        plt.close()

        comparison_results['uniformly-sampled-timepoints'] = bool(uniform_sample_timepoints)

        if pl.isMCMC(model):
            # calculate the error over every sample, r
            A_predicted = model.graph[STRNAMES.INTERACTIONS_OBJ].get_trace_from_disk()
            A_predicted[np.isnan(A_predicted)] = 0
            comparison_results['error-interactions'] = np.zeros(A_predicted.shape[0], dtype=float)
            for i in range(A_predicted.shape[0]):
                comparison_results['error-interactions'][i] = interaction_error_metric(
                    A_comparison, A_predicted[i])

            summ = pl.variables.summary(comparison_results['error-interactions'])
            f.write('Error Type: {}\n'.format(interaction_error_metric.__name__))

            for key,val in summ.items():
                f.write('\t{}: {}\n'.format(key,val))

            # Plot the error
            fig = plt.figure()
            ax = fig.add_subplot(111)
            xs = np.arange(len(comparison_results['error-interactions']))
            ys = np.squeeze(comparison_results['error-interactions'])

            ax.plot(xs, ys, alpha=0.5)
            ax.set_yscale('log')
            ax.set_xlabel('Sample')
            ax.set_ylabel(interaction_error_metric.__name__)
            ax.set_title('Interactions {}\nmean={}'.format(interaction_error_metric.__name__, summ['mean']))
            
            fig.tight_layout()
            plt.savefig(errorpath + 'interactions.pdf')
            plt.close()
                    
        else:
            A_predicted = model.graph[STRNAMES.CLUSTER_INTERACTION_VALUE].value
            for i in range(A_predicted.shape[0]):
                A_predicted[i,i] = 0
            comparison_results['error-interactions'] = interaction_error_metric( 
                A_comparison, A_predicted)

            f.write('Error {}: {}\n'.format(interaction_error_metric.__name__, 
                comparison_results['error-interactions']))


    # Write the expcted value of growth, self-interactions, and perturbations
    f.write('\n\nExpected Growth\n')
    f.write('===============\n')
    f.write('Error Type: {}\n'.format(growth_error_metric.__name__))

    if pl.isMCMC(model):
        growth_trace = model.graph[STRNAMES.GROWTH_VALUE].get_trace_from_disk()

    if comparison is not None:
        growth_comparison = comparison.graph[STRNAMES.GROWTH_VALUE].value.ravel()

        if pl.isMCMC(model):
            comparison_results['error-growth'] = np.zeros(growth_trace.shape[0], dtype=float)
            for i in range(growth_trace.shape[0]):
                comparison_results['error-growth'][i] = growth_error_metric(growth_comparison, 
                    growth_trace[i])
            
            summ = pl.variables.summary(comparison_results['error-growth'])
            f.write('Error total: {}\n'.format(growth_error_metric.__name__))
            for key,val in summ.items():
                f.write('\t{}: {}\n'.format(key,val))

            # Plot the error
            fig = plt.figure()
            ax = fig.add_subplot(111)
            xs = np.arange(len(comparison_results['error-growth']))
            ys = np.squeeze(comparison_results['error-growth'])

            ax.plot(xs, ys, alpha=0.5)
            ax.set_xlabel('Sample')
            ax.set_ylabel(growth_error_metric.__name__)
            ax.set_title('Growths {}\nmean={}'.format(growth_error_metric.__name__, 
                summ['mean']))
            
            fig.tight_layout()
            plt.savefig(errorpath + 'growth.pdf')
            plt.close()

        else:
            res = []
            comparison_results['error-growth'] = growth_error_metric(growth_comparison, growth)

            f.write('Mean error: {}\n'.format(comparison_results['error-growth']))

        for oidx in range(len(ASVS)):
            if pl.isMCMC(model):
                a = np.zeros(growth_trace.shape[0])
                for i in range(growth_trace.shape[0]):
                    a[i] = growth_error_metric(growth_comparison[oidx], growth_trace[i,oidx])
                summ = pl.variables.summary(growth_trace[:,oidx])
                f.write('{}. Truth: {:.5f}, Mean {}: {:.5f}\n'.format(
                    pl.asvname_formatter(format=asv_prefix_formatter, asv=oidx, asvs=ASVS),
                    growth_comparison[oidx], growth_error_metric.__name__, 
                    np.mean(a)))
                for key,val in summ.items():
                    f.write('\t{}: {:.5f}\n'.format(key,val))
                
                
            else:
                res.append([growth[oidx], 
                    growth_comparison[oidx], np.absolute(growth[oidx] - growth_comparison[oidx]), 
                    growth_error_metric(growth_comparison[oidx], growth[oidx]),
                    pl.asvname_formatter(format=asv_prefix_formatter, asv=oidx, asvs=ASVS)])
                
        if pl.isML(model):
            dftemp = pd.DataFrame(columns=['Predicted', 'Truth', 'Abs Diff', 'Metric Error', 'Name'], 
                data=res)
            f.write(dftemp.to_string(formatters=['{:.5f}'.format, '{:.5f}'.format, '{:.5f}'.format, '{:.5f}'.format, '{}'.format]))
            f.write('\n')
    else:
        for oidx in range(len(ASVS)):
            f.write('{}:'.format(pl.asvname_formatter(
                format=asv_prefix_formatter, asv=oidx, asvs=ASVS)))
            if pl.isMCMC(model):
                f.write('\n')
                summ = pl.variables.summary(growth_trace[:,oidx])
                for key,val in summ.items():
                    f.write('\t{}: {:.5f}'.format(key,val))
            else:
                f.write('\t{:.5f}'.format(growth[oidx]))

    f.write('\n\nExpected self-interactions\n')
    f.write('==========================\n')
    f.write('Error Type: {}\n'.format(si_error_metric.__name__))
    if pl.isMCMC(model):
        si_trace = model.graph[STRNAMES.SELF_INTERACTION_VALUE].get_trace_from_disk()

    if comparison is not None:
        si_comparison = comparison.graph[STRNAMES.SELF_INTERACTION_VALUE].value.ravel()

        if pl.isMCMC(model):
            comparison_results['error-self-interactions'] = np.zeros(si_trace.shape[0], dtype=float)
            for i in range(si_trace.shape[0]):
                comparison_results['error-self-interactions'][i] = si_error_metric(si_comparison, 
                    si_trace[i])
            
            summ = pl.variables.summary(comparison_results['error-self-interactions'])
            f.write('Error total: {}\n'.format(si_error_metric.__name__))
            for key,val in summ.items():
                f.write('\t{}: {}\n'.format(key,val))

            # Plot the error
            fig = plt.figure()
            ax = fig.add_subplot(111)
            xs = np.arange(len(comparison_results['error-self-interactions']))
            ys = np.squeeze(comparison_results['error-self-interactions'])

            ax.plot(xs, ys, alpha=0.5)
            ax.set_yscale('log')
            ax.set_xlabel('Sample')
            ax.set_ylabel(si_error_metric.__name__)
            ax.set_title('Self-interactionss {}\nmean={}'.format(si_error_metric.__name__, 
                summ['mean']))
            
            fig.tight_layout()
            plt.savefig(errorpath + 'self_interactions.pdf')
            plt.close()

        else:
            comparison_results['error-self-interactions'] = si_error_metric(si_comparison, 
                self_interactions)

            f.write('Error total: {}\n'.format(comparison_results['error-self-interactions']))
            res = []

        for oidx in range(len(ASVS)):
            if pl.isMCMC(model):
                a = np.zeros(si_trace.shape[0])
                for i in range(si_trace.shape[0]):
                    a[i] = si_error_metric(si_comparison[oidx], si_trace[i,oidx])
                summ = pl.variables.summary(si_trace[:,oidx])
                f.write('{}. Truth: {:.5E}, Mean {}: {:.5E}\n'.format(
                    pl.asvname_formatter(format=asv_prefix_formatter, asv=oidx, asvs=ASVS),
                    si_comparison[oidx], si_error_metric.__name__, 
                    np.mean(a)))
                for key,val in summ.items():
                    f.write('\t{}: {:.5E}\n'.format(key,val))
            else:
                res.append([self_interactions[oidx], 
                    si_comparison[oidx], np.absolute(self_interactions[oidx] - si_comparison[oidx]), 
                    si_error_metric(si_comparison[oidx], self_interactions[oidx]),
                    pl.asvname_formatter(format=asv_prefix_formatter, asv=oidx, asvs=ASVS)])

        if pl.isML(model):
            dftemp = pd.DataFrame(columns=['Predicted', 'Truth', 'Abs Diff', 'Metric Error', 'Name'], 
                data=res)
            f.write(dftemp.to_string(formatters=['{:.5E}'.format, '{:.5E}'.format, '{:.5E}'.format, '{:.5f}'.format, '{}'.format]))
            f.write('\n')
    else:

        f.write('{}:'.format(pl.asvname_formatter(
            format=asv_prefix_formatter, asv=oidx, asvs=ASVS)))
        if pl.isMCMC(model):
            f.write('\n')
            summ = pl.variables.summary(si_trace[:,oidx])
            for key,val in summ.items():
                f.write('\t{}: {:.5f}'.format(key,val))
        else:
            f.write('\t{:.5f}'.format(self_interactions[oidx]))

    if model.graph.perturbations is not None:
        f.write('\n\nExpected Perturbations\n')
        f.write('======================\n')
        f.write('Error Type: {}\n'.format(pert_error_metric.__name__))
        for pidx, pred_perturbation in enumerate(model.graph.perturbations):
            if subjset.perturbations[pidx].name is not None:
                pname = subjset.perturbations[pidx].name
            else:
                pname = pidx
            f.write('\nPerturbation {}\n'.format(pname))
            f.write('------------\n')

            if pl.isMCMC(model):
                pert_trace = pred_perturbation.get_trace_from_disk()
                pert_trace[np.isnan(pert_trace)] = 0 

                if comparison is not None:
                    pert_comparison = comparison.graph.perturbations[pidx].item_array()

                    aaa = np.zeros(pert_trace.shape[0], dtype=float)
                    for i in range(len(aaa)):
                        aaa[i] = pert_error_metric(pert_comparison, pert_trace[i])
                    comparison_results['error-{}'.format(pname)] = aaa

                    summ = pl.variables.summary(comparison_results['error-{}'.format(pname)])
                    f.write('Error total: {}\n'.format(pert_error_metric.__name__))
                    for key,val in summ.items():
                        f.write('\t{}: {}\n'.format(key,val))

                    # Plot the error
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    xs = np.arange(len(comparison_results['error-{}'.format(pname)]))
                    ys = np.squeeze(comparison_results['error-{}'.format(pname)])

                    ax.plot(xs, ys, alpha=0.5)
                    ax.set_xlabel('Sample')
                    ax.set_ylabel(pert_error_metric.__name__)
                    ax.set_title('Perturbation {} {}\nmean={}'.format(pname,
                        pert_error_metric.__name__, summ['mean']))
                    
                    fig.tight_layout()
                    plt.savefig(errorpath + 'pert{}.pdf'.format(pname))
                    plt.close()

                
                for oidx in range(len(ASVS)):
                    f.write('{}:'.format(pl.asvname_formatter(
                        format=asv_prefix_formatter, asv=oidx, asvs=ASVS)))
                    summ = pl.variables.summary(pert_trace[:,oidx])
                    pert_b = perturbation_bayes_factor(
                        perturbation=pred_perturbation, oidx=oidx)
                    if comparison is not None:
                        aaa = np.zeros(pert_trace.shape[0], dtype=float)
                        for i in range(len(aaa)):
                            aaa[i] = pert_error_metric(pert_comparison[oidx], 
                                pert_trace[i,oidx])
                        mean_error = np.mean(aaa)
                        f.write(' truth: {:.5f}, mean {}: {:.5f}\n'.format(
                            pert_comparison[oidx], pert_error_metric.__name__,
                            mean_error))
                    else:
                        f.write('\n')
                    
                    for key,val in summ.items():
                        f.write('\t{}: {:.5f}\n'.format(key,val))
                    f.write('\tbayes factor: {}\n'.format(pert_b))

            else:
                pred_array = pred_perturbation.magnitude.value
                if comparison is not None:
                    perturbation_comparison = comparison.graph.perturbations[pidx].array()
                    comparison_results['error-{}'.format(pname)] = pert_error_metric(pred_array, 
                        perturbation_comparison, force_reshape=True)
                    f.write('Mean error: {}\n'.format(
                        comparison_results['error-{}'.format(pname)]))
                    res = []
                else:
                    f.write('Predicted\n')
                
                for oidx in range(len(ASVS)):
                    if comparison is not None:
                        res.append([
                            pred_array[oidx], perturbation_comparison[oidx], 
                            np.absolute(pred_array[oidx] - perturbation_comparison[oidx]),
                            pert_error_metric(perturbation_comparison[oidx], pred_array[oidx]),
                            pl.asvname_formatter(format=asv_prefix_formatter, asv=oidx, asvs=ASVS)])

                    else:
                        f.write('\t{:.5f}: {}\n'.format( 
                            pred_array[oidx],
                            pl.asvname_formatter(format=asv_prefix_formatter, asv=oidx, asvs=ASVS)))
                if comparison is not None:
                    dftemp = pd.DataFrame(columns=['Predicted', 'Truth', 'Abs Diff', 'Metric Error', 'Name'], 
                        data=res)
                    f.write(dftemp.to_string(formatters=['{:.5f}'.format, '{:.5f}'.format, '{:.5f}'.format, '{:.5f}'.format, '{}'.format]))

    # Clustering
    if pl.isMCMC(model):
        if model.is_in_inference_order(STRNAMES.CLUSTERING):
            cluster_assignments = CLUSTERING.toarray()
            f.write('\n\nConsensus cluster assignments:\n')
            f.write('==============================\n')
            for idx,lst in enumerate(cluster_assignments):
                f.write('Cluster {} - Size {}\n'.format(idx, len(lst)))
                for oidx in lst:
                    # Get rid of index because that does not really make sense here
                    label = pl.asvname_formatter(
                        format=asv_prefix_formatter.replace('%(index)s',''),
                        asv=ASVS.index[oidx],
                        asvs=ASVS)
                    f.write('\t- {}\n'.format(label))
            ca_pred = cluster_assignments
        if comparison is not None:
            cluster_assignments = comparison.graph[STRNAMES.CLUSTERING_OBJ].toarray()
            f.write('Comparison cluster assignments:\n')
            f.write('===============================\n')
            for idx,lst in enumerate(cluster_assignments):
                f.write('Cluster {} - Size {}\n'.format(idx, len(lst)))
                for oidx in lst:
                    # Get rid of index because that does not really make sense here
                    label = pl.asvname_formatter(
                        format=asv_prefix_formatter.replace('%(index)s',''),
                        asv=ASVS.index[oidx],
                        asvs=ASVS)
                    f.write('\t- {}\n'.format(label))
            ca_truth = cluster_assignments

            if model.is_in_inference_order(STRNAMES.CLUSTERING):
                cocluster_trace = CLUSTERING.coclusters.get_trace_from_disk()
                vi = np.zeros(cocluster_trace.shape[0], dtype=float)
                for i in range(cocluster_trace.shape[0]):
                    ca_pred = pl.cluster.toarray_from_cocluster(cocluster_trace[i])
                    vi[i] = clus_error_metric(ca_truth, ca_pred, n=len(ASVS))

                comparison_results['error-clustering'] = vi
                summ = pl.variables.summary(vi)
                f.write('{}\n'.format(
                    clus_error_metric.__name__.replace('_', ' ').title()))
                for key,val in summ.items():
                    f.write('\t{}: {:.5f}\n'.format(key, val))

                # Plot the error
                fig = plt.figure()
                ax = fig.add_subplot(111)
                xs = np.arange(len(comparison_results['error-clustering']))
                ys = np.squeeze(comparison_results['error-clustering'])

                ax.plot(xs, ys, alpha=0.5)
                ax.set_xlabel('Sample')
                # ax.set_yscale('log')
                ax.set_ylabel(clus_error_metric.__name__)
                ax.set_title(' {}\nmean={}'.format(
                    clus_error_metric.__name__.replace('_', ' ').title(), 
                    summ['mean']))
                
                fig.tight_layout()
                plt.savefig(errorpath + 'clustering.pdf')
                plt.close()

            else:
                f.write('Variation of Information: No Clustering\n')
                comparison_results['varitaion-of-information'] = np.nan
    else:
        if comparison is not None:
            comparison_results['varitaion-of-information'] = np.nan

    # Forward simulate for each subject in val_subjset
    logging.info('Starting forward sims')
    results = metrics.Metrics(model=model, limit_of_detection=1e5, simulation_dt=sim_dt, 
        output_dt=output_dt, log_integration=True, traj_fillvalue=traj_fillvalue,
        perturbations_additive=perturbations_additive, mp=10)
    if comparison is not None:
        results.add_truth_metrics(comparison_results)
    for subject in val_subjset:
        if comparison is not None:
            truth = comparison.graph.data.subjects[subject.name]
        else:
            truth = None
        # if pl.isML(model):
        #     results.lookahead(time_lookahead=lookaheads, subject=subject, dtype='abs', 
        #         percentile=percentile, truth=truth, error_metric=traj_error_metric)
        for sim_options in forward_sims:
            logging.info('{}-{}'.format(subject.name, sim_options))
            if sim_options == 'sim-full':
                results.sim_full(subject=subject, dtype='abs',percentile=percentile, truth=truth,
                    error_metric=traj_error_metric)
            elif type(sim_options) == tuple:
                if len(sim_options) != 2:
                    raise ValueError('`sim_option` ({}) must be length 2'.format(sim_options))
                results.sim_days(n_days=sim_options[1], start=sim_options[0], subject=subject, 
                    dtype='abs',percentile=percentile, truth=truth, 
                    error_metric=traj_error_metric)
            else:
                raise ValueError('`sim_option` ({}) not recognized'.format(sim_options))

    # Plot the ASVs
    results.plot(basepath=basepath, yscale='log', legend=True, 
        title_formatter='%(subjname)s: (%(name)s): %(genus)s\n{}: %(error)s'.format(
            traj_error_metric.__name__))

    f = open(basepath+'results.txt', 'a')
    f = results.readify(f=f, asv_format='%(name)s, %(genus)s')
    f.close()
    
    with open(results_filename, 'wb') as output:
        pickle.dump(results, output, protocol=pickle.HIGHEST_PROTOCOL)

    if run_on_copy:
        shutil.rmtree(src_basepath, ignore_errors=True)

def make_df(basepath, name, df=None):
    '''Given a basepath add the metrics, noise, and replicate informations.
    If a dataframe is passed in then we append to that dataframe

    Parameters
    ----------
    basepath : str
        This is the path to get all of the runs from
    name : str
        This is the name of the run that gets added to the data frame
    df : pandas.DataFrame, optional
        This is the already created dataframe that we would append to. If nothing
        is provided then a new dataframe is created

    Returns
    -------
    pandas.DataFrame
    '''
    MASTER_COLUMNS = ['Model', 'Error-traj', 'Simulation Type', 'Subject Name',
        'Error-interactions', 'Error-growth', 'Error-self-interactions', 
        'Process Variance', 'Measurement Noise', 'N Timepoints', 
        'Replicates', 'ASVs', 'Error-clustering', 'Uniform-Sampling-Timepoints']
    
    if df is None:
        df = pd.DataFrame(columns=MASTER_COLUMNS)

    if not pl.isstr(basepath):
        raise TypeError('`basepath` ({}) must be str'.format(type(basepath)))
    if basepath[-1] != '/':
        basepath += '/'

    lst = os.listdir(basepath)
    for iiii, model_run in enumerate(lst):
        print('{}/{}'.format(iiii, len(lst)))
        model_path = basepath + model_run

        try:
            params = config.SimulationConfig.load(model_path + '/{}'.format(config.SYNPARAMS_FILENAME))
        except:
            # This is a path that does not have a model in it
            continue
        measurement_noise = params.MEASUREMENT_NOISE_LEVEL
        process_variance = params.PROCESS_VARIANCE_LEVEL
        nt = params.TIMES
        
        metric_path = model_path + '/validation/results.pkl'
        if not os.path.isfile(metric_path):
            continue
        try:
            metric = metrics.Metrics.load(metric_path)
        except:
            print('skipping over {}'.format(model_path))
            continue

        # Get the parameters
        for subjname in metric.results:
            for simtype in metric.results[subjname]:

                temp_d = metric.results[subjname][simtype]
                truth_metrics = metric.truth_metrics

                n_replicates = metric.model.graph.data.n_replicates
                n_asvs = len(metric.model.graph.data.asvs)

                if pl.isMCMC(metric.model):
                    error_interactions = np.mean(truth_metrics['error-interactions'])
                    error_clustering = np.mean(truth_metrics['error-clustering'])
                    error_growth = np.mean(truth_metrics['error-growth'])
                    error_si = np.mean(truth_metrics['error-self-interactions'])
                    error_total = np.mean(temp_d['error-total'])
                else:
                    error_interactions = truth_metrics['error-interactions']
                    error_clustering = None
                    error_growth = truth_metrics['error-growth']
                    error_si = truth_metrics['error-self-interactions']
                    error_total = temp_d['error-total']

                arr = [[
                    name, error_total, simtype, subjname, 
                    error_interactions, error_growth, error_si,
                    process_variance, measurement_noise, nt,
                    n_replicates, n_asvs, error_clustering,
                    truth_metrics['uniformly-sampled-timepoints']]]

                dftemp = pd.DataFrame(data=arr, columns=MASTER_COLUMNS)

                df = pd.concat((df, dftemp), axis=0)

    return df

def make_boxplots(df, x, y, hue=None, only=None, yscale='linear', title=None, legend=True,**kwargs):
    '''Make a boxplot with the given pandas.DataFrame and parameters.

    All parameters go into `seaborn.boxplot` except:

    Parameters
    ----------
    only : dict, optional
        If specified, it indexes out the simulation types that you want to
        compare in isolation. Example:
            only = {'Replicates': 3}
            This will only make a boxplot of the elements in the dataframe that
            have the value 3 in the column 'Replicates'.
    title : str, optional
        Title for the axis object. If None and `only` is not None, then it will
        display the parameters in `only` in the title.

    Returns
    -------
    matplotlib.pyplot.Axes
    '''
    if only is not None:
        dftemp = df
        for col, val in only.items():
            dftemp = dftemp[dftemp[col] == val]
    else:
        dftemp = df
    ax = sns.boxplot(x=x, y=y, data=dftemp, hue=hue, **kwargs)
    ax.set_yscale(yscale)
    if title is None:
        if only is not None:
            s = None
            for key,val in only.items():
                if s is None:
                    s = '{}={}'.format(key, val)
                else:
                    s += ', {}={}'.format(key,val)
            ax.set_title(s)
    else:
        ax.set_title(title)
    if not legend:
        ax.get_legend().remove()
    else:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    return ax
    
def readify_chain_fixed_topology(src_basepath, piechart_axis_layout, healthy, 
    abund_times_start, abund_times_end, dst_basepath=None, 
    asv_prefix_formatter='%(name)s %(genus)s %(species)s', yscale_log=True, 
    minimum_bayes_factor=10, heatmaptaxlevel='order', datataxlevel='order',
    phylogenetic_tree_filename=None):
    '''Makes a humna readable printout of the graph that was learned. This is similar to
    `readify_chain` except we assume that the clustering was fixed during inference - thus
    we make condensed view of the interactions and the perturbations. We only plot the
    parameters related to the clustering - interactions, perturbations, priors.

    Parameters
    ----------
    src_basepath : str
        Folder of the inference and graph object
    piechart_axis_layout : 2-tuple(int)
        Layout for the axis on the 
    abund_times_start, abund_times_end : float
        These are the start and end times to get the mean abundance
    dst_basepath : str, None
        If Specified, this is the place to plot. If nothing is specified, then we plot
        at `src_basepath` + 'posterior_fixed_topology/'
    asv_prefix_formatter: str, Optional
        This is a formatter for the prefix for each ASV. For variables that
        are per ASV (like growth parameters) it is nice to get the
        attributes of the ASV like it's genus or the ASV name given by DADA
        instead of just an index. This formatter lets you do this. The format
        will be passed into `pl.asvname_formatter` and the output of
        this will be prepended to the value. Look at the documentation for
        that function to get the valid formatting options.
        If you do not want one then set as `None`.
    yscale_log : bool
        If True, plot everything in log-scale
    minimum_bayes_factor : numeric, None
        This is the minimum bayes factor needed to plot an interation or perturbation.
        If None, then we assume the minimum is 0.
    phylogenetic_tree_filename : str
        This is the filename of the phylogenetic tree
    '''
    # Check the parameters and setup
    # ------------------------------
    config.LoggingConfig()
    if not pl.isstr(src_basepath):
        raise TypeError('`src_basepath` ({}) must be a str'.format(type(src_basepath)))
    if not os.path.isdir(src_basepath):
        raise ValueError('`src_basepath` ({}) does not exist')
    if src_basepath[-1] != '/':
            src_basepath = src_basepath + '/'
    if dst_basepath is None:
        dst_basepath = src_basepath + 'posterior_fixed_topology/'
    else:
        if not pl.isstr(dst_basepath):
            raise TypeError('`dst_basepath` ({}) must be a str'.format(type(dst_basepath)))
        os.makedirs(dst_basepath, exist_ok=True)

    if asv_prefix_formatter is None:
        asv_prefix_formatter = ''
    if not pl.isstr(asv_prefix_formatter):
        raise TypeError('`asv_prefix_formatter` ({}) must be a str'.format(type(asv_prefix_formatter)))
    
    if not pl.isbool(yscale_log):
        raise TypeError('`yscale_log` ({}) must be a bool'.format(type(yscale_log)))
    if minimum_bayes_factor is None:
        minimum_bayes_factor = 0
    if not pl.isnumeric(minimum_bayes_factor):
        raise TypeError('`minimum_bayes_factor` ({}) must be a numeric'.format(
            type(minimum_bayes_factor)))
    if minimum_bayes_factor < 0:
        raise ValueError('`minimum_bayes_factor` ({}) must be >= 0'.format(minimum_bayes_factor))

    # Start plotting
    basepath = dst_basepath
    chain = pl.inference.BaseMCMC.load(src_basepath + config.MCMC_FILENAME)
    try:
        fparams = config.FilteringConfig.load(src_basepath + config.FPARAMS_FILENAME)
    except:
        fparams = None
    subjset = pl.SubjectSet.load(src_basepath + config.SUBJSET_FILENAME)

    # Unnormalize if necessary
    subjset, chain = unnormalize_parameters(subjset=subjset, mcmc=chain)
    subjset.save(src_basepath + config.SUBJSET_FILENAME)

    # Check if we have an entire chain, partial train, or only burnin
    if chain.sample_iter < 100:
        logging.critical('There are too few samples to find the posterior ({} samples)'.format(
            chain.sample_iter))
        return
    if chain.sample_iter > chain.burnin:
        SECTION = 'posterior'
        LEN_POSTERIOR = chain.sample_iter+1 - chain.burnin
    elif chain.sample_iter <= chain.burnin:
        SECTION = 'burnin'
        LEN_POSTERIOR = chain.sample_iter + 1


    # if the directory already exists, delete it and make a new one
    if os.path.isdir(basepath):
        shutil.rmtree(basepath, ignore_errors=True)
    os.makedirs(basepath, exist_ok=True)

    CLUSTERING = chain.graph[STRNAMES.CLUSTERING_OBJ]
    ASVS = CLUSTERING.items

    if piechart_axis_layout == 'auto':
        n_clusters = len(CLUSTERING)

        if n_clusters <= 2:
            piechart_axis_layout = (1,2)
        elif n_clusters <= 4:
            piechart_axis_layout = (2,2)
        elif n_clusters <= 6:
            piechart_axis_layout = (2,3)
        elif n_clusters <= 8:
            piechart_axis_layout = (2,4)
        elif n_clusters == 9:
            piechart_axis_layout = (3,3)
        elif n_clusters == 12:
            piechart_axis_layout = (4,3)
        elif n_clusters <= 15:
            piechart_axis_layout = (3,5)
        elif n_clusters <= 20:
            piechart_axis_layout = (5,4)
        elif n_clusters <= 25:
            piechart_axis_layout = (5,5)
        elif n_clusters <= 30:
            piechart_axis_layout = (6,5)
        elif n_clusters <= 36:
            piechart_axis_layout = (6,6)
        elif n_clusters <= 42:
            piechart_axis_layout = (6,7)
        elif n_clusters <= 49:
            piechart_axis_layout = (7,7)
        elif n_clusters <= 64:
            piechart_axis_layout = (8,8)
        else:
            raise ValueError('Cannot automatically set piechart with more than 64 clusters ({})'.format(
                len(CLUSTERING)))

    if not pl.istuple(piechart_axis_layout):
        raise TypeError('`piechart_axis_layout` ({}) must be a tuple'.format(type(
            piechart_axis_layout)))
    if len(piechart_axis_layout) != 2:
        raise ValueError('`piechart_axis_layout` ({}) must be 2'.format(
            len(piechart_axis_layout)))

    fpath = basepath + 'overview.txt'
    f = open(fpath, 'w')
    f.write('###################################\n')
    f.write('Output from chain `{}`\n'.format(chain.graph.name))
    f.write('###################################\n')
    f.write('Seed: {}\n'.format(chain.graph.seed))
    f.write('Total number of samples: {}\n'.format(chain.n_samples))
    f.write('Burnin: {}\n'.format(chain.burnin))
    f.write('Number of samples actually done: {}\n'.format(chain.sample_iter))
    f.write('Number of ASVs: {}\n'.format(chain.graph.data.asvs.n_asvs))
    f.write('Inference Order:\n')
    for i,ele in enumerate(chain.inf_order):
        f.write('\t{}\n'.format(str(chain.graph[ele].name)))
    f.write('Learned Variables:\n')
    for ele in chain.tracer.being_traced:
        f.write('\t{}\n'.format(str(ele)))

    f.write('Fixed topology, {} clusters:\n'.format(len(CLUSTERING)))
    for cidx, cluster in enumerate(CLUSTERING):
        f.write('Cluster {}\n'.format(cidx))
        for aidx in cluster:
            asv = ASVS.index[aidx]
            f.write('\t- {}\n'.format(pl.asvname_formatter(format=
                asv_prefix_formatter.replace('%(index)s', ''),
                asv=asv, asvs=ASVS)))

    f.write('\n\nPhylogeny metrics\n')
    f.write('-----------------\n')
    if phylogenetic_tree_filename is not None:
        logging.info('Adding in phylogenetic tree metrics')

        TREE = ete3.Tree(phylogenetic_tree_filename)
        names = [str(name) for name in ASVS.names.order]
        TREE.prune(names, True)


        logging.info('Precalculating pairwise phylogentic distances')
        phylo_dists = np.zeros(shape=(len(ASVS), len(ASVS)))
        for iaidx, iname in enumerate(names):
            iname = str(iname)
            for jaidx, jname in enumerate(names):
                jname = str(jname)
                if iaidx == jaidx:
                    phylo_dists[iaidx, iaidx] = np.nan
                    continue
                phylo_dists[iaidx, jaidx] = TREE.get_distance(iname, jname)

        intra_dist_total = 0
        inter_dist_total = 0
        for icidx, icluster in enumerate(CLUSTERING):
            f.write('Cluster {}\n'.format(icidx))

            # Calculate intra- and inter- distances
            intra_dist = []
            inter_dist = []
            for iidx in icluster.members:
                for jidx in icluster.members:
                    if iidx == jidx:
                        continue
                    
                    intra_dist.append(phylo_dists[iidx, jidx])

                for jidx in range(len(names)):
                    if jidx in icluster.members:
                        continue
                    inter_dist.append(phylo_dists[iidx, jidx])

            summ_intra = pl.variables.summary(intra_dist)
            summ_inter = pl.variables.summary(inter_dist)
            intra_dist_total += summ_intra['mean']
            inter_dist_total += summ_inter['mean']
            
            f.write('\tIntracluster distance:\n')
            for k,v in summ_intra.items():
                f.write('\t\t{}: {}\n'.format(k,v))

            f.write('\tIntercluster distance:\n')
            for k,v in summ_inter.items():
                f.write('\t\t{}: {}\n'.format(k,v))
            

        f.write('Mean global phylogenetic distances: {}\n'.format(np.nanmean(phylo_dists)))
        f.write('Mean inter-cluster distance: {}\n'.format(inter_dist_total/len(CLUSTERING)))
        f.write('Mean intra-cluster distance: {}\n'.format(intra_dist_total/len(CLUSTERING)))

    f.close()

    # Plot interaction parameters
    # ---------------------------
    interactionspath = basepath + 'interactions/'
    finteractionspath = interactionspath + 'output.txt'
    os.makedirs(interactionspath, exist_ok=True)
    f = open(finteractionspath, 'w')
    f.close()

    total_interactions = None
    total_interactions_asvs = None
    if chain.tracer.is_being_traced(STRNAMES.INTERACTIONS_OBJ):
        interactions = chain.graph[STRNAMES.INTERACTIONS_OBJ]
        f = open(finteractionspath, 'a')

        asv_interactions_trace = interactions.get_trace_from_disk(section=SECTION)
        cluster_interactions_trace = _condense_interactions(
            matrix=asv_interactions_trace, clustering=CLUSTERING)
        summ = pl.variables.summary(cluster_interactions_trace, set_nan_to_0=True)
        expected_interactions_asvs = pl.variables.summary(asv_interactions_trace, set_nan_to_0=True)['mean']

        yticklabels = []
        xticklabels = []
        for i in range(len(CLUSTERING)):
            yticklabels.append('Cluster{} {}'.format(i,i))
            xticklabels.append('{}'.format(i))

        # Plot the interactions
        for key, arr in summ.items():
            try:
                pl.visualization.render_interaction_strength(
                    interaction_matrix=arr, log_scale=yscale_log, asvs=None,
                    yticklabels=yticklabels, include_tick_marks=False,
                    xticklabels=xticklabels, include_colorbar=True, 
                    title='{} {}'.format(key.capitalize(), LATEXNAMES.CLUSTER_INTERACTION_VALUE))
                fig = plt.gcf()
                fig.tight_layout()
                plt.savefig(interactionspath + '{}_matrix.pdf'.format(key.replace(' ','_')))
                plt.close()
            except:
                logging.warning('Failed plotting {}'.format(key))

        bayes_factors_asvs = interactions.generate_bayes_factors_posthoc(
            prior=chain.graph[STRNAMES.CLUSTER_INTERACTION_INDICATOR].prior, section=SECTION)
        bayes_factors_clusters = _condense_interactions(bayes_factors_asvs, CLUSTERING)
        # try:
        pl.visualization.render_bayes_factors(
            bayes_factors=bayes_factors_clusters, asvs=None,
            xticklabels=xticklabels, max_value=25, yticklabels=yticklabels, 
            include_tick_marks=False)
        # except:
        #     logging.critical('Failed plotting Bayes factor')
        fig = plt.gcf()
        fig.tight_layout()
        plt.savefig(interactionspath+'bayes_factors.pdf')
        plt.close()

        # Make interactions with the minimum bayes factor
        expected_interactions = summ['mean']
        mask = bayes_factors_clusters < minimum_bayes_factor
        expected_interactions[mask] = 0

        mask_asvs = bayes_factors_asvs < minimum_bayes_factor
        expected_interactions_asvs[mask_asvs] = 0

        total_interactions = expected_interactions
        total_interactions_asvs = expected_interactions_asvs
        pl.visualization.render_interaction_strength(
            interaction_matrix=expected_interactions, log_scale=yscale_log, asvs=None,
            yticklabels=yticklabels, include_tick_marks=False,
            xticklabels=xticklabels, include_colorbar=True, 
            title='Expected Interaction strength\nBayes Factor >= {}'.format(
                minimum_bayes_factor))
        fig = plt.gcf()
        fig.tight_layout()
        plt.savefig(interactionspath+'expected_interactions.pdf')
        plt.close()

        # Calculate the in/out degree for each Cluster
        trace = interactions.get_trace_from_disk(section=SECTION)
        trace = ~np.isnan(trace)
        trace = _condense_interactions(trace, CLUSTERING)
        inout_dict = {'in': 100*np.sum(trace, axis=2)/len(CLUSTERING), 'out':100*np.sum(trace, axis=1)/len(CLUSTERING)}

        f.write('\nIn degree for each Cluster\n')
        f.write(  '--------------------------\n')
        summary = pl.variables.summary(inout_dict['in'])
        for key,arr in summary.items():
            f.write('{}\n'.format(key))
            for idx,ele in enumerate(arr):
                f.write('\tCluster {}: {}\n'.format(idx, ele))

        f.write('\nOtu degree for each Cluster\n')
        f.write(  '---------------------------\n')
        summary = pl.variables.summary(inout_dict['out'])
        for key,arr in summary.items():
            f.write('{}\n'.format(key))
            for idx,ele in enumerate(arr):
                f.write('\tCluster {}: {}\n'.format(idx, ele))
        f.close()

        # Plot the in degree
        inpath = interactionspath + 'in_degree/'
        os.makedirs(inpath, exist_ok=True)
        in_arr = inout_dict['in']
        for cidx in range(len(CLUSTERING)):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            if SECTION == 'burnin':
                n_burnin = len(in_arr)
            else:
                n_burnin = chain.burnin
            ax = pl.visualization.render_trace(var=in_arr, idx=cidx, ax=ax, 
                plt_type='trace', include_burnin=True, scatter=True, n_burnin=n_burnin, 
                alpha=0.5, title='In-degree Cluster {}'.format(cidx), rasterized=True, 
                ylabel='Degree (%)')
            plt.savefig(inpath + 'cluster{}.pdf'.format(cidx))
            plt.close()

        mean_in = np.mean(a=in_arr, axis=1)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if SECTION == 'burnin':
            n_burnin = len(mean_in)
        else:
            n_burnin = chain.burnin
        ax = pl.visualization.render_trace(var=mean_in, ax=ax, 
                plt_type='trace', include_burnin=True, scatter=True, n_burnin=n_burnin, 
                alpha=0.90, title='Mean In-Degree', rasterized=True, ylabel='Degree (%)')
        # ax.fill_between(xs, y1=mean_low, y2=mean_high, alpha=0.15, color='blue')
        plt.savefig(inpath + 'mean.pdf')
        plt.close()

        # Plot out degree
        outpath = interactionspath + 'out_degree/'
        os.makedirs(outpath, exist_ok=True)
        out_arr = inout_dict['out']
        for cidx in range(len(CLUSTERING)):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            if SECTION == 'burnin':
                n_burnin = len(out_arr)
            else:
                n_burnin = chain.burnin
            ax = pl.visualization.render_trace(var=out_arr, idx=cidx, ax=ax, 
                plt_type='trace', include_burnin=True, scatter=True, n_burnin=n_burnin, 
                alpha=0.5, title='Out-degree\nCluster {}'.format(cidx), rasterized=True, 
                ylabel='Degree (%)')
            plt.savefig(outpath + 'cluster{}.pdf'.format(cidx))
            plt.close()

        mean_out = np.mean(a=out_arr, axis=1)
        # mean_low = np.quantile(a=out_arr, axis=1, q=.25)
        # mean_high = np.quantile(a=out_arr, axis=1, q=.75)
        # xs = np.arange(len(mean_high)) - chain.burnin

        fig = plt.figure()
        ax = fig.add_subplot(111)
        if SECTION == 'burnin':
            n_burnin = len(mean_out)
        else:
            n_burnin = chain.burnin
        ax = pl.visualization.render_trace(var=mean_out, ax=ax, 
                plt_type='trace', include_burnin=True, scatter=True, n_burnin=n_burnin, 
                alpha=0.90, title='Mean Out-Degree', rasterized=True, ylabel='Degree (%)')
        # ax.fill_between(xs, y1=mean_low, y2=mean_high, alpha=0.15, color='blue')
        plt.savefig(outpath + 'mean.pdf')
        plt.close()

    if chain.is_in_inference_order(STRNAMES.PRIOR_VAR_INTERACTIONS):
        f = open(finteractionspath, 'a')
        pv = chain.graph[STRNAMES.PRIOR_VAR_INTERACTIONS]
        summary = pl.variables.summary(pv, section=SECTION)
        f.write('\n\n###################################\n')
        f.write('Prior Variance Interactions\n')
        f.write('###################################\n')
        for key,ele in summary.items():
            f.write('{}: {}\n'.format(key,ele))
        ax1, _ = pl.visualization.render_trace(var=pv, plt_type='both', log_scale=True, 
            section=SECTION, include_burnin=True, rasterized=True)

        l,h = ax1.get_xlim()
        xs = np.arange(l,h,step=(h-l)/100) 
        ys = []
        for x in xs:
            ys.append(pl.random.sics.pdf(value=x, 
                dof=pv.prior.dof.value,
                scale=pv.prior.scale.value))
        ax1.plot(xs, ys, label='prior', alpha=0.5, color='red')
        ax1.legend()

        fig = plt.gcf()
        fig.suptitle('Prior Variance Interactions, {}'.format(
            LATEXNAMES.PRIOR_VAR_INTERACTIONS))
        plt.savefig(interactionspath + 'var.pdf')
        plt.close()
        f.close()

    if chain.is_in_inference_order(STRNAMES.PRIOR_MEAN_INTERACTIONS):
        f = open(finteractionspath, 'a')
        mean = chain.graph[STRNAMES.PRIOR_MEAN_INTERACTIONS]
        summary = pl.variables.summary(mean, section=SECTION)
        f.write('\n\n###################################\n')
        f.write('Prior Mean Interactions\n')
        f.write('###################################\n')
        for key,ele in summary.items():
            f.write('{}: {}\n'.format(key,ele))
        ax1, _ = pl.visualization.render_trace(var=mean, plt_type='both', 
            log_scale=True, section=SECTION, include_burnin=True, rasterized=True)

        l,h = ax1.get_xlim()
        xs = np.arange(l,h,step=(h-l)/100) 
        ys = []
        for x in xs:
            ys.append(pl.random.normal.pdf(value=x, 
                mean=mean.prior.mean.value,
                std=np.sqrt(mean.prior.var.value)))
        ax1.plot(xs, ys, label='prior', alpha=0.5, color='red')
        ax1.legend()

        fig = plt.gcf()
        fig.suptitle('Prior Mean Interactions, {}'.format(
            LATEXNAMES.PRIOR_MEAN_INTERACTIONS))
        plt.savefig(interactionspath + 'mean.pdf')
        plt.close()
        f.close()

    if chain.is_in_inference_order(STRNAMES.INDICATOR_PROB):
        pi_z = chain.graph[STRNAMES.INDICATOR_PROB]
        f = open(finteractionspath, 'a')

        f.write('\n\n###################################\n')
        f.write('Indicator Probability\n')
        f.write('###################################\n')
        summary = pl.variables.summary(pi_z, section=SECTION)
        for key,val in summary.items():
            f.write('{}: {}\n'.format(key,val))
        pl.visualization.render_trace(var=pi_z, plt_type='both',
            section=SECTION, include_burnin=True, rasterized=True)
        fig = plt.gcf()
        fig.suptitle('Probability of an Interaction, {}'.format(
            LATEXNAMES.INDICATOR_PROB))
        plt.savefig(interactionspath + 'indicator_prob_trace.pdf')
        plt.close()
        f.close()

    # Plot perturbation parameters
    # ----------------------------
    total_perts = None
    total_perts_bf = None
    if STRNAMES.PERT_VALUE in chain.graph:

        total_perts = np.zeros(shape=( len(CLUSTERING),len(subjset.perturbations)))
        total_perts_bf = np.zeros(shape=( len(CLUSTERING),len(subjset.perturbations)))

        for pidx in range(len(subjset.perturbations)):
            pert = subjset.perturbations[pidx]
            if pert.name is None:
                pname = 'pert{}'.format(pidx)
            else:
                pname = pert.name

            perturbation_path = basepath + '{} perturbations/'.format(pname)
            os.makedirs(perturbation_path, exist_ok=True)
            f = open(perturbation_path + 'output.txt', 'w')

            perturbation = chain.graph.perturbations[pidx]

            f.write('################################\n')
            f.write('Perturbation {}\n'.format(pidx))
            f.write('\tStart: {}\n'.format(perturbation.start))
            f.write('\tEnd: {}\n'.format(perturbation.end))
            f.write('\tLearn magnitude? {}\n'.format(
                chain.is_in_inference_order(STRNAMES.PERT_VALUE)))
            f.write('\tLearn indicator? {}\n'.format(
                chain.is_in_inference_order(STRNAMES.PERT_INDICATOR)))
            f.write('\tLearn probability? {}\n'.format(
                chain.is_in_inference_order(STRNAMES.PERT_INDICATOR_PROB)))
            f.write('\tLearn magnitude prior variance? {}\n'.format( 
                chain.is_in_inference_order(STRNAMES.PRIOR_VAR_PERT)))
            f.write('\tLearn magnitude prior mean? {}\n'.format( 
                chain.is_in_inference_order(STRNAMES.PRIOR_MEAN_PERT)))
            f.write('################################\n')

            if chain.is_in_inference_order(STRNAMES.PERT_INDICATOR_PROB):
                
                f.write('\n\nProbability\n')
                prob_sum = pl.variables.summary(
                    perturbation.probability.get_trace_from_disk())
                for key,val in prob_sum.items():
                    f.write('\t{}: {}\n'.format(key,val))

                ax1, _ = pl.visualization.render_trace(
                    var=perturbation.probability, plt_type='both', rasterized=True)
                fig = plt.gcf()
                fig.suptitle('{} perturbation probability'.format(pname))
                plt.savefig(perturbation_path+'probability.pdf')
                plt.close()

            if chain.is_in_inference_order(STRNAMES.PRIOR_VAR_PERT):
                f.write('\n\nMagnitude Prior Variance\n')
                var_sum = pl.variables.summary( 
                    perturbation.magnitude.prior.var, section=SECTION)
                for key,val in var_sum.items():
                    f.write('\t{}: {}\n'.format(key,val))
                
                ax1, _ = pl.visualization.render_trace( 
                    var=perturbation.magnitude.prior.var, plt_type='both',
                        section=SECTION, include_burnin=True, log_scale=True, rasterized=True)
                fig = plt.gcf()
                fig.suptitle('{} perturbation magnitude prior variance'.format(pname))
                plt.savefig(perturbation_path+'prior_var.pdf')
                plt.close()

                prior_std_trace = np.sqrt(perturbation.magnitude.prior.var.get_trace_from_disk( 
                    section=SECTION))
            else:
                prior_std_trace = None

            if chain.is_in_inference_order(STRNAMES.PRIOR_MEAN_PERT):
                f.write('\n\nMagnitude Prior Mean\n')
                var_sum = pl.variables.summary( 
                    perturbation.magnitude.prior.mean, section=SECTION)
                for key,val in var_sum.items():
                    f.write('\t{}: {}\n'.format(key,val))
                
                ax1, _ = pl.visualization.render_trace( 
                    var=perturbation.magnitude.prior.mean, plt_type='both',
                    section=SECTION, include_burnin=True, rasterized=True)
                fig = plt.gcf()
                fig.suptitle('{} perturbation magnitude prior mean'.format(pname))
                plt.savefig(perturbation_path+'prior_mean.pdf')
                plt.close()

                prior_mean_trace = np.sqrt(perturbation.magnitude.prior.mean.get_trace_from_disk( 
                    section=SECTION))
            else:
                prior_mean_trace = None

            # Create the histogram for the prior if any of the priors were learned
            if prior_std_trace is not None or prior_mean_trace is not None:
                if prior_std_trace is None:
                    prior_std_trace = np.sqrt(perturbation.magnitude.prior.var.value) * \
                        np.ones(LEN_POSTERIOR, dtype=float)
                if prior_mean_trace is None:
                    prior_mean_trace = perturbation.magnitude.prior.mean.value * \
                        np.ones(LEN_POSTERIOR, dtype=float)

                prior_hist = np.zeros(len(prior_std_trace), dtype=float)
                for i in range(len(prior_hist)):
                    prior_hist[i] = pl.random.normal.sample(mean=prior_mean_trace[i], std=prior_std_trace[i])
            else:
                prior_hist = None
            

            perturbation_trace = perturbation.get_trace_from_disk(section=SECTION)
            perturbation_trace = _condense_perturbations(perturbation_trace, CLUSTERING)
            for cidx in range(len(CLUSTERING)):
                f.write('\n\nCluster - {}:\n'.format(cidx))
                f.write('---------------\n')

                try:
                    # This will fail if it was never turned on (always np.nan)
                    ax_posterior, ax_trace = pl.visualization.render_trace(
                        var=perturbation_trace, idx=cidx, plt_type='both', section=SECTION,
                        include_burnin=True, rasterized=True)
                    left,right = ax_posterior.get_xlim()

                    if ax_posterior is not None:
                        # Plot the prior
                        if prior_hist is not None:
                            pl.visualization.render_trace(var=prior_hist, plt_type='hist',
                                label='prior', color='red', alpha=0.5, ax=ax_posterior, rasterized=True)
                        else:
                            l,h = ax_posterior.get_xlim()
                            xs = np.arange(l,h,step=(h-l)/100)
                            prior = perturbation.magnitude.prior
                            ys = []
                            for x in xs:
                                ys.append(prior.pdf(value=x))
                            ax_posterior.plot(xs, ys, label='prior', alpha=0.5, color='red')
                    
                    ax_posterior.set_xlim(left=left*0.8, right=right*1.2)
                except Exception as e:
                    logging.critical('Perturbation `{}` for cluster {} was empty (all np.nan or 0s). ' \
                        'Skipping: {}'.format(pname, cidx, e))

                fig = plt.gcf()
                fig.suptitle('Cluster {} perturbation magnitude'.format(cidx))
                plt.savefig(perturbation_path+'cluster{}.pdf'.format(cidx))
                plt.close()
                pert_sum = pl.variables.summary(perturbation_trace[:,cidx], set_nan_to_0=True)

                for key,val in pert_sum.items():
                    f.write('\t{}: {}\n'.format(key,val))

                if chain.is_in_inference_order(STRNAMES.PERT_INDICATOR):
                    # Calculate bayes factor
                    try:
                        # Get a random asv from the cluster
                        oidx = list(CLUSTERING[CLUSTERING.order[cidx]].members)[0]
                        ind_sum = perturbation_bayes_factor(perturbation, oidx)
                        f.write('\tbayes factor: {}\n'.format(ind_sum))

                        if ind_sum < minimum_bayes_factor:
                            total_perts[cidx, pidx] = 0
                        else:
                            total_perts[cidx, pidx] = pert_sum['mean']
                        total_perts_bf[cidx, pidx] = ind_sum
                    except:
                        logging.critical('Cannot calculate a bayes factor for perturbation without ' \
                            'a prior on the probability ')
                        f.write('\tbayes factor: NA\n')
                else:
                    total_perts[cidx, pidx] = pert_sum['mean']
                    total_perts_bf[cidx, pidx] = np.nan
            f.close()

        # Plot all of the perturbations together
        maxv = np.max(np.absolute(total_perts))
        ax = sns.heatmap(total_perts, cmap="RdBu", center=0, vmin=-maxv, vmax=maxv, 
            xticklabels=[pert.name for pert in subjset.perturbations],
            yticklabels=['Cluster {}'.format(cidx) for cidx in range(len(CLUSTERING))],
            linewidths=0.5)
        fig = plt.gcf()
        fig.suptitle('Expected Perturbations\nBayes factor >= {}'.format(minimum_bayes_factor))
        fig.subplots_adjust(left=0.19)
        plt.savefig(basepath + 'expected_perturbations.pdf')
        plt.close()

    # Plot the totalness
    # ------------------
    if total_interactions is not None and total_perts is not None:
        fig = plt.figure(figsize=(12,6))
        gs = fig.add_gridspec(15,5)
        axperts = fig.add_subplot(gs[:10, 1:3])
        axperts_cbar = fig.add_subplot(gs[14, 1:3])
        colors = sns.color_palette('muted')
        gramneg_color = colors[0]
        grampos_color = colors[1]

        sns.heatmap(total_perts, cmap="RdBu", center=0, vmin=-maxv, vmax=maxv, 
            xticklabels=[pert.name for pert in subjset.perturbations],
            yticklabels=False,
            linewidths=0.5, ax=axperts, cbar_ax=axperts_cbar, 
            cbar_kws={'orientation': 'horizontal'})
        axperts.tick_params(left=False, bottom=False)
        axperts.set_title('Perturbations', fontsize=20)

        for idx,tick in enumerate(axperts.xaxis.get_major_ticks()):
            if subjset.perturbations[idx].name.lower() == 'vancomycin':
                tick.label.set_color(grampos_color)
            elif subjset.perturbations[idx].name.lower() == 'gentamicin':
                tick.label.set_color(gramneg_color)

        axinter = fig.add_subplot(gs[:10,3:])
        axinter_cbar = fig.add_subplot(gs[14, 3:])

        annot = []
        for i in range(total_interactions.shape[0]):
            annot.append([])
            for j in range(total_interactions.shape[1]):
                b = '+'
                if total_interactions[i,j] < 0:
                    b = '\u2212' # uicode minus sign
                annot[-1].append(b)
        annot = np.asarray(annot)
        M = np.log10(np.absolute(total_interactions))
        M[~np.isfinite(M)] = np.nan
        M[M==0] = np.nan
        for i in range(M.shape[0]):
            M[i,i] = np.nan

        cmap = sns.cubehelix_palette(1000, as_cmap=True)
        cmap.set_bad('whitesmoke')
        sns.heatmap(M, annot=annot, fmt='', square=False, 
            robust=True, linewidths=0.5, cmap=cmap, #linecolor='white', 
            cbar_ax=axinter_cbar, cbar_kws={'orientation': 'horizontal'},
            yticklabels=False, ax=axinter)
        axinter.tick_params(bottom=False)
        axinter_cbar.set_title('$\\log_{10}$')
        axinter.set_title('Interactions', fontsize=20)

        fig.subplots_adjust(left=0.10, top=0.79)
        if healthy:
            prefix = 'Healthy Consortium, '  
        else:
            prefix = 'Ulcerative Colitis Consortium, '
        fig.suptitle(prefix + 'Bayes factor >= {}'.format(minimum_bayes_factor),
            fontsize=25)

        df = analyze_clusters.analyze_clusters_df(chain, taxlevel=heatmaptaxlevel)
        # Sort the columns in alphabetical order
        cols = np.asarray(list(df.columns))
        cols = np.sort(cols)
        df = df[cols]
        
        axtaxa = fig.add_subplot(gs[:10,0])
        axtaxa_cbar = fig.add_subplot(gs[14, 0])

        cmap = sns.cubehelix_palette(1000, as_cmap=True)
        cmap.set_bad('whitesmoke')
        df[df==0] = np.nan

        sns.heatmap(df, linewidths=0.5, cmap=cmap, cbar_ax=axtaxa_cbar, 
            cbar_kws={'orientation': 'horizontal'}, 
            yticklabels=['Cluster {}'.format(cidx) for cidx in range(len(CLUSTERING))],
            ax=axtaxa)
        axtaxa.tick_params(bottom=False, left=False)
        axtaxa.set_xticks(np.arange(len(df.columns)))
        axtaxa.set_xticklabels(df.columns, rotation=-45, ha='left')
        axtaxa.set_title(heatmaptaxlevel.capitalize(), fontsize=20)

        for idx, tick in enumerate(axtaxa.xaxis.get_major_ticks()):
            taxa = df.columns[idx]
            if analyze_clusters.is_gram_negative_taxa(taxa, heatmaptaxlevel, chain.graph.data.asvs):
                color = gramneg_color
            else:
                color = grampos_color
            tick.label.set_color(color)

        plt.savefig(basepath + 'interactions_and_perturbations.pdf')
        # plt.show()
        plt.close()

        total_perts_bf[np.isinf(total_perts_bf)] = 150000
        annot = []
        for i in range(total_perts_bf.shape[0]):
            annot.append([])
            for j in range(total_perts_bf.shape[1]):
                if total_perts_bf[i,j] >= minimum_bayes_factor:
                    annot[-1].append('>= {}'.format(minimum_bayes_factor))
                else:
                    annot[-1].append('{:.4f}'.format(total_perts_bf[i,j]))
        annot = np.asarray(annot)


        ax = sns.heatmap(data=total_perts_bf, vmax=10, cmap='Blues', linewidths=0.5, 
            annot=annot, fmt='',
            yticklabels=['Cluster {}'.format(cidx) for cidx in range(len(CLUSTERING))],
            xticklabels=[pert.name for pert in subjset.perturbations])
        ax.set_title('Perturbation Bayes Factors')
        plt.savefig(basepath + 'perturbation_bf.pdf')
        plt.close()

    # # Plot the data summary piecharts
    # # -------------------------------
    # fig = plt.figure(figsize=(14,9))
    # colors__ = sns.color_palette(n_colors=10)
    # if piechart_axis_layout[0]* piechart_axis_layout[1] < len(CLUSTERING):
    #     raise ValueError('`piechart_axis_layout` ({}) must multiple to equal ' \
    #         'the number of clusters ({})'.format(piechart_axis_layout, len(CLUSTERING)))

    # # Get the matrix at the times
    # all_times = []
    # subjects = chain.graph.data.subjects
    # for subj in subjects:
    #     all_times = np.append(all_times, subj.times)
    # all_times = np.sort(np.unique(all_times))
    # start_idx = np.searchsorted(all_times, abund_times_start)
    # end_idx = np.searchsorted(all_times, abund_times_end)
    # times = all_times[start_idx:end_idx]
    # M = subjects.matrix(agg='mean', times=times, dtype='abs')


    # df = analyze_clusters.analyze_clusters_df(chain, taxlevel=datataxlevel, 
    #     prop_total=None, include_nan=True)
    # print(df)
    # M = df.to_numpy()
    # labels = np.asarray(list(df.columns))
    # idxs = np.argsort(labels)
    # labels = np.asarray(list(df.columns))
    # colors = {}
    # for i,colidx in enumerate(idxs):
    #     label = labels[colidx]
    #     colors[label] = colors__[i]

    # maxclussize = np.log(np.max([len(cluster) for cluster in CLUSTERING])+1)
    # for cidx, cluster in enumerate(CLUSTERING):
    #     ax = fig.add_subplot(*(piechart_axis_layout + (cidx+1,)))
    #     ax.set_title('Cluster {}\n{} ASVs'.format(cidx, len(cluster)),
    #         fontsize=16)
    #     sizes = M[cidx, :]
    #     mask = sizes > 0

    #     sizes_ = sizes[mask]
    #     labels_ = labels[mask]
    #     colors_ = []

    #     for label in labels_:
    #         colors_.append(colors[label])

    #     ax.pie(sizes_, labels=None, colors=colors_, autopct=None, #'%1.1f%%', 
    #         startangle=90, radius=np.log(len(cluster)+1))
    #     ax.set_xlim(left=-maxclussize, right=maxclussize)
    #     ax.set_ylim(bottom=-maxclussize, top=maxclussize)
        
    # # Make the legend
    # ax = fig.add_subplot(111, facecolor='none')

    # handles = []
    # for label, color in colors.items():
    #     l = ax.scatter([], [], color=color, label=label, linestyle='-')
    #     handles.append(l)
    # legend = plt.legend(handles=handles, 
    #     title='$\\bf{'+datataxlevel.capitalize() +'}$',
    #     bbox_to_anchor=(1.05, 1), loc='upper left',
    #     fontsize=16, title_fontsize=20)
    # ax.add_artist(legend)

    # ax.spines['top'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.xaxis.set_major_locator(plt.NullLocator())
    # ax.xaxis.set_minor_locator(plt.NullLocator())
    # ax.yaxis.set_major_locator(plt.NullLocator())
    # ax.yaxis.set_minor_locator(plt.NullLocator())

    # fig.subplots_adjust(left=0.02, right=0.72, bottom=0.04, hspace=0.37)
    # if healthy:
    #     prefix = 'Healthy Consortium '
    # else:
    #     prefix = 'Ulcerative Colitis Consortium '
    # fig.suptitle(prefix + 'Cluster Composition', fontsize=25, fontweight='bold')
    # plt.savefig(basepath + 'data_summ.pdf')
    # plt.show()

    # Make the two networkx objects to send to cytoscape
    # --------------------------------------------------
    # Make the asv graph (uncondensed)

    # np.save(basepath + 'total_interactions_asvs.py', total_interactions_asvs)
    # np.save(basepath + 'bayes_factors_asvs.py', bayes_factors_asvs)
    # np.save(basepath + 'total_interactions.npy', total_interactions)
    # np.save(basepath + 'bayes_factors.npy', bayes_factors_clusters)

    # total_interactions_asvs = np.load(basepath + 'total_interactions_asvs.py.npy')
    # bayes_factors_asvs = np.load(basepath + 'bayes_factors_asvs.py.npy')
    # total_interactions = np.load(basepath + 'total_interactions.npy')
    # bayes_factors_clusters = np.load(basepath + 'bayes_factors.npy')

    # print('here mf')
    G = nx.MultiDiGraph()
    for asvname in ASVS.names.order:
        G.add_node(asvname)

    # Add edges
    for iidx in range(len(ASVS)):
        target_asv = ASVS.names.order[iidx]
        for jidx in range(len(ASVS)):
            if iidx == jidx:
                continue
            magnitude = total_interactions_asvs[iidx, jidx]
            if magnitude == 0:
                continue
            G.add_edge(ASVS.names.order[jidx], ASVS.names.order[iidx], 
                weight=total_interactions_asvs[iidx, jidx],
                bayes_factor=bayes_factors_asvs[iidx, jidx])
    nx.write_graphml(G, basepath + 'asv_network.xml')

    # Plot the clusters
    G = nx.MultiDiGraph()
    for cidx, cluster in enumerate(CLUSTERING):
        G.add_node('Cluster {}'.format(cidx), size=len(cluster))

    # Add edges
    for icidx in range(len(CLUSTERING)):
        iname = 'Cluster {}'.format(icidx)
        for jcidx in range(len(CLUSTERING)):
            if icidx == jcidx:
                continue
            if bayes_factors_clusters[icidx, jcidx] < minimum_bayes_factor:
                continue
            jname = 'Cluster {}'.format(jcidx)
            magnitude = total_interactions[icidx, jcidx]
            if magnitude == 0:
                continue
            G.add_edge(jname, iname, 
                weight=total_interactions[icidx, jcidx],
                bayes_factor=bayes_factors_clusters[icidx, jcidx])
    nx.write_graphml(G, basepath + 'cluster_network.xml')

def _condense_perturbations(arr, clustering):
    '''Condense the perturbations from asv-asv to cluster-cluster.
    
    Parameters
    ----------
    matrix : np.ndarray(..., n)
        n are the number of ASVs
    clustering : pylab.cluster.Clustering
        This is the clustering object

    Returns
    -------
    np.ndarray(...,m,m)
        m is the number of clusters
    '''
    m = len(clustering)
    ret = np.zeros(shape=arr.shape[:-1] + (m,))

    for icidx, icluster in enumerate(clustering):
        # Pick a single asv from the cluster (it should not matter which)
        iaidx = list(icluster.members)[0]
        ret[..., icidx] = arr[..., iaidx]

    return ret

def _condense_interactions(matrix, clustering):
    '''Condense the interactions from asv=asv to cluster-cluster.
    
    Parameters
    ----------
    matrix : np.ndarray(..., n, n)
        n are the number of ASVs
    clustering : pylab.cluster.Clustering
        This is the clustering object

    Returns
    -------
    np.ndarray(...,m,m)
        m is the number of clusters
    '''
    m = len(clustering)
    ret = np.zeros(shape=matrix.shape[:-2] + (m,m))

    for icidx, icluster in enumerate(clustering):
        # Pick a single asv from the cluster (it should not matter which)
        iaidx = list(icluster.members)[0]

        for jcidx, jcluster in enumerate(clustering):
            if jcidx == icidx:
                continue
            jaidx = list(jcluster.members)[0]
            ret[..., icidx, jcidx] = matrix[..., iaidx, jaidx]

    return ret

# @numba.jit(nopython=True)
def calc_eigan_over_gibbs(ret, growth, si, interactions):
    '''Calculate the stability of dynamical system with the growth rates 
    `growth`, self-limiting terms `si`, and interaction matrix `interactions`

    Parameters
    ----------
    ret : np.ndarray(n_gibbs, n_asvs)
        Return array
    growth : np.ndarray(n_gibbs, n_asvs)
        Growth rates array
    si : np.ndarray(n_gibbs, n_asvs)
        Self-limiting terms array
    interactions : np.ndarray(n_gibbs, n_asvs, n_asvs)
        Interaction matrix for each Gibbs step
    '''
    n_gibbs = ret.shape[0]

    for i in range(n_gibbs):
        # print('\ni', i)
        # if i % 1000 == 0:
        #     print('{}/{}'.format(i, n_gibbs))
        r = growth[i]
        s = -si[i,:]
        A = interactions[i,:,:]

        for jj in range(A.shape[0]):
            # print('\njj', jj)
            # print(A[jj,:])
            # print('\t{:.5E}, {:.5E}'.format(s[jj], np.sum(A[jj,:])))
            A[jj,jj] = s[jj]

        # plt.imshow(np.log(np.absolute(A)))
        # plt.show()

        # sys.exit()

        # print(s.shape)
        # print(np.diag(s).shape)
        # print()
        # print(s)
        # print()
        # print(np.diag(A))

        ret[i,:,:] = np.diag(r) @ A
    
    return ret


    