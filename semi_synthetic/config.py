'''This module holds classes for the logging and model configuration
parameters that are set manually in here. There are also the filtering
functions used to preprocess the data

Learned negative binomial dispersion parameters
-----------------------------------------------
a0
	median: 3.021173158076349e-05
	mean: 3.039336514482573e-05
	25th percentile: 2.8907661553542307e-05
	75th percentile: 3.1848563862236224e-05
	acceptance rate: 0.3636
a1
	median: 0.03610445832385458
	mean: 0.036163868481381596
	25th percentile: 0.034369620675005035
	75th percentile: 0.0378392670993046
	acceptance rate: 0.5324

'''
import logging
import numpy as np
import pylab as pl
import pandas as pd
import sys
import math

sys.path.append('..')
from names import STRNAMES

# Parameters for a0 and a1
NEGBIN_A0 = 3.039336514482573e-05
NEGBIN_A1 = 0.036163868481381596
LEARNED_LOGNORMAL_SCALE = 0.3411239239789811

# File locations
GRAPH_NAME = 'graph'
MCMC_FILENAME = 'mcmc.pkl'
SUBJSET_FILENAME = 'subjset.pkl'
VALIDATION_SUBJSET_FILENAME = 'validate_subjset.pkl'
SYNDATA_FILENAME = 'syndata.pkl'
GRAPH_FILENAME = 'graph.pkl'
HDF5_FILENAME = 'traces.hdf5'
TRACER_FILENAME = 'tracer.pkl'
PARAMS_FILENAME = 'params.pkl'
FPARAMS_FILENAME = 'filtering_params.pkl'
SYNPARAMS_FILENAME = 'synthetic_params.pkl'
MLCRR_RESULTS_FILENAME = 'mlcrr_results.pkl'
COMPARISON_FILENAME = 'comparison.pkl'
RESTART_INFERENCE_SEED_RECORD = 'restart_seed_record.txt'
INTERMEDIATE_RESULTS_FILENAME = 'intermediate_results.tsv'

SEMI_SYNTHETIC_MESHES = [
    (
        [5], # Number of replicates
        [55], # Number of timepoints
        2, # Total number of data seeds
        1, # Total number of initialization seeds
        [0.05, 0.2, 0.4], # Measurement Noises
        [0.05], # Process variances
        [1], # Clustering on/off
        0, # Uniform sampling of timepoints
        0 # Boxplot type
    )] #,    
    # (
    #     [2,3,4,5], # Number of replicates
    #     [55], # Number of timepoints
    #     10, # Total number of data seeds
    #     1, # Total number of initialization seeds
    #     [0.3], # Measurement Noises
    #     [0.1], # Process variances
    #     [1], # Clustering on/off
    #     0, # Uniform sampling of timepoints
    #     1 # Boxplot type
    # ),
    # (
    #     [4], # Number of replicates
    #     [35, 45, 50, 55, 65], # Number of timepoints
    #     10, # Total number of data seeds
    #     1, # Total number of initialization seeds
    #     [0.3], # Measurement Noises
    #     [0.1], # Process variances
    #     [1], # Clustering on/off
    #     1, # Uniform sampling of timepoints
    #     2 # Boxplot type
    # )]

def calculate_reads_a0a1(desired_percent_variation):
    '''
    When we scale the a0 and a1 terms, we are assuming that you want to
    scale the high abundance bacteria for that signal and we scale the
    a0 parameter such that they stay relative to each other.

    If == -1 we set to the full noise
    '''
    if desired_percent_variation == -1:
        desired_percent_variation = 0.05
    p = desired_percent_variation / 0.05
    return NEGBIN_A0*p, NEGBIN_A1*p

def calculate_reads_a0a1_new(desired_percent_variation):
    '''
    When we scale the a0 and a1 terms, we are assuming that you want to
    scale the high abundance bacteria for that signal and we scale the
    a0 parameter such that they stay relative to each other.

    If == -1 we set to the full noise
    '''
    # if desired_percent_variation == -1:
    #     desired_percent_variation = 0.05
    # p = desired_percent_variation / 0.05
    # return NEGBIN_A0*p, NEGBIN_A1*p
    d_a1 = {
        0.05: 3e-6,
        0.2: 6e-3,
        0.4: 2e-1}
    d_a0 = {
        0.05: 5e-8,
        0.2: 5e-5,
        0.4: 5e-4}
    return d_a0[desired_percent_variation], d_a1[desired_percent_variation]

def get_qpcr_noise(desired_percent_variation):
    d_qpcr = {
        0.05: 0.01,
        0.2: 0.1,
        0.4: 0.4}
    return d_qpcr[desired_percent_variation]

def isModelConfig(x):
    '''Checks if the input array is a model config object

    Parameters
    ----------
    x : any
        Instance we are checking

    Returns
    -------
    bool
        True if `x` is a a model config object
    '''
    return x is not None and issubclass(x.__class__, _BaseModelConfig)

def make_subj_name(basepath, data_seed, n_asvs, process_variance, measurement_noise,
    n_replicates, n_times, exact=False, uniform_sampling_timepoints=False):
    '''Make the name for the file
    '''
    if n_replicates is None:
        return basepath + 'subjset_base_ds{}_n{}_pv{}_mn{}_nt{}_us{}.pkl'.format( 
            data_seed, n_asvs, process_variance, measurement_noise, n_times,
            bool(uniform_sampling_timepoints))
    else:
        return basepath + 'subjset_ds{}_n{}_pv{}_mn{}_nr{}_nt{}_us{}_exact{}.pkl'.format( 
            data_seed, n_asvs, process_variance, measurement_noise, n_replicates,
            n_times, bool(uniform_sampling_timepoints), exact)

def make_syndata_base_name(basepath, dset, ds):
    '''Make filename for base dataset
    '''
    return basepath + 'syndata_dyn_dset{}_ds{}.pkl'.format(dset, ds)

def make_syndata_data_name(basepath, dset, ds, pv, ntimes, uniform_sampling_timepoints, nrep):
    '''Make filename for base dataset
    '''
    return basepath + 'syndata_dyn_dset{}_ds{}_pv{}_nr{}_nt{}_us{}.pkl'.format(dset, ds, 
        pv, nrep, ntimes, bool(uniform_sampling_timepoints))

def make_val_subjset_name(basepath, ds, pv, mn=None, nt=None, uniform_sampling_timepoints=None):
    '''Make validation set name. If `mn` is None, then it is an exact subjset.
    Else it has noise
    '''
    if mn is None:
        return basepath + 'val_subjset_exact_ds{}_pv{}.pkl'.format(ds, pv)
    else:
        return basepath + 'val_subjset_noise_ds{}_pv{}_ms{}_nt{}_us{}.pkl'.format(
                ds, pv, mn, nt, bool(uniform_sampling_timepoints))


class _BaseModelConfig(pl.base.Saveable):

    def __str__(self):
        s = '{}'.format(self.__class__.__name__)
        for k,v in vars(self).items():
            s += '\n\t{}: {}'.format(k,v)
        return s

    def suffix(self):
        raise NotImplementedError('Need to implement')


class ModelConfigMCMC(_BaseModelConfig):
    '''Configuration parameters for the model


    System initialization
    ---------------------
    SEED : int
        The random seed - set for all different modules (through pylab)
    DATA_FILENAME : str
        Location of the real data
    BURNIN : int
        Number of initial iterations to throw away
    N_SAMPLES : int
        Total number of iterations to perform
    CHECKPOINT : int
        This is the number of iterations that are saved to RAM until it is written
        to disk
    INTERMEDIATE_STEP : float, None
        This is the time step to do the intermediate points
        If there are no intermediate points added then we add no time points
    ADD_MIN_REL_ABUDANCE : bool
        If this is True, it will add the minimum relative abundance to all the data
    PROCESS_VARIANCE_TYPE : str
        Type of process variance to learn
        Options
            'homoscedastic'
                Not Implemented
            'heteroscedastic-global'
                Learn a heteroscadastic process variance (scales with the abundance)
                with global parameters (same v1 and v2 for every ASV)
            'heteroscedastic-per-asv'
                Learn v1 and v2 for each ASV separately
    DATA_DTYPE : str
        Type of data we are going to regress on
        Options:
            'abs': absolute abundance (qPCR*relative_abundance) data
            'rel': relative abundance data
            'raw': raw count data
    DIAGNOSTIC_VARIABLES : list(str)
        These are the names of the variables that you want to trace that are not
        necessarily variables we are learning. These are more for monitoring the
        inference
    QPCR_NORMALIZATION_MAX_VALUE : int, None
        Max value to set the qpcr value to. Rescale everything so that it is proportional
        to each other. If None there are no rescalings
    C_M : numeric
        This is the level of reintroduction of microbes each day

    Which parameters to learn
    -------------------------
    If the following parameters are true, then we add them to the inference order.
    These should all be `bool`s except for `INFERENCE_ORDER` which should be a list
    of `str`s.

    LEARN_BETA : bool
        Growth, self-interactions, interactions
    LEARN_CONCENTRATION : bool
        Concentration parameter of the clustering of the interactions
    LEARN_CLUSTER_ASSIGNMENTS : bool
        Cluster assignments to cluster the interactions
    LEARN_INDICATORS : bool
        Clustered interaction indicators of the interactions
    LEARN_INDICATOR_PROBABILITY : bool
        Probability of a positive interaction indicator
    LEARN_PRIOR_VAR_GROWTH : bool
        Prior variance of the growth
    LEARN_PRIOR_VAR_SELF_INTERACTIONS : bool
        Prior variance of the self-interactions
    LEARN_PRIOR_VAR_INTERACTIONS : bool
        Prior variance of the clustered interactions
    LEARN_PROCESS_VAR : bool
        Process variance parameters
    LEARN_FILTERING : bool
        Learn the auxiliary and the latent trajectory
    LEARN_PERT_VALUE : bool
        Magnitudes of the perturbation effects
    LEARN_PERT_INDICATOR : bool
        Clustered indicators of the perturbation effects
    LEARN_PERT_INDICATOR_PROBABILITY : bool
        Probability of a cluster ebing affected by the perturbation
    INFERENCE_ORDER : list
        This is the global order to learn the paramters. If one of the above parameters are
        False, then their respective name is removed from the inference order

    Initialization parameters
    -------------------------
    These are the arguments to send to the initialization. These should all be dictionaries
    which maps a string (argument) to its value to be passed into the `initialize` function.
    Last parameter is the initialization order, which is the order that we call the
    `initialize` function. These functions are called whether if the variables are being
    learned or not
    '''
    def __init__(self, output_basepath, data_path, data_seed, init_seed, a0, a1,
        n_samples, burnin, checkpoint, pcc, clustering_on):
        '''Initialize
        '''
        self.OUTPUT_BASEPATH = output_basepath
        self.DATA_PATH = data_path
        self.DATA_SEED = data_seed
        self.INIT_SEED = init_seed

        self.DATA_FILENAME = '../pickles/real_subjectset.pkl'
        self.BURNIN = burnin
        self.N_SAMPLES = n_samples
        self.CHECKPOINT = checkpoint
        self.ADD_MIN_REL_ABUNDANCE = False
        self.PROCESS_VARIANCE_TYPE = 'multiplicative-global'
        self.DATA_DTYPE = 'abs'
        self.DIAGNOSTIC_VARIABLES = ['n_clusters']

        self.DATA_LOGSCALE = False
        self.GROWTH_TRUNCATION_SETTINGS = 'positive'
        self.SELF_INTERACTIONS_TRUNCATION_SETTINGS = 'positive'

        self.QPCR_NORMALIZATION_MAX_VALUE = 100
        self.C_M = 1e5

        # This is whether to use the log-scale dynamics or not
        self.DATA_LOGSCALE = True
        self.PERTURBATIONS_ADDITIVE = False
        self.ZERO_INFLATION_TRANSITION_POLICY = None

        self.MP_FILTERING = 'debug'
        self.MP_INDICATORS = None
        self.MP_CLUSTERING = 'debug'
        self.MP_ZERO_INFLATION = None
        self.RELATIVE_LOG_MARGINAL_INDICATORS = True
        self.RELATIVE_LOG_MARGINAL_PERT_INDICATORS = True
        self.RELATIVE_LOG_MARGINAL_CLUSTERING = False
        self.PERCENT_CHANGE_CLUSTERING = pcc

        self.NEGBIN_A0 = a0
        self.NEGBIN_A1 = a1
        self.CLUSTERING_ON = clustering_on

        self.N_QPCR_BUCKETS = 3

        self.INTERMEDIATE_VALIDATION_T = 10 * 60 #8 * 3600 # Every 8 hours
        self.INTERMEDIATE_VALIDATION_KWARGS = None

        self.LEARN = {
            STRNAMES.REGRESSCOEFF: True,
            STRNAMES.PRIOR_VAR_GROWTH: False,
            STRNAMES.PRIOR_VAR_SELF_INTERACTIONS: False,
            STRNAMES.PRIOR_VAR_INTERACTIONS: True,
            STRNAMES.PRIOR_VAR_PERT: True,
            STRNAMES.PRIOR_MEAN_GROWTH: True,
            STRNAMES.PRIOR_MEAN_SELF_INTERACTIONS: True,
            STRNAMES.PRIOR_MEAN_INTERACTIONS: True,
            STRNAMES.PRIOR_MEAN_PERT: True,
            STRNAMES.PROCESSVAR: True,
            STRNAMES.FILTERING: True,
            STRNAMES.ZERO_INFLATION: False,
            STRNAMES.CLUSTERING: True, #clustering_on,
            STRNAMES.CONCENTRATION: True,
            STRNAMES.CLUSTER_INTERACTION_INDICATOR: True,
            STRNAMES.INDICATOR_PROB: True,
            STRNAMES.PERT_INDICATOR: True,
            STRNAMES.QPCR_SCALES: False,
            STRNAMES.QPCR_DOFS: False,
            STRNAMES.QPCR_VARIANCES: False,
            STRNAMES.PERT_INDICATOR_PROB: True}

        self.INFERENCE_ORDER = [
            STRNAMES.CLUSTER_INTERACTION_INDICATOR,
            STRNAMES.INDICATOR_PROB,
            STRNAMES.PERT_INDICATOR,
            STRNAMES.PERT_INDICATOR_PROB,
            STRNAMES.REGRESSCOEFF,
            STRNAMES.PRIOR_MEAN_INTERACTIONS,
            STRNAMES.PRIOR_MEAN_PERT,
            STRNAMES.PRIOR_MEAN_GROWTH,
            STRNAMES.PRIOR_MEAN_SELF_INTERACTIONS,
            STRNAMES.PRIOR_VAR_GROWTH,
            STRNAMES.PRIOR_VAR_SELF_INTERACTIONS,
            STRNAMES.PRIOR_VAR_INTERACTIONS,
            STRNAMES.PRIOR_VAR_PERT,
            STRNAMES.PROCESSVAR,
            STRNAMES.ZERO_INFLATION,
            STRNAMES.QPCR_SCALES,
            STRNAMES.QPCR_DOFS,
            STRNAMES.QPCR_VARIANCES,
            STRNAMES.FILTERING,
            STRNAMES.CLUSTERING,
            STRNAMES.CONCENTRATION]

        self.INITIALIZATION_KWARGS = {
            STRNAMES.QPCR_VARIANCES: {
                'value_option': 'empirical'},
            STRNAMES.QPCR_SCALES: {
                'value_option': 'prior-mean',
                'scale_option': 'empirical',
                'dof_option': 'diffuse',
                'proposal_option': 'auto',
                'target_acceptance_rate': 'optimal',
                'end_tune': 'half-burnin',
                'tune': 50,
                'delay':0},
            STRNAMES.QPCR_DOFS: {
                'value_option': 'diffuse',
                'low_option': 'valid',
                'high_option': 'med',
                'proposal_option': 'auto',
                'target_acceptance_rate': 'optimal',
                'end_tune': 'half-burnin',
                'tune': 50,
                'delay': 0},
            STRNAMES.PERT_VALUE: {
                'value_option': 'prior-mean',
                'delay':0},
            STRNAMES.PERT_INDICATOR_PROB: {
                'value_option': 'prior-mean',
                'hyperparam_option': 'strong-sparse',
                'delay':0},
            STRNAMES.PERT_INDICATOR: {
                'value_option': 'all-off',
                'delay':0},
            STRNAMES.PRIOR_VAR_PERT: {
                'value_option': 'prior-mean',
                'scale_option': 'diffuse',
                'dof_option': 'diffuse',
                'delay': 0},
            STRNAMES.PRIOR_MEAN_PERT: {
                'value_option': 'prior-mean',
                'mean_option': 'zero',
                'var_option': 'diffuse',
                'delay':0},
            STRNAMES.PRIOR_VAR_GROWTH: {
                'value_option': 'prior-mean',
                'scale_option': 'inflated-median',
                'dof_option': 'diffuse',
                'proposal_option': 'tight',
                'target_acceptance_rate': 'optimal',
                'end_tune': 'half-burnin',
                'tune': 50,
                'delay':0},
            STRNAMES.PRIOR_MEAN_GROWTH: {
                'value_option': 'prior-mean',
                'mean_option': 'manual',
                'var_option': 'diffuse-linear-regression',
                'proposal_option': 'auto',
                'target_acceptance_rate': 0.44,
                'tune': 50,
                'end_tune': 'half-burnin',
                'truncation_settings': self.GROWTH_TRUNCATION_SETTINGS,
                'delay':0, 'mean': 1},
            STRNAMES.GROWTH_VALUE: {
                'value_option': 'linear-regression', #'prior-mean',
                'truncation_settings': self.GROWTH_TRUNCATION_SETTINGS,
                'delay': 0},
            STRNAMES.PRIOR_VAR_SELF_INTERACTIONS: {
                'value_option': 'prior-mean',
                'scale_option': 'inflated-median',
                'dof_option': 'diffuse',
                'proposal_option': 'tight',
                'target_acceptance_rate': 'optimal',
                'end_tune': 'half-burnin',
                'tune': 50,
                'delay':0},
            STRNAMES.PRIOR_MEAN_SELF_INTERACTIONS: {
                'value_option': 'prior-mean',
                'mean_option': 'median-linear-regression',
                'var_option': 'diffuse-linear-regression',
                'proposal_option': 'auto',
                'target_acceptance_rate': 0.44,
                'tune': 50,
                'end_tune': 'half-burnin',
                'truncation_settings': self.SELF_INTERACTIONS_TRUNCATION_SETTINGS,
                'delay':0, 'mean': 1},
            STRNAMES.SELF_INTERACTION_VALUE: {
                'value_option': 'linear-regression',
                'truncation_settings': self.SELF_INTERACTIONS_TRUNCATION_SETTINGS,
                'delay': 0},
            STRNAMES.PRIOR_VAR_INTERACTIONS: {
                'value_option': 'auto',
                'dof_option': 'diffuse',
                'scale_option': 'same-as-aii',
                'mean_scaling_factor': 1,
                'delay': 0},
            STRNAMES.PRIOR_MEAN_INTERACTIONS: {
                'value_option': 'prior-mean',
                'mean_option': 'zero',
                'var_option': 'same-as-aii',
                'delay':0},
            STRNAMES.CLUSTER_INTERACTION_VALUE: {
                'value_option': 'all-off',
                'delay': 0},
            STRNAMES.CLUSTER_INTERACTION_INDICATOR: {
                'delay': 0,
                'run_every_n_iterations': 1},
            STRNAMES.INDICATOR_PROB: {
                'value_option': 'auto',
                'hyperparam_option': 'strong-sparse',
                'delay': 0},
            STRNAMES.FILTERING: {
                'x_value_option': 'loess',
                # 'q_value_option': 'coupling', #'loess',
                # 'hyperparam_option': 'manual',
                'tune': (int(self.BURNIN/2), 50),
                'a0': self.NEGBIN_A0,
                'a1': self.NEGBIN_A1,
                'v1': 1e-4,
                'v2': 1e-4,
                'proposal_init_scale':.0001,
                'intermediate_interpolation': 'linear-interpolation',
                'intermediate_step': None, #('step', (1, None)), 
                'essential_timepoints': 'union',
                'delay': 2,
                'window': 6,
                'plot_initial': False,
                'target_acceptance_rate': 0.44},
            STRNAMES.ZERO_INFLATION: {
                'value_option': None,
                'delay': 0},
            STRNAMES.CONCENTRATION: {
                'value_option': 'prior-mean',
                'hyperparam_option': 'diffuse',
                'delay': 0, 'n_iter': 20},
            STRNAMES.CLUSTERING: {
                'value_option': 'spearman',
                'delay': 2,
                'n_clusters': 30,
                'percent_mix': self.PERCENT_CHANGE_CLUSTERING,
                'run_every_n_iterations': 4},
            STRNAMES.REGRESSCOEFF: {
                'update_jointly_pert_inter': True,
                'update_jointly_growth_si': False,
                'tune': 50,
                'end_tune': 'half-burnin'},
            STRNAMES.PROCESSVAR: {
                # 'v1': 0.2**2,
                # 'v2': 1,
                # 'q_option': 'previous-t'}, #'previous-t'},
                'dof_option': 'diffuse', # 'half', 
                'scale_option': 'med',
                'value_option': 'prior-mean',
                'delay': 0}
        }

        self.INITIALIZATION_ORDER = [
            STRNAMES.FILTERING,
            STRNAMES.ZERO_INFLATION,
            STRNAMES.CONCENTRATION,
            STRNAMES.CLUSTERING,
            STRNAMES.PROCESSVAR,
            STRNAMES.PRIOR_MEAN_GROWTH,
            STRNAMES.PRIOR_VAR_GROWTH,
            STRNAMES.GROWTH_VALUE,
            STRNAMES.PRIOR_MEAN_SELF_INTERACTIONS,
            STRNAMES.PRIOR_VAR_SELF_INTERACTIONS,
            STRNAMES.SELF_INTERACTION_VALUE,
            STRNAMES.PRIOR_MEAN_INTERACTIONS,
            STRNAMES.PRIOR_VAR_INTERACTIONS,
            STRNAMES.CLUSTER_INTERACTION_VALUE,
            STRNAMES.CLUSTER_INTERACTION_INDICATOR,
            STRNAMES.INDICATOR_PROB,
            STRNAMES.PRIOR_MEAN_PERT,
            STRNAMES.PRIOR_VAR_PERT,
            STRNAMES.PERT_INDICATOR,
			STRNAMES.PERT_VALUE,
            STRNAMES.PERT_INDICATOR_PROB,
            STRNAMES.REGRESSCOEFF,
            STRNAMES.QPCR_SCALES,
            STRNAMES.QPCR_DOFS,
            STRNAMES.QPCR_VARIANCES]

    def suffix(self):
        '''Create a suffix with the parameters
        '''
        perts = 'addit' if self.PERTURBATIONS_ADDITIVE else 'mult'
        
        s = '_ds{}_is{}_b{}_ns{}_co{}_perts{}'.format(
            self.DATA_SEED, self.INIT_SEED, self.BURNIN, self.N_SAMPLES,
            self.CLUSTERING_ON, perts)
        return s


class MLCRRConfig(_BaseModelConfig):
    '''This is the configuration file for the Maximum Likelihood Constrained Ridge
    Regression model defined in the original MDSINE [1] model. This defines the 
    parameters for the model as well as the cross validation configuration.

    System parameters
    -----------------
    DATE_FILENAME

    DATA_DTYPE

    QPCR_nORMALIZATION_MAX_VALUE

    GROWTH_TRUNCATION_SETTINGS

    SELF_INTERACTION_TRUNCATION_SETTINGS

    Cross validation parameters
    ---------------------------
    CV_MAP_MIN, CV_MAP_MAX, CV_MAP_N : int
        Defines the numerical region over which the algorithm will
        search for regularization parameters settings. The region is
        from 10^CV_MAP_MIN to 10^CV_MAP_MAX with CV_MAP_N number of 
        spaces. 
        Example:
            CV_MAP_MIN = -3
            CV_MAP_MAX = 2
            CV_MAP_N = 5
            [0.0010, 0.0178, 0.3162, 5.6234, 100.0000]
    CV_REPLICATES : int
        This is the number of different "shuffles" or replications for cross-fold validation.
        This can and will be parallelized if N_CPUS > 1.
    CV_LEAVE_K_OUT : int
        Leave K out for cross validation: defaults to 1.

    Parameters
    ----------
    output_basepath : str
        Basepath to save everything to
    data_seed : int
        Seed to initialize the data to
    init_seed : int
        Seed to initialize the model to
    n_cpus : int
        Number of available processors available
    '''
    def __init__(self, output_basepath, data_seed, init_seed, data_path, n_cpus=1):

        self.OUTPUT_BASEPATH = output_basepath
        self.DATA_PATH = data_path
        self.DATA_SEED = data_seed
        self.INIT_SEED = init_seed
        self.N_CPUS = n_cpus

        self.DATA_FILENAME = '../pickles/real_subjectset.pkl'
        self.DATA_DTYPE = 'abs'
        self.QPCR_NORMALIZATION_MAX_VALUE = 100

        self.GROWTH_TRUNCATION_SETTINGS = 'positive'
        self.SELF_INTERACTION_TRUNCATION_SETTINGS = 'positive'

        self.C_M = 1e5

        # Cross validation parameters
        self.CV_MAP_MIN = -3
        self.CV_MAP_MAX = 2
        self.CV_MAP_N = 15
        self.CV_REPLICATES = 15
        self.CV_LEAVE_K_OUT = 1

    def suffix(self):
        '''Create a suffix for saving the runs
        '''
        mapmin = str(self.CV_MAP_MIN).replace('-','neg')
        mapmax = str(self.CV_MAP_MAX).replace('-','neg')
        s = '_ds{}_is{}_mmin{}_mmax{}_mn{}_cvnr{}_cvlko{}'.format(
            self.DATA_SEED, self.INIT_SEED, 
            mapmin, mapmax, self.CV_MAP_N, self.CV_REPLICATES,
            self.CV_LEAVE_K_OUT)
        return s


class SimulationConfigBoxplots(_BaseModelConfig):
    '''These are the parameters used to make the synthetic datasets for the
    the boxplots

    System Parameters
    -----------------
    pv_value : float
        What to set the process variance as
    simulation_dt : dt
        The smaller step size we use for froward integration so the integration
        does not become unstable
    n_days : int
        Total number of days to run the simulation for
    times : str, int, float
        How to generate the times
        if str:
            'darpa-study-sampling'
                Denser in the beginning, and around the ends of perturbations
        if int/float
            This is the density to sample at (0.5 means sample every half of
            a day)
    n_replicates : int
        How many replicates of subjects to run the inference with
    init_low, init_high : float
        The low and high to initialize the data at using a uniform
        distribution
    max_abundance : float
        Max abundance
    n_asvs : int
        How many ASVs to simulate
    healthy_patients : bool
        Which consortia of mice to use as noise approximators
    process_variance_level : float
        What to set the process variance to
    measurement_noise_level : float
        What to set the measurement noise to

    '''
    def __init__(self, times, n_replicates, n_asvs, healthy,
        process_variance_levels, measurement_noise_levels, dataset):
        self.PV_VALUES = [pv**2 for pv in process_variance_levels]
        self.SIMULATION_DT = 0.01
        self.N_DAYS = 65
        self.TIMES = times
        self.N_REPLICATES = n_replicates
        self.INIT_LOW = 1e5
        self.INIT_HIGH = 1e7
        self.MAX_ABUNDANCE = 1e8
        self.N_ASVS = n_asvs
        self.HEALTHY_PATIENTS = healthy
        self.PROCESS_STD_LEVELS = process_variance_levels
        self.MEASUREMENT_NOISE_LEVELS = measurement_noise_levels
        self.DATASET = dataset
        self.LOG_DYNAMICS = True

        self.DSET = 'semi-synthetic'
        self.DATA_FILENAME = '../pickles/real_subjectset.pkl'
        self.SEMI_SYNTH_CHAIN_FILENAME = 'base_data/healthy_mcmc.pkl'
        self.SEMI_SYNTH_HDF5_FILENAME = 'base_data/healthy_traces.hdf5'
        self.PREPROCESSED_SEMI_SYNTH_FILENAME = 'base_data/preprocessed_semisynthetic_healthy.pkl'
        self.SEMI_SYNTH_MIN_BAYES_FACTOR = 10
        self.SEMI_SYNTH_FIRST_TIMEPOINT = 1.5

        # (prob_pos, prob_affect, prob_strength, mean_strength, std_strength)
        self.PERTURBATIONS = True

        self.NEGBIN_A0S = []
        self.NEGBIN_A1S = []
        for mn in measurement_noise_levels:
            a0, a1 = calculate_reads_a0a1(mn)
            self.NEGBIN_A0S.append(a0)
            self.NEGBIN_A1S.append(a1)

        self.QPCR_NOISE_SCALES = measurement_noise_levels

    def suffix(self):
        max_abund = self.MAX_ABUNDANCE
        if max_abund is not None:
            max_abund = '{:.2E}'.format(max_abund)
        if len(self.PERTURBATIONS) == 0:
            perts = None
        else:
            perts = len(self.PERTURBATIONS)
        s = '_dset{}_nr{}_no{}_nd{}_ms{}_pv{}_ma{}_np{}'.format(
            self.DATASET,
            self.N_REPLICATES,
            self.N_ASVS,
            self.N_DAYS,
            self.MEASUREMENT_NOISE_LEVELS,
            self.PROCESS_STD_LEVELS,
            max_abund, perts)
        return s


class SimulationConfig(_BaseModelConfig):
    '''These are the paramters used to make a synthetic dataset

    System Parameters
    -----------------
    pv_value : float
        What to set the process variance as
    simulation_dt : dt
        The smaller step size we use for froward integration so the integration
        does not become unstable
    n_days : int
        Total number of days to run the simulation for
    times : str, int, float
        How to generate the times
        if str:
            'darpa-study-sampling'
                Denser in the beginning, and around the ends of perturbations
        if int/float
            This is the density to sample at (0.5 means sample every half of
            a day)
    n_replicates : int
        How many replicates of subjects to run the inference with
    init_low, init_high : float
        The low and high to initialize the data at using a uniform
        distribution
    max_abundance : float
        Max abundance
    n_asvs : int
        How many ASVs to simulate
    healthy_patients : bool
        Which consortia of mice to use as noise approximators
    process_variance_level : float
        What to set the process variance to
    measurement_noise_level : float
        What to set the measurement noise to

    '''
    def __init__(self, times, n_replicates, n_asvs, healthy, uniform_sampling_timepoints,
        process_variance_level, measurement_noise_level):
        self.PV_VALUE = process_variance_level**2
        self.SIMULATION_DT = 0.01
        self.N_DAYS = 65
        self.TIMES = times
        self.N_REPLICATES = n_replicates
        self.UNIFORM_SAMPLING_TIMEPOINTS = bool(uniform_sampling_timepoints)
        self.INIT_LOW = 1e5
        self.INIT_HIGH = 1e7
        self.MAX_ABUNDANCE = 1e8
        self.N_ASVS = n_asvs
        self.HEALTHY_PATIENTS = healthy
        self.PROCESS_VARIANCE_LEVEL = process_variance_level
        self.MEASUREMENT_NOISE_LEVEL = measurement_noise_level

        self.CHAIN_FILENAME = '../output_real/pylab24/real_runs/perts_mult/' \
            'fixed_top/healthy1_5_0.0001_rel_2_5/ds0_is0_b5000_ns20000_mo-1_'\
            'logTrue_pertsmult/graph_leave_out-1/mcmc.pkl'

        self.PERTURBATIONS = (0.3, '2', [0.1, 0.5, 0.4], [0.5, 1, 2], 0.1)
        self.DSET = 'semi-synthetic'
        self.NEGBIN_A0, self.NEGBIN_A1 = calculate_reads_a0a1(measurement_noise_level)
        self.QPCR_NOISE_SCALE = measurement_noise_level

        self.TOPOLOGY_METRIC_SIGNED = True
        self.TOPOLOGY_METRIC_AVERAGE = 'weighted'

    def suffix(self):
        max_abund = self.MAX_ABUNDANCE
        if max_abund is not None:
            max_abund = '{:.2E}'.format(max_abund)
        if len(self.PERTURBATIONS) == 0:
            perts = None
        else:
            perts = len(self.PERTURBATIONS)
        s = '_nr{}_no{}_nd{}_ms{}_pv{}_ma{}_np{}_nt{}_us{}'.format(
            self.N_REPLICATES,
            self.N_ASVS,
            self.N_DAYS,
            self.MEASUREMENT_NOISE_LEVEL,
            self.PROCESS_VARIANCE_LEVEL,
            max_abund, perts,
            self.TIMES,
            self.UNIFORM_SAMPLING_TIMEPOINTS)
        return s
        

class FilteringConfig(pl.Saveable):
    '''These are the parameters for Filtering

    Different types of filtering
    ----------------------------
    `at_least_counts`
        For each ASV in the subjectset `subjset`, delete all ASVs that
        do not have at least a minimum number of counts `min_counts`
        for less than `min_num_subjects` subjects.

        Parameters
        ----------
        colonization_time : numeric, None
            This is the day that you want to start taking the relative abundance.
            We only lok at the relative abundance after the colonization period.
            If this is `None` then it is set to 0.
        min_counts : numeric
            This is the minimum number of counts it needs to have
        min_num_subjects : int
            This is the minimum number of subjects that there must be a relative
            abundance

    `consistency`
        Filters the subjects by looking at the consistency of the counts.
        There must be at least `min_num_counts` for at least
        `min_num_consecutive` consecutive timepoints for at least
        `min_num_subjects` subjects for the ASV to be classified as valid.

        Parameters
        ----------
        min_num_consecutive: int
            This is the minimum number of consecutive timepoints that there
            must be at least `min_num_counts`
        min_num_counts : int
            This is the minimum number of counts that there must be at each
            consecutive timepoint
        min_num_subjects : int, None
            This is how many subjects this must be true for for the ASV to be
            valid. If it is None then it only requires one subject.

    Additional Parameters
    ---------------------
    healthy : bool
        If True, do regression on the healthy patients
    '''
    def __init__(self, healthy):
        self.COLONIZATION_TIME = 5
        self.THRESHOLD = 0.00025
        self.DTYPE = 'rel'
        self.MIN_NUM_SUBJECTS = 'all' #2
        self.MIN_NUM_CONSECUTIVE = 7
        self.HEALTHY = healthy

    def __str__(self):
        return 'healthy{}_{}_{}_{}_{}_{}'.format(
            self.HEALTHY,
            self.COLONIZATION_TIME,
            self.THRESHOLD,
            self.DTYPE,
            self.MIN_NUM_SUBJECTS,
            self.MIN_NUM_CONSECUTIVE)

    def suffix(self):
        return str


class LoggingConfig(pl.Saveable):
    '''These are the parameters for logging

    FORMAT : str
        This is the logging format for stdout
    LEVEL : logging constant, int
        This is the level to log at for stdout
    NUMPY_PRINTOPTIONS : dict
        These are the printing options for numpy.

    Parameters
    ----------
    basepath : str
        If this is specified, then we also want to log to a file. Set up a
        steam and a file
    '''
    def __init__(self, basepath=None):
        self.FORMAT = '%(levelname)s:%(module)s.%(lineno)s: %(message)s'
        self.LEVEL = logging.INFO
        self.NUMPY_PRINTOPTIONS = {
            'threshold': sys.maxsize, 'linewidth': sys.maxsize}

        if basepath is not None:
            path = basepath + 'logging.log'
            self.PATH = path
            handlers = [
                logging.FileHandler(self.PATH, mode='w'),
                logging.StreamHandler()]
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
            logging.basicConfig(level=self.LEVEL, format=self.FORMAT, handlers=handlers)
        else:
            self.PATH = None
            logging.basicConfig(format=self.FORMAT, level=self.LEVEL)
        
        np.set_printoptions(**self.NUMPY_PRINTOPTIONS)
        pd.set_option('display.max_columns', None)
