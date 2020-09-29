'''Implement the Maximum Likelihood Constrained Ridge Regression specified in
the original MDSINE [1] paper.

[1] B. Vanni, et al., "Mdsine: Microbial dynamical systems inference engine for 
    microbiome time-series analysis," Genome Biology, 17(1):121, 2016.
'''

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

import config
import main_ICML

sys.path.append('..')
import model
import synthetic
import names
import preprocess_filtering as filtering
import data
import main_base
import mlcrr

STRNAMES = names.STRNAMES
REPRNAMES = names.REPRNAMES

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-seed', '-d', type=int,
        help='Seed to initialize the data',
        dest='data_seed')
    parser.add_argument('--init_seed', '-i', type=int,
        help='Seed to initialize the inference',
        dest='init_seed')
    parser.add_argument('--basepath', '-b', type=str,
        help='Folder to save the output', default='output_mlcrr/',
        dest='basepath')
    parser.add_argument('--data-path', '-db', type=str,
        help='Folder to lead the data from', dest='data_path')
    parser.add_argument('--healthy', '-hy', type=bool,
        help='Whether or not to use the healthy patients or not',
        default=False, dest='healthy')
    parser.add_argument('--measurement-noise-level', '-m', type=float,
        help='Amount of measurement noise for the counts',
        default=0.1, dest='measurement_noise_level')
    parser.add_argument('--process-variance-level', '-p', type=float,
        help='Amount of process variance',
        default=0.05, dest='process_variance_level')
    parser.add_argument('--n-asvs', '-n', type=int,
        help='Number of ASVs', default=20,
        dest='n_asvs')
    parser.add_argument('--n-times', '-nt', type=int,
        help='Number of time points', 
        dest='n_times')
    parser.add_argument('--n-replicates', '-nr', type=int,
        help='Number of replicates to use for the data of inference',
        dest='n_replicates', default=5)
    parser.add_argument('--n-cpus', '-c', type=int,
        help='Number of cpus to use for the data of inference',
        dest='n_cpus', default=1)
    parser.add_argument('--uniform-sampling', '-us', type=int,
        help='Whether or not to use uniform sampling or not',
        default=0, dest='uniform_sampling')

    return parser.parse_args()

def make_comparison(syndata_filename, exact_subjset):
    '''Create a graph and data that has all the true values in it.

    Needs to all be in the same graph and needs to have the same variable names
    as in the inference graph.
    '''
    synth = synthetic.SyntheticData.load(syndata_filename)

    # Set the variables in the graph
    GRAPH = pl.graph.Graph(name='synthetic')
    d = data.Data(asvs=exact_subjset.asvs, subjects=exact_subjset, G=GRAPH, 
        data_logscale=True)
    n_asvs = d.n_asvs

    growth = pl.variables.Variable(G=GRAPH, name=STRNAMES.GROWTH_VALUE, 
        shape=(n_asvs,))
    growth.value = synth.dynamics.growth

    self_interactions = pl.variables.Variable(G=GRAPH, name=STRNAMES.SELF_INTERACTION_VALUE,
        shape=(n_asvs,))
    self_interactions.value = synth.dynamics.self_interactions

    interactions = pl.variables.Variable(G=GRAPH, name=STRNAMES.CLUSTER_INTERACTION_VALUE, 
        shape=(n_asvs, n_asvs))
    interactions.value = synth.dynamics.interactions.get_datalevel_value_matrix(
        set_neg_indicators_to_nan=False)

    if synth.dynamics.perturbations is not None:
        for synth_perturbation in synth.dynamics.perturbations:
            perturbation = pl.contrib.Perturbation( 
                start=synth_perturbation.start, 
                end=synth_perturbation.end,
                name=synth_perturbation.name,
                G=GRAPH, asvs=exact_subjset.asvs)
            indicators = synth_perturbation.indicator.item_bool_array()
            values = synth_perturbation.item_array(only_pos_ind=True)
            iii = 0
            for oidx in range(n_asvs):
                if indicators[oidx]:
                    perturbation.indicator.value[oidx] = True
                    perturbation.magnitude.value[oidx] = values[iii]
                    iii += 1
                else:
                    perturbation.indicator.value[oidx] = False
                    perturbation.magnitude.value[oidx] = 0
    
    # set the inference
    mlcrr = pl.inference.MLRR(constrain=True, graph=GRAPH)
    return mlcrr

if __name__ == '__main__':
    '''Build the parameters, model, and run the cross validation
    '''
    args = parse_args()
    ONLY_PLOT = True

    # Generate parameters
    synparams = config.SimulationConfig(times=args.n_times, n_replicates=args.n_replicates,
        n_asvs=args.n_asvs, healthy=args.healthy, 
        process_variance_level=args.process_variance_level,
        measurement_noise_level=args.measurement_noise_level, 
        uniform_sampling_timepoints=args.uniform_sampling)
    lparams = config.LoggingConfig()
    params = config.MLCRRConfig(output_basepath=args.basepath, data_seed=args.data_seed,
        init_seed=args.init_seed, n_cpus=args.n_cpus, data_path=args.data_path)

    logging.info('Init settings')
    logging.info('Data seed: {}'.format(args.data_seed))
    logging.info('Init seed: {}'.format(args.init_seed))
    logging.info('measurement noise level: {}'.format(args.measurement_noise_level))
    logging.info('process variance level: {}'.format(args.process_variance_level))
    logging.info('basepath: {}'.format(args.basepath))
    logging.info('n_asvs: {}'.format(args.n_asvs))

    pl.seed(params.DATA_SEED)
    graph_name = 'graph' + params.suffix() + synparams.suffix()
    basepath = params.OUTPUT_BASEPATH + graph_name + '/'
    os.makedirs(basepath, exist_ok=True)

    subjset_filename = basepath + config.SUBJSET_FILENAME
    validate_subjset_filename = basepath + config.VALIDATION_SUBJSET_FILENAME
    mlcrr_filename = basepath + config.MLCRR_RESULTS_FILENAME
    graph_filename = basepath + config.GRAPH_FILENAME
    params_filename = basepath + config.PARAMS_FILENAME
    fparams_filename = basepath + config.FPARAMS_FILENAME
    syndata_filename = basepath + config.SYNDATA_FILENAME
    synparams_filename = basepath + config.SYNPARAMS_FILENAME

    if not ONLY_PLOT:
        params.save(params_filename)
        synparams.save(synparams_filename)

        # Get the steady states of the real data
        syndata = synthetic.SyntheticData.load(config.make_syndata_base_name(
            basepath=params.DATA_PATH, dset='icml', ds=params.DATA_SEED))
        subjset = pl.base.SubjectSet.load(config.make_subj_name(
            basepath=params.DATA_PATH, data_seed=params.DATA_SEED, n_asvs=args.n_asvs, 
            process_variance=args.process_variance_level, measurement_noise=args.measurement_noise_level,
            n_replicates=args.n_replicates, n_times=args.n_times))

        subjset.save(subjset_filename)
        syndata.save(syndata_filename)

        # Plot
        for i, subj in enumerate(subjset):
            ax = pl.visualization.abundance_over_time(subj=subj, dtype='abs', legend=True,
                taxlevel=None, set_0_to_nan=True, yscale_log=True, lca=False,
                color_code_clusters=True, clustering=syndata.dynamics.clustering)
            plt.savefig(basepath + 'data{}.pdf'.format(i))
        pl.visualization.abundance_over_time(subj=subjset, dtype='qpcr', taxlevel=None, 
            set_0_to_nan=True, yscale_log=True, clustering=syndata.dynamics.clustering)
        plt.savefig(basepath + 'qpcr.pdf')
        pl.visualization.abundance_over_time(subj=subjset, dtype='read-depth', taxlevel=None, 
            set_0_to_nan=True, yscale_log=True, clustering=syndata.dynamics.clustering)
        plt.savefig(basepath + 'read_depth.pdf')

        for pert in syndata.dynamics.perturbations:
            print(pert)
        # sys.exit()

        # Rescale if necessary
        if params.QPCR_NORMALIZATION_MAX_VALUE is not None:
            subjset.normalize_qpcr(max_value=params.QPCR_NORMALIZATION_MAX_VALUE)
            logging.info('Normalizing qPCR values. Normalization constant: {:.3E}'.format(
                subjset.qpcr_normalization_factor))

        # Run the model
        results = mlcrr.runCV(params=params, subjset=subjset_filename, graph_name=graph_name)
        results.save(mlcrr_filename)
        params.save(params_filename)

    params = config.MLCRRConfig.load(params_filename)
    mlcrr_result = pl.inference.MLRR.load(mlcrr_filename)
    
    noise_subjset = pl.base.SubjectSet.load(config.make_val_subjset_name(
        basepath=params.DATA_PATH, ds=params.DATA_SEED, 
        pv=args.process_variance_level, mn=args.measurement_noise_level,
        nt=args.n_times, uniform_sampling_timepoints=args.uniform_sampling))
    exact_subjset = pl.base.SubjectSet.load(config.make_val_subjset_name(
        basepath=params.DATA_PATH, ds=params.DATA_SEED, 
        pv=args.process_variance_level))

    noise_subjset.save(basepath + config.VALIDATION_SUBJSET_FILENAME)
    comparison = make_comparison(syndata_filename, exact_subjset=exact_subjset)

    main_base.validate(
        src_basepath=basepath, model=mlcrr_result, 
        forward_sims=['sim-full'],
        yscale_log=True, run_on_copy=True,
        asv_prefix_formatter='%(index)s: (%(name)s)',
        yticklabels='(%(name)s): %(index)s',
        perturbations_additive=True,
        uniform_sample_timepoints=bool(args.uniform_sampling),
        mp=params.N_CPUS, comparison=comparison,
        traj_error_metric=pl.metrics.logPE,
        pert_error_metric=pl.metrics.RMSE,
        interaction_error_metric=pl.metrics.RMSE,
        growth_error_metric=pl.metrics.logPE,
        si_error_metric=pl.metrics.logPE,
        traj_fillvalue=params.C_M/2,
        clus_error_metric=None)

    

