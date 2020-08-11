'''This module runs the model with the ICML data
'''

import logging
import numpy as np
import sys
import os
import pickle
import copy
import random
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats

import pylab as pl
import config
import posterior
import synthetic
import main_base
import diversity.alpha
from names import STRNAMES
import preprocess_filtering as filtering
import model
import data

import argparse
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-seed', '-d', type=int,
        help='Seed to initialize the data',
        dest='data_seed', default=None)
    parser.add_argument('--init_seed', '-i', type=int,
        help='Seed to initialize the inference',
        dest='init_seed', default=None)
    parser.add_argument('--qpcr-noise-level', '-q', type=float,
        help='Amount of measurement noise for the qPCR measurements',
        default=0.1, dest='qpcr_noise_level')
    parser.add_argument('--measurement-noise-level', '-m', type=float,
        help='Amount of measurement noise for the counts',
        default=0.1, dest='measurement_noise_level')
    parser.add_argument('--process-variance-level', '-p', type=float,
        help='Amount of process variance',
        default=0.05, dest='process_variance_level')
    parser.add_argument('--basepath', '-b', type=str,
        help='Folder to save the output', default='output_ICML/',
        dest='basepath')
    parser.add_argument('--n-asvs', '-n', type=int,
        help='Number of ASVs', default=150,
        dest='n_asvs')
    parser.add_argument('--n-samples', '-ns', type=int,
        help='Total number of Gibbs steps to do',
        dest='n_samples', default=4000)
    parser.add_argument('--burnin', '-nb', type=int,
        help='Total number of burnin steps',
        dest='burnin', default=2000)
    parser.add_argument('--times', '-t', type=int,
        help='Number of time poiints',
        dest='n_times')
    parser.add_argument('--n-replicates', '-nr', type=int,
        help='Number of replicates to use for the data of inference',
        dest='n_replicates', default=5)
    parser.add_argument('--percent-change-clustering', '-pcc', type=float,
        help='Percent of ASVs to update during clustering every time it runs',
        default=1.0, dest='percent_change_clustering')
    parser.add_argument('--clustering-on', '-c', type=int,
        help='If True, turns clustering on, Else there is no clustering.',
        default=1, dest='clustering_on')
    parser.add_argument('--healthy', '-hy', type=bool,
        help='Whether or not to use the healthy patients or not',
        default=False, dest='healthy')
    args = parser.parse_args()
    args.clustering_on = bool(args.clustering_on)

    return args

def make_comparison(syndata_filename, exact_subjset):
    '''Create the graph and data that has all of the true values in it

    Parameters
    ----------
    syndata_filename : str
        Filename for synthetic data
    '''
    synth = synthetic.SyntheticData.load(syndata_filename)

    # Set the variables in the graph
    GRAPH = pl.graph.Graph(name='synthetic')
    d = data.Data(asvs=exact_subjset.asvs, subjects=exact_subjset, G=GRAPH, 
        data_logscale=True)
    n_asvs = d.n_asvs
    
    clustering = pl.cluster.Clustering(clusters=synth.dynamics.clustering.toarray(),
        items=d.asvs, G=GRAPH, name=STRNAMES.CLUSTERING_OBJ)
    
    growth = pl.variables.Variable(G=GRAPH, name=STRNAMES.GROWTH_VALUE, 
        shape=(n_asvs,))
    growth.value = synth.dynamics.growth

    self_interactions = pl.variables.Variable(G=GRAPH, name=STRNAMES.SELF_INTERACTION_VALUE,
        shape=(n_asvs,))
    self_interactions.value = synth.dynamics.self_interactions

    interactions = pl.Interactions(clustering=clustering, use_indicators=True, 
        name=STRNAMES.INTERACTIONS_OBJ, G=GRAPH)

    for i,interaction in enumerate(interactions):
        syn_interaction = synth.dynamics.interactions.iloc(i)
        interaction.indicator = syn_interaction.indicator
        interaction.value = syn_interaction.value

    if synth.dynamics.perturbations is not None:
        for synth_perturbation in synth.dynamics.perturbations:
            perturbation = pl.contrib.ClusterPerturbation(
                start=synth_perturbation.start,
                end=synth_perturbation.end,
                name=synth_perturbation.name,
                clustering=clustering,
                G=GRAPH)
            indicators = synth_perturbation.indicator.cluster_bool_array()
            values = synth_perturbation.cluster_array(only_pos_ind=True)
            
            iii = 0
            for cidx, cid in enumerate(clustering.order):
                if indicators[cidx]:
                    perturbation.indicator.value[cid] = True
                    perturbation.magnitude.value[cid] = values[iii]
                    iii += 1
                else:
                    perturbation.indicator.value[cid] = False
                    perturbation.magnitude.value[cid] = 0
    
    # Set the inference
    mcmc = pl.inference.BaseMCMC(burnin=100, n_samples=200, graph=GRAPH)
    return mcmc

if __name__ == '__main__':
    '''Build the parameters of the model and run

    1. Set up logging
    2. Get the ICML synthetic system
    3. qPCR normalization (for numerical stability)
    4. Plot (Optional)
    5. Specify parameters of the model
    6. Run
    '''
    args = parse_args()

    if args.data_seed is None:
        args.data_seed = 9890014
    if args.init_seed is None:
        args.init_seed = 11114703

    if args.n_samples <= args.burnin:
        raise ValueError('`n_samples` ({}) must be larger than burnin ({})'.format(
            args.n_samples, args.burnin))

    # Constants
    ONLY_PLOT = False
    
    # Start
    ##############################
    config.LoggingConfig()
    fparams = config.FilteringConfig(healthy=args.healthy)
    synparams = config.SimulationConfig(times=args.n_times, n_replicates=args.n_replicates,
        n_asvs=args.n_asvs, healthy=args.healthy, 
        process_variance_level=args.process_variance_level,
        measurement_noise_level=args.measurement_noise_level)
    params = config.ModelConfigICML(output_basepath=args.basepath, data_seed=args.data_seed,
        data_path=None,
        init_seed=args.init_seed, a0=synparams.NEGBIN_A0, a1=synparams.NEGBIN_A1,
        n_samples=args.n_samples, burnin=args.burnin, pcc=args.percent_change_clustering,
        clustering_on=args.clustering_on)

    logging.info('Init settings')
    logging.info('Data seed: {}'.format(args.data_seed))
    logging.info('Init seed: {}'.format(args.init_seed))
    logging.info('measurement noise level: {}'.format(args.measurement_noise_level))
    logging.info('qPCR noise level: {}'.format(args.qpcr_noise_level))
    logging.info('process variance level: {}'.format(args.process_variance_level))
    logging.info('basepath: {}'.format(args.basepath))
    logging.info('n_asvs: {}'.format(args.n_asvs))

    pl.seed(params.DATA_SEED)    
    graph_name = 'graph'+ params.suffix() + synparams.suffix()
    basepath = params.OUTPUT_BASEPATH + graph_name + '/'
    
    os.makedirs(basepath, exist_ok=True) # Make the folder
    
    chain_result_filename = basepath + config.MCMC_FILENAME
    subjset_filename = basepath + config.SUBJSET_FILENAME
    validate_subjset_filename = basepath + config.VALIDATION_SUBJSET_FILENAME
    graph_filename = basepath + config.GRAPH_FILENAME
    hdf5_filename = basepath + config.HDF5_FILENAME
    tracer_filename = basepath + config.TRACER_FILENAME
    params_filename = basepath + config.PARAMS_FILENAME
    fparams_filename = basepath + config.FPARAMS_FILENAME
    syndata_filename = basepath + config.SYNDATA_FILENAME
    synparams_filename = basepath + config.SYNPARAMS_FILENAME
    exact_subjset_filename = basepath + 'exact_subjset.pkl'

    subjset_real = pl.SubjectSet.load(params.DATA_FILENAME)
    if not ONLY_PLOT:
        params.save(params_filename)
        fparams.save(fparams_filename)
        synparams.save(synparams_filename)

        # Get the steady states of the real data
        if not synparams.HEALTHY_PATIENTS:
            sidxs = ['2','3','4','5']
        else:
            sidxs = ['6','7','8','9','10']
        for sidx in sidxs:
                subjset_real.pop_subject(sidx)
        try:
            syndata = synthetic.SyntheticData.load(syndata_filename)
            LOAD_SYNDATA_FROM_PKL = True
        except Exception as e:
            LOAD_SYNDATA_FROM_PKL = False
            logging.critical('Cant load:\n{}'.format(e))
        # LOAD_SYNDATA_FROM_PKL = False
        if not LOAD_SYNDATA_FROM_PKL:
            # Generate the synthetic data
            
            if synparams.N_DAYS == 'from-data':
                n_days = np.max([np.max(subj.times) for subj in subjset_real])
            else:
                n_days = synparams.N_DAYS

            # Sample the perturbations of the dynamics
            syndata = synthetic.SyntheticData(log_dynamics=params.DATA_LOGSCALE, 
                n_days=n_days, perturbations_additive=params.PERTURBATIONS_ADDITIVE)
            syndata.icml_topology(n_asvs=synparams.N_ASVS, 
                max_abundance=synparams.MAX_ABUNDANCE) #, scale_interaction=1)

            if subjset_real.perturbations is not None and synparams.PERTURBATIONS:
                starts = []
                ends = []
                for perturbation in subjset_real.perturbations:
                    starts.append(perturbation.start)
                    ends.append(perturbation.end)
                syndata.icml_perturbations(starts=starts, ends=ends)
                for perturbation in syndata.dynamics.perturbations:
                    print(perturbation)

            print(syndata.dynamics.clustering.toarray())
            print('self-interactions')
            print(syndata.dynamics.self_interactions)
            print('growths')
            print(syndata.dynamics.growth)
            print(syndata.dynamics.interactions)

            # Generate the trajectories
            print('N times', synparams.TIMES)
            syndata.set_times(N=synparams.TIMES)
            print('master times', syndata.master_times)
            # syndata.master_times = np.arange(0, syndata.n_days, step=0.1)
            syndata.save(syndata_filename)

        # Generate the trajectories
        pl.seed(params.DATA_SEED)
        init_dist = pl.variables.Uniform(
            low=synparams.INIT_LOW, 
            high=synparams.INIT_HIGH)
        processvar = model.MultiplicativeGlobal(asvs=syndata.asvs)
        processvar.value = synparams.PV_VALUE
        for _ in range(synparams.N_REPLICATES):
            syndata.generate_trajectories(init_dist=init_dist, 
                dt=synparams.SIMULATION_DT, 
                processvar=processvar)


        # plt.plot(syndata.times[0], syndata.data[0])
        # plt.show()

        # logging.critical('Destroying perturbations')
        # syndata.dynamics.perturbations = None
        try:
            subjset = syndata.simulateRealRegressionDataNegBinMD(
                a0=synparams.NEGBIN_A0, a1=synparams.NEGBIN_A1, 
                qpcr_noise_scale=synparams.QPCR_NOISE_SCALE, subjset=subjset_real)
        except:
            for ridx in range(syndata.n_replicates):
                print(syndata.data[ridx])
            raise
        subjset_exact = syndata.simulateExactSubjset()
        subjset_exact.save(exact_subjset_filename)

        # Plot
        for i, subj in enumerate(subjset):
            ax = pl.visualization.abundance_over_time(subj=subj, dtype='abs', legend=True,
                taxlevel=None, set_0_to_nan=True, yscale_log=params.DATA_LOGSCALE, 
                color_code_clusters=True, clustering=syndata.dynamics.clustering)
            plt.savefig(basepath + 'data{}.pdf'.format(i))
        pl.visualization.abundance_over_time(subj=subjset, dtype='qpcr', taxlevel=None, 
            set_0_to_nan=True, yscale_log=params.DATA_LOGSCALE, clustering=syndata.dynamics.clustering)
        plt.savefig(basepath + 'qpcr.pdf')
        pl.visualization.abundance_over_time(subj=subjset, dtype='read-depth', taxlevel=None, 
            set_0_to_nan=True, yscale_log=params.DATA_LOGSCALE, clustering=syndata.dynamics.clustering)
        plt.savefig(basepath + 'read_depth.pdf')

        # Fit a lognorm distribution and see the variance.
        d = []
        for ridx in range(len(subjset_exact)):
            true = np.log(subjset_exact.iloc(ridx).matrix()['abs'])
            pred = np.log(subjset.iloc(ridx).matrix()['abs'])

            d = np.append(d,(true-pred).ravel())
        d = d[np.isfinite(d)]
        logging.info('Measurement Error: {}'.format(scipy.stats.norm.fit(d)))

        if params.QPCR_NORMALIZATION_MAX_VALUE is not None:
            subjset.normalize_qpcr(max_value=params.QPCR_NORMALIZATION_MAX_VALUE)
            logging.info('Normalizing qPCR values. Normalization constant: {:.3E}'.format(
                subjset.qpcr_normalization_factor))
            old_c_m = params.C_M
            old_v2 = params.INITIALIZATION_KWARGS[STRNAMES.FILTERING]['v2']
            params.C_M = params.C_M * subjset.qpcr_normalization_factor
            params.INITIALIZATION_KWARGS[STRNAMES.FILTERING]['v2'] *= subjset.qpcr_normalization_factor
            logging.info('Old `c_m`: {:.2E}. New `c_m`: {:.2E}'.format( 
                old_c_m, params.C_M))
            logging.info('Old `v_2`: {:.2E}. New `v2`: {:.2E}'.format( 
                old_v2, params.INITIALIZATION_KWARGS[STRNAMES.FILTERING]['v2']))
            params.INITIALIZATION_KWARGS[STRNAMES.SELF_INTERACTION_VALUE]['rescale_value'] = \
                subjset.qpcr_normalization_factor

        subjset.save(subjset_filename)
        
        if params.INITIALIZATION_KWARGS[STRNAMES.GROWTH_VALUE]['value_option'] == 'manual':
            params.INITIALIZATION_KWARGS[
                STRNAMES.GROWTH_VALUE]['value'] = syndata.dynamics.growth
        
        if params.INITIALIZATION_KWARGS[STRNAMES.SELF_INTERACTION_VALUE]['value_option'] == 'manual':
            params.INITIALIZATION_KWARGS[
                STRNAMES.SELF_INTERACTION_VALUE]['value'] = syndata.dynamics.self_interactions
            if params.QPCR_NORMALIZATION_MAX_VALUE is not None:
                params.INITIALIZATION_KWARGS[
                    STRNAMES.SELF_INTERACTION_VALUE]['value'] /= subjset.qpcr_normalization_factor
   
        if params.INITIALIZATION_KWARGS[STRNAMES.PERT_VALUE]['value_option'] == 'manual':
            value = []
            for perturbation in syndata.dynamics.perturbations:
                arr = perturbation.cluster_array()
                arr[arr == 0] = np.nan
                value.append(arr)
            params.INITIALIZATION_KWARGS[STRNAMES.PERT_VALUE]['value'] = value

        if params.INITIALIZATION_KWARGS[STRNAMES.CLUSTER_INTERACTION_VALUE]['value_option'] == 'manual':
            params.INITIALIZATION_KWARGS[
                STRNAMES.CLUSTER_INTERACTION_VALUE]['value'] = syndata.dynamics.interactions.get_values()
            if params.QPCR_NORMALIZATION_MAX_VALUE is not None:
                params.INITIALIZATION_KWARGS[
                    STRNAMES.CLUSTER_INTERACTION_VALUE]['value'] /= subjset.qpcr_normalization_factor
            params.INITIALIZATION_KWARGS[
                STRNAMES.CLUSTER_INTERACTION_VALUE]['indicators'] = \
                    syndata.dynamics.interactions.get_indicators()
        if params.INITIALIZATION_KWARGS[STRNAMES.CLUSTERING]['value_option'] == 'manual':
            params.INITIALIZATION_KWARGS[STRNAMES.CLUSTERING]['value'] = \
                syndata.dynamics.clustering.toarray()
            
        # Run the model
        chain_result = main_base.run(
            params=params, 
            graph_name=graph_name, 
            data_filename=subjset_filename,
            graph_filename=graph_filename,
            tracer_filename=tracer_filename,
            hdf5_filename=hdf5_filename,
            mcmc_filename=chain_result_filename,
            checkpoint_iter=params.CHECKPOINT,
            crash_if_error=True)
        chain_result.save(chain_result_filename)
        params.save(params_filename)

    # Validate
    chain_result = pl.inference.BaseMCMC.load(chain_result_filename)

    # Make the exact subjset and noisy one
    pl.seed(params.VALIDATION_SEED)
    syndata = synthetic.SyntheticData.load(syndata_filename)

    init_dist = pl.variables.Uniform(
        low=synparams.INIT_LOW, 
        high=synparams.INIT_HIGH)
    processvar = model.MultiplicativeGlobal(asvs=syndata.asvs)
    processvar.value = synparams.PV_VALUE

    processvar = model.MultiplicativeGlobal(asvs=syndata.asvs)
    processvar.value = synparams.PV_VALUE
    syndata.master_times = np.arange(0, syndata.n_days, step=1/8)
    syndata.generate_trajectories(init_dist=init_dist, 
        dt=synparams.SIMULATION_DT, 
        processvar=processvar)

    noisy_subjset = syndata.simulateRealRegressionDataNegBinMD(
        a0=synparams.NEGBIN_A0, a1=synparams.NEGBIN_A1, 
        qpcr_noise_scale=synparams.QPCR_NOISE_SCALE, subjset=subjset_real)
    # Remove times that are not in the original data
    times_to_remove = []
    keep_ts = chain_result.graph.data.times[0]
    for t in noisy_subjset.iloc(0).times:
        if t not in keep_ts:
            times_to_remove.append(t)
    noisy_subjset.pop_times(times_to_remove)
    noisy_subjset.save(basepath + config.VALIDATION_SUBJSET_FILENAME)
    exact_subjset = syndata.simulateExactSubjset()

    comparison = make_comparison(syndata_filename=syndata_filename, 
        exact_subjset=exact_subjset)

    # Plot the chain
    params = config.ModelConfigICML.load(params_filename)
    main_base.readify_chain(
        src_basepath=basepath,
        params=params,
        yscale_log=params.DATA_LOGSCALE, 
        center_color_for_strength=True,
        run_on_copy=True,
        plot_filtering_thresh=False,
        exact_filename=exact_subjset_filename,
        syndata=syndata_filename)
    
    main_base.validate(
        src_basepath=basepath, model=chain_result, 
        forward_sims=['sim-full'],
        yscale_log=True, run_on_copy=True,
        asv_prefix_formatter='%(index)s: (%(name)s)',
        yticklabels='(%(name)s): %(index)s',
        mp=5, comparison=comparison,
        output_dt=1/8, perturbations_additive=params.PERTURBATIONS_ADDITIVE,
        traj_error_metric=pl.metrics.PE,
        pert_error_metric=pl.metrics.RMSE,
        interaction_error_metric=pl.metrics.RMSE,
        growth_error_metric=pl.metrics.PE,
        si_error_metric=pl.metrics.PE,
        clus_error_metric=pl.metrics.variation_of_information)

    # Delete the large files 
    # os.remove(chain_result_filename)
