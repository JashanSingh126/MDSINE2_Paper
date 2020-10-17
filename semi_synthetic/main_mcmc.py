b'''This module runs MCMC inference on the synthetic data
'''

import logging
import numpy as np
import sys
import os
import pickle
import copy
import random
import pandas as pd
import argparse

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
from sklearn.metrics import normalized_mutual_info_score

import pylab as pl
import config

sys.path.append('..')
import synthetic
import main_base
from names import STRNAMES
import preprocess_filtering as filtering
import model
import data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-name', '-jn', type=str,
        help='Name of the job',
        dest='job_name', default=None)
    parser.add_argument('--data-seed', '-d', type=int,
        help='Seed to initialize the data',
        dest='data_seed', default=None)
    parser.add_argument('--init_seed', '-i', type=int,
        help='Seed to initialize the inference',
        dest='init_seed', default=None)
    parser.add_argument('--measurement-noise-level', '-m', type=float,
        help='Amount of measurement noise for the counts',
        default=0.1, dest='measurement_noise_level')
    parser.add_argument('--process-variance-level', '-p', type=float,
        help='Amount of process variance',
        default=0.05, dest='process_variance_level')
    parser.add_argument('--basepath', '-b', type=str,
        help='Folder to save the output', default='output/',
        dest='basepath')
    parser.add_argument('--data-path', '-db', type=str,
        help='Folder to lead the data from', dest='data_path')
    parser.add_argument('--n-asvs', '-n',
        help='Number of ASVs', default=None,
        dest='n_asvs')
    parser.add_argument('--n-samples', '-ns', type=int,
        help='Total number of Gibbs steps to do',
        dest='n_samples', default=4000)
    parser.add_argument('--burnin', '-nb', type=int,
        help='Total number of burnin steps',
        dest='burnin', default=2000)
    parser.add_argument('--checkpoint', '-ckpt', type=int,
        help='When to save to disk',
        dest='checkpoint', default=200)
    parser.add_argument('--n-times', '-nt', type=int,
        help='Number of time points', 
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
    parser.add_argument('--uniform-sampling', '-us', type=int,
        help='Whether or not to use uniform sampling or not',
        default=0, dest='uniform_sampling')
    parser.add_argument('--continue', '-cont', type=int,
        help='Continue inference at the last place where disk is recorded and initialize with the ' \
            'data seed secified in continue', default=None, dest='continue_inference')
    args = parser.parse_args()
    args.clustering_on = bool(args.clustering_on)
    args.uniform_sampling = bool(args.uniform_sampling)

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
    if args.n_samples <= args.burnin:
        raise ValueError('`n_samples` ({}) must be larger than burnin ({})'.format(
            args.n_samples, args.burnin))
    continue_inference = args.continue_inference

    # Constants
    ONLY_PLOT = False
    
    # Start
    ##############################
    synparams = config.SimulationConfig(times=args.n_times, n_replicates=args.n_replicates,
        n_asvs=args.n_asvs, healthy=args.healthy, uniform_sampling_timepoints=args.uniform_sampling,
        process_variance_level=args.process_variance_level,
        measurement_noise_level=args.measurement_noise_level)
    params = config.ModelConfigMCMC(output_basepath=args.basepath, data_seed=args.data_seed,
        data_path=args.data_path, checkpoint=args.checkpoint,
        init_seed=args.init_seed, a0=synparams.NEGBIN_A0, a1=synparams.NEGBIN_A1,
        n_samples=args.n_samples, burnin=args.burnin, pcc=args.percent_change_clustering,
        clustering_on=args.clustering_on)
 
    graph_name = args.job_name #'graph'+ params.suffix() + synparams.suffix()
    basepath = params.OUTPUT_BASEPATH + graph_name + '/'
    config.LoggingConfig() #basepath=basepath)

    logging.info('Init settings')
    logging.info('Data seed: {}'.format(args.data_seed))
    logging.info('Init seed: {}'.format(args.init_seed))
    logging.info('measurement noise level: {}'.format(args.measurement_noise_level))
    logging.info('process variance level: {}'.format(args.process_variance_level))
    logging.info('Number of time points: {}'.format(args.n_times))
    logging.info('Number of replicates: {}'.format(args.n_replicates))
    logging.info('basepath: {}'.format(basepath))
    logging.info('Uniform sampling: {}'.format(args.uniform_sampling))
    logging.info('n_asvs: {}'.format(args.n_asvs))
    
    chain_result_filename = basepath + config.MCMC_FILENAME
    subjset_filename = basepath + config.SUBJSET_FILENAME
    validate_subjset_filename = basepath + config.VALIDATION_SUBJSET_FILENAME
    graph_filename = basepath + config.GRAPH_FILENAME
    hdf5_filename = basepath + config.HDF5_FILENAME
    tracer_filename = basepath + config.TRACER_FILENAME
    params_filename = basepath + config.PARAMS_FILENAME
    syndata_filename = basepath + config.SYNDATA_FILENAME
    synparams_filename = basepath + config.SYNPARAMS_FILENAME
    comparison_filename = basepath + config.COMPARISON_FILENAME
    seed_restart_filename = basepath + config.RESTART_INFERENCE_SEED_RECORD

    if continue_inference is None:
        os.makedirs(basepath, exist_ok=True) # Make the folder
        pl.seed(params.DATA_SEED)   
        exact_filename = config.make_subj_name(basepath=params.DATA_PATH, data_seed=params.DATA_SEED, 
            n_asvs=args.n_asvs, process_variance=args.process_variance_level, uniform_sampling_timepoints=args.uniform_sampling,
            measurement_noise=args.measurement_noise_level, n_replicates=args.n_replicates, 
            n_times=args.n_times, exact=True)

        syndata = synthetic.SyntheticData.load(config.make_syndata_base_name(
            basepath=params.DATA_PATH, dset=synparams.DSET, ds=params.DATA_SEED))
        subjset = pl.base.SubjectSet.load(config.make_subj_name(
            basepath=params.DATA_PATH, data_seed=params.DATA_SEED, n_asvs=args.n_asvs, 
            process_variance=args.process_variance_level, measurement_noise=args.measurement_noise_level,
            n_replicates=args.n_replicates, n_times=args.n_times, uniform_sampling_timepoints=args.uniform_sampling))

        # Save these locally
        syndata.save(syndata_filename)
        subjset.save(subjset_filename)

        exact_subjset = pl.base.SubjectSet.load(config.make_val_subjset_name(
            basepath=params.DATA_PATH, ds=params.DATA_SEED, 
            pv=args.process_variance_level))
        comparison = make_comparison(syndata_filename=syndata_filename, 
            exact_subjset=exact_subjset)
        comparison.save(comparison_filename)

        dfnew = pd.DataFrame([[0, params.DATA_SEED]], columns=['Iteration', 'Seed'])
        dfnew.to_csv(seed_restart_filename, sep='\t', index=False)
    else:
        logging.warning('CONTINUING INFERENCE FROM LAST SAVED PART AT DISK')
        if not os.path.isdir(basepath):
            raise ValueError('You want to continue inference with seed {} but the path ' \
                '{} does not exist'.format(continue_inference, basepath))

        # check if the inference is finished yet
        df_old = pd.read_csv(seed_restart_filename, sep='\t')
        if np.any(np.isnan(df_old.to_numpy())):
            # This means it is finished. Set ONLY_PLOT to true
            ONLY_PLOT = True
            logging.warning('Inference is finished. Only plotting')
        else:
            # Set the seed we are starting at
            pl.seed(continue_inference)

            # Get iteration we are starting at
            mcmc = pl.inference.BaseMCMC.load(chain_result_filename)
            iter_start = mcmc.tracer.get_disk_trace_iteration()
            logging.info('restarting inference at point {}'.format(iter_start))


            dfnew = pd.DataFrame([[iter_start, continue_inference]], columns=['Iteration', 'Seed'])
            df_old = pd.read_csv(seed_restart_filename, sep='\t')
            df = df_old.append(dfnew)
            df.to_csv(seed_restart_filename, sep='\t', index=False)

            # Set continue inference to `iter_start`
            continue_inference = iter_start

            syndata = synthetic.SyntheticData.load(syndata_filename)
            subjset = pl.base.SubjectSet.load(subjset_filename)
            comparison = pl.inference.BaseMCMC.load(comparison_filename)

    if not ONLY_PLOT:
        if continue_inference is None:
            os.makedirs(basepath, exist_ok=True) # Make the folder
            if params.INITIALIZATION_KWARGS[STRNAMES.CLUSTERING]['value_option'] == 'manual':
                syn = synthetic.SyntheticData.load(config.make_syndata_base_name(
                    basepath=params.DATA_PATH, dset=synparams.DSET, ds=params.DATA_SEED))
                clustering = syn.dynamics.clustering
                arr = clustering.toarray()

                logging.warning('Fixed clustering set:')
                logging.warning('array: {}'.format(arr))

                params.INITIALIZATION_KWARGS[STRNAMES.CLUSTERING]['value'] = arr

            # Set the validation arg for intermediate_validation_kwargs
            params.INTERMEDIATE_VALIDATION_KWARGS = {
                'comparison_filename': comparison_filename,
                'ds': params.DATA_SEED,
                'mn': synparams.MEASUREMENT_NOISE_LEVEL,
                'pv': synparams.PROCESS_VARIANCE_LEVEL,
                'nt': synparams.TIMES,
                'nr': synparams.N_REPLICATES,
                'us': synparams.UNIFORM_SAMPLING_TIMEPOINTS}

            params.save(params_filename)
            synparams.save(synparams_filename)

            # Plot
            for i, subj in enumerate(subjset):
                ax = pl.visualization.abundance_over_time(subj=subj, dtype='abs', legend=True,
                    taxlevel=None, set_0_to_nan=True, yscale_log=params.DATA_LOGSCALE, lca=False,
                    color_code_clusters=True, clustering=syndata.dynamics.clustering, )
                plt.savefig(basepath + 'data{}.pdf'.format(i))
            pl.visualization.abundance_over_time(subj=subjset, dtype='qpcr', taxlevel=None, 
                set_0_to_nan=True, yscale_log=params.DATA_LOGSCALE, clustering=syndata.dynamics.clustering)
            plt.savefig(basepath + 'qpcr.pdf')
            pl.visualization.abundance_over_time(subj=subjset, dtype='read-depth', taxlevel=None, 
                set_0_to_nan=True, yscale_log=params.DATA_LOGSCALE, clustering=syndata.dynamics.clustering)
            plt.savefig(basepath + 'read_depth.pdf')

            logging.info('Measurement noise level: {}'.format(args.measurement_noise_level))
            exact = pl.base.Subject.load(exact_filename)
            d = []
            for ridx in range(len(subjset)):
                e = np.log(exact.iloc(ridx).matrix()['abs'])
                s = np.log(subjset.iloc(ridx).matrix()['abs'])

                d = np.append(d, (e-s).ravel())
            d = d[np.isfinite(d)]
            emp_noise = scipy.stats.norm.fit(d)
            logging.info('Empirical Noise: {}'.format(emp_noise))

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
        else:
            params = config.ModelConfigMCMC.load(params_filename)
            synparams = config.SimulationConfig.load(synparams_filename)        
            
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
            continue_inference=continue_inference,
            crash_if_error=True,
            intermediate_validation_func=main_base.semi_synthetic_intermediate_validation_func,
            intermediate_validation_t=params.INTERMEDIATE_VALIDATION_T,
            intermediate_validation_kwargs=params.INTERMEDIATE_VALIDATION_KWARGS)
        chain_result.save(chain_result_filename)
        params.save(params_filename)

        # Mark that we have finished the inference
        dfnew = pd.DataFrame([[np.nan, np.nan]], columns=['Iteration', 'Seed'])
        df = pd.read_csv(seed_restart_filename, sep='\t')
        df = df.append(dfnew)
        df.to_csv(seed_restart_filename, sep='\t', index=False)

    params = config.ModelConfigMCMC.load(params_filename)
    # main_base.readify_chain(
    #     src_basepath=basepath, 
    #     params=params,
    #     yscale_log=params.DATA_LOGSCALE, 
    #     center_color_for_strength=True,
    #     run_on_copy=False,
    #     plot_filtering_thresh=False,
    #     exact_filename=exact_filename,
    #     syndata=syndata_filename)
    
    chain_result = pl.inference.BaseMCMC.load(chain_result_filename)
    noise_subjset = pl.base.SubjectSet.load(config.make_val_subjset_name(
        basepath=params.DATA_PATH, ds=params.DATA_SEED, 
        pv=args.process_variance_level, mn=args.measurement_noise_level,
        nt=args.n_times, uniform_sampling_timepoints=args.uniform_sampling))
    noise_subjset.save(basepath + config.VALIDATION_SUBJSET_FILENAME)

    main_base.validate(
        src_basepath=basepath, model=chain_result, 
        forward_sims=['sim-full'],
        yscale_log=True, run_on_copy=True,
        asv_prefix_formatter='%(index)s: (%(name)s)',
        yticklabels='(%(name)s): %(index)s',
        mp=None, comparison=comparison, 
        perturbations_additive=params.PERTURBATIONS_ADDITIVE,
        traj_error_metric=scipy.stats.spearmanr,
        network_topology_metric=pl.metrics.rocauc_posterior_interactions,
        network_topology_metric_kwargs={
            'signed': synparams.TOPOLOGY_METRIC_SIGNED,
            'average': synparams.TOPOLOGY_METRIC_AVERAGE},
        pert_error_metric=pl.metrics.RMSE,
        interaction_error_metric=pl.metrics.RMSE,
        growth_error_metric=pl.metrics.RMSE,
        si_error_metric=pl.metrics.RMSE,
        traj_fillvalue=params.C_M/2,
        lookaheads=None,
        clus_error_metric=normalized_mutual_info_score)

    # Delete the large files 
    # os.remove(chain_result_filename)
