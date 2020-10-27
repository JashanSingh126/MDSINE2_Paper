'''Make all of the subjects for the data seeds and noise levels.

The maximum times is set the to the number of times in the real data.

The number, start, and end of a perturbation is set with the data

'''
import numpy as np
import logging
import sys
import pandas as pd
import h5py
import inspect
import random
import copy
import os
import shutil
import math
import argparse

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as plticker

import pylab as pl

sys.path.append('..')
import synthetic
import config
import main_base
import model
from config import make_subj_name, make_syndata_base_name, make_syndata_data_name

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--basepath', '-b', type=str, 
        help='Basepath to save all of the subjects',
        dest='save_path')
    parser.add_argument('--n-replicates', '-nr', type=int,
        help='How many replicates of data to run with.', dest='n_replicates',
        default=[2,3,4,5], nargs='+')
    parser.add_argument('--measurement-noises', '-m', type=float,
        help='What measurement noises to run it at', default=[0.1, 0.2, 0.3, 0.4],
        dest='measurement_noises', nargs='+')
    parser.add_argument('--process-variances', '-p', type=float,
        help='What process varainces to run with', default=[0.05],
        dest='process_variances', nargs='+')
    parser.add_argument('--n-asvs', '-o', 
        help='Number of ASVs', dest='n_asvs', default=None)
    parser.add_argument('--n-data-seeds', '-d', type=int,
        help='Number of data seeds for each noise level', 
        dest='n_data_seeds', default=5)
    parser.add_argument('--dataset', '-dset', type=str,
        help='Dataset to produce', 
        dest='dataset')
    parser.add_argument('--validation-data-seed', '-vds', type=int,
        help='Validation dataset data seed', 
        dest='validation_data_seed', default=1111119118)
    parser.add_argument('--n-times', '-nt', type=int,
        help='Number of time points', 
        dest='n_times', default=[30, 45, 60, 75, 90], nargs='+')
    parser.add_argument('--read-depth', '-rd', type=int,
        help='Read depth', 
        dest='read_depth', default=75000)
    
    return parser.parse_args()

def make_full_objects_single_data_seed(data_seed, val_seed, params, basepath, read_depth):
    '''Make the full size subject set and validation set for a specific 
    set of dynamics.
    What we cannot subsample:
        process variances
        measurement noises

    What we can subsample:
        times
        replicates

    We subsample different ranges of replicates and times.

    **More information not different information**

    Parameters
    ----------
    data_seeds : int
    params : config.SimulationConfigBoxplots
    basepath : str
    '''
    max_times = np.max(params.TIMES)
    max_replicates = np.max(params.N_REPLICATES)

    os.makedirs(basepath, exist_ok=True)

    SUBJSET_BASE  = pl.base.SubjectSet.load(params.DATA_FILENAME)
    times = []
    for subj in SUBJSET_BASE:
        times = np.append(times, subj.times)
    _MASTER_TIMEPOINTS = np.sort(np.unique(times))
    REQUIRED_TIMEPOINTS = [_MASTER_TIMEPOINTS[0], _MASTER_TIMEPOINTS[-1]]
    for perturbation in SUBJSET_BASE.perturbations:
        REQUIRED_TIMEPOINTS.append(perturbation.start)
        REQUIRED_TIMEPOINTS.append(perturbation.end)
    

    # Take a union of all the time points
    mt = [list(subj.times) for subj in SUBJSET_BASE]
    a = []
    for l in mt:
        a = a + l
    ts = set(a)
    # MASTER_TIMES = np.sort(list(ts))
    N_DAYS = np.max(list(ts))

    pl.seed(data_seed)
    

    # Generate dynamics
    if params.DATASET == 'icml':
        syndata_base = synthetic.SyntheticData(log_dynamics=params.LOG_DYNAMICS, 
            n_days=N_DAYS, perturbations_additive=True)
        if args.real_growth:
            syndata_base.icml_topology_real(n_asvs=params.N_ASVS, max_abundance=params.MAX_ABUNDANCE)
        else:
            syndata_base.icml_topology(n_asvs=params.N_ASVS, max_abundance=params.MAX_ABUNDANCE)

        if SUBJSET_BASE.perturbations is not None:
            starts = []
            ends = []
            for perturbation in SUBJSET_BASE.perturbations:
                starts.append(perturbation.start)
                ends.append(perturbation.end)
        else:
            starts = None
            ends = None
        if params.PERTURBATIONS:
            syndata_base.icml_perturbations(starts=starts, ends=ends)
        init_dist = pl.variables.Uniform(low=params.INIT_LOW, high=params.INIT_HIGH)
    elif params.DATASET == 'semi-synthetic':
        if params.PREPROCESSED_SEMI_SYNTH_FILENAME is not None:
            syndata_base = synthetic.SyntheticData.load(params.PREPROCESSED_SEMI_SYNTH_FILENAME)
        else:
            syndata_base = synthetic.make_semisynthetic(
                chain=params.SEMI_SYNTH_CHAIN_FILENAME,
                min_bayes_factor=params.SEMI_SYNTH_MIN_BAYES_FACTOR,
                init_dist_timepoint=params.SEMI_SYNTH_FIRST_TIMEPOINT,
                hdf5_filename=params.SEMI_SYNTH_HDF5_FILENAME,
                set_times=False)
        
        init_dist = syndata_base.init_dist
    else:
        raise ValueError('`dataset` ({}) not recognized'.format(params.DATASET))
        
    # Save the dynamics for this dataseed
    syndata_base.save(make_syndata_base_name(basepath=basepath, 
        dset=params.DATASET, ds=data_seed))
    
    # Generate trajectories for data
    for uniform_sampling_timepoints in [False, True]:
        syndata = synthetic.SyntheticData.load(make_syndata_base_name(basepath=basepath, 
            dset=params.DATASET, ds=data_seed))
        pl.seed(data_seed)
        for pv_idx, pv in enumerate(params.PROCESS_STD_LEVELS):

            logging.info('PV {}. {}/{}'.format(pv, pv_idx, len(params.PROCESS_STD_LEVELS)))

            processvar = model.MultiplicativeGlobal(asvs=syndata.asvs)
            processvar.value = params.PV_VALUES[pv_idx]

            if uniform_sampling_timepoints:
                # Make a union of all the timepoints we have to generate so that we can subsample the
                # necessary times further down the pipeline
                times = []
                for nts in params.TIMES:
                    temp_t = np.arange(0, N_DAYS, step=N_DAYS/nts)
                    for i in range(len(temp_t)):
                        temp_t[i] = round(temp_t[i], 2)
                    times = np.append(times, temp_t)
                times = np.sort(np.unique(times))
                syndata.set_timepoints(times=times)
            else:
                syndata.set_timepoints(_MASTER_TIMEPOINTS)
            logging.info('Master times {}'.format(syndata.master_times))
            for _ in range(max_replicates):
                syndata.generate_trajectories(init_dist=init_dist, 
                    dt=params.SIMULATION_DT, 
                    processvar=processvar)

            # Save the dynamics for this dataseed and replicates
            syndata.save(make_syndata_data_name(basepath=basepath, 
                dset=params.DATASET, ds=data_seed, pv=pv,
                ntimes=max_times, nrep=max_replicates, 
                uniform_sampling_timepoints=uniform_sampling_timepoints))
            master_times = syndata.master_times

            # simulate reads
            pl.seed(data_seed)
            syndata.simulate_reads(a0=config.NEGBIN_A0, a1=config.NEGBIN_A1, 
                read_depth=read_depth)
            

            # Make the measurement noises
            for mn_idx, mn in enumerate(params.MEASUREMENT_NOISE_LEVELS):

                logging.info('measurement {}. {}/{}'.format(mn, mn_idx, len(params.MEASUREMENT_NOISE_LEVELS)))
                pl.seed(data_seed)
                syndata.simulate_qpcr(qpcr_noise_scale=mn)

                subjset_master = syndata.get_subjset()
                subjset_exact_master = syndata.simulateExactSubjset()

                # ax = pl.visualization.abundance_over_time(subj=subjset_master.iloc(0), dtype='abs', legend=True,
                #     taxlevel=None, set_0_to_nan=True, yscale_log=True, 
                #     color_code_clusters=True, clustering=syndata.dynamics.clustering)
                # ax.set_title('mn {}'.format(mn))

                # Make subset of times
                NTS = np.sort(params.TIMES)
                curr_times = _MASTER_TIMEPOINTS
                for nts in NTS:
                    # Subsample the times
                    temp_subjset = copy.deepcopy(subjset_master)
                    temp_exact_subjset = copy.deepcopy(subjset_exact_master)
                    logging.info('nts: {}'.format(nts))
                    if uniform_sampling_timepoints:
                        syndata.set_times_without_timeseries(N=nts, uniform_sampling=uniform_sampling_timepoints)
                    else:
                        times = np.array(curr_times)
                        times = synthetic.subsample_timepoints(times=times, N=nts, 
                            required=REQUIRED_TIMEPOINTS)
                        
                    times = syndata.master_times

                    # subsample the times
                    a = []
                    for t in master_times:
                        if t not in times:
                            a.append(t)
                    temp_subjset.pop_times(times=a)
                    temp_exact_subjset.pop_times(times=a)

                    # Subsample for all of the replicates
                    replicates = list(params.N_REPLICATES)
                    while len(replicates) > 0:
                        nrep = np.max(replicates)
                        while nrep != len(temp_subjset):
                            sid = np.random.randint(0, len(temp_subjset))
                            temp_subjset.pop_subject(sid=sid)
                            temp_exact_subjset.pop_subject(sid=sid)

                        replicates.remove(nrep)
                        temp_subjset.save(make_subj_name(basepath=basepath, data_seed=data_seed, 
                            n_asvs=params.N_ASVS, process_variance=pv, measurement_noise=mn, 
                            n_replicates=nrep, n_times=nts, uniform_sampling_timepoints=uniform_sampling_timepoints))
                        temp_exact_subjset.save(make_subj_name(basepath=basepath, data_seed=data_seed, 
                            n_asvs=params.N_ASVS, process_variance=pv, measurement_noise=mn, 
                            n_replicates=nrep, n_times=nts, exact=True, uniform_sampling_timepoints=uniform_sampling_timepoints))

            # plt.show()

            # Make validation data for data seed and process variance
            val_syndata = synthetic.SyntheticData.load(make_syndata_base_name(basepath=basepath, 
                dset=params.DATASET, ds=data_seed))
            pl.seed(val_seed)

            if uniform_sampling_timepoints:
                ts = []
                for t in params.TIMES:
                    temp_t = np.arange(0, N_DAYS, step=N_DAYS/nts)
                    for i in range(len(temp_t)):
                        temp_t[i] = round(temp_t[i], 2)
                    ts = np.append(ts, temp_t)
                VALIDATION_TIMES = np.sort(np.unique(ts))
            else:
                VALIDATION_DT = 1/8
                VALIDATION_TIMES = np.arange(N_DAYS, step=VALIDATION_DT)

            val_syndata.master_times = VALIDATION_TIMES
            val_syndata.generate_trajectories(init_dist=init_dist, 
                dt=params.SIMULATION_DT, processvar=processvar)
            val_subjset = val_syndata.simulateExactSubjset()
            val_subjset.save(config.make_val_subjset_name(basepath=basepath, 
                ds=data_seed, pv=pv))

            master_times = val_syndata.master_times

            # For each noise make a validation
            for mn_idx, mn in enumerate(params.MEASUREMENT_NOISE_LEVELS):
                a0 = params.NEGBIN_A0S[mn_idx]
                a1 = params.NEGBIN_A1S[mn_idx]
                qpcr_std = params.QPCR_NOISE_SCALES[mn_idx]

                pl.seed(val_seed)
                val_subjset = val_syndata.simulateRealRegressionDataNegBinMD( 
                    a0=a0, a1=a1, qpcr_noise_scale=qpcr_std, 
                    subjset=params.DATA_FILENAME)

                # Subsample the times
                # times = pl.util.subsample_timeseries(T=MASTER_TIMES, sizes=params.TIMES)
                NTS = np.sort(params.TIMES)
                curr_times = _MASTER_TIMEPOINTS
                for nts in NTS:
                    temp_val_subjset = copy.deepcopy(val_subjset)
                    if uniform_sampling_timepoints:
                        val_syndata.set_times_without_timeseries(N=nts, uniform_sampling=uniform_sampling_timepoints)
                    else:
                        times = np.array(curr_times)
                        times = synthetic.subsample_timepoints(times=times, N=nts, 
                            required=REQUIRED_TIMEPOINTS)

                    times = val_syndata.master_times

                    a = []
                    for t in master_times:
                        if t not in times:
                            a.append(t)

                    temp_val_subjset.pop_times(times=a)
                    temp_val_subjset.save(config.make_val_subjset_name(basepath=basepath, 
                        ds=data_seed, pv=pv, mn=mn, nt=nts, uniform_sampling_timepoints=uniform_sampling_timepoints))

if __name__ == '__main__':

    config.LoggingConfig()
    args = parse_args()
    params = config.SimulationConfigBoxplots(times=args.n_times, 
        n_replicates=args.n_replicates, n_asvs=args.n_asvs, healthy=True, 
        process_variance_levels=args.process_variances, 
        measurement_noise_levels=args.measurement_noises, dataset=args.dataset)
    for ds in range(args.n_data_seeds):
        logging.info('\n\n\nData seed {}/{}'.format(ds, args.n_data_seeds))

        make_full_objects_single_data_seed(data_seed=ds, params=params, 
            val_seed=args.validation_data_seed, basepath=args.save_path,
            read_depth=args.read_depth)

    