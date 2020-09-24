'''This module runs real data on the model
'''

import logging
import numpy as np
import sys
import os
import os.path
import pickle
import pandas as pd
import argparse

import matplotlib.pyplot as plt
import seaborn as sns

import pylab as pl
import config
import posterior
import main_base
import diversity.alpha
from names import STRNAMES
import preprocess_filtering as filtering
from preprocess_filtering import HEALTHY_SUBJECTS, UNHEALTHY_SUBJECTS

def get_top(subjset, max_n_asvs):
    '''Delete all ASVs besides the top `MAX_N_ASVS`
    '''
    if max_n_asvs is None:
        return subjset
    totals = np.zeros(len(subjset.asvs))
    for subj in subjset:
        matrix = subj.matrix()['abs']
        totals = totals + np.sum(matrix, axis=1)

    idxs = np.argsort(totals)
    invalid_oidxs = idxs[:-max_n_asvs]
    subjset.pop_asvs(invalid_oidxs)
    return subjset

def get_asvs(subjset, asvs):
    to_delete = []
    for oidx in subjset.asvs:
        oname = subjset.asvs[oidx].name
        if oname not in asvs:
            to_delete.append(oname)
    subjset = subjset.pop_asvs(to_delete)
    return subjset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-seed', '-d', type=int,
        help='Seed to initialize the data',
        dest='data_seed')
    parser.add_argument('--init_seed', '-i', type=int,
        help='Seed to initialize the inference',
        dest='init_seed')
    parser.add_argument('--basepath', '-b', type=str,
        help='Folder to save the output', default='output_real/',
        dest='basepath')
    parser.add_argument('--n-samples', '-ns', type=int,
        help='Total number of Gibbs steps to do',
        dest='n_samples', default=4000)
    parser.add_argument('--burnin', '-nb', type=int,
        help='Total number of burnin steps',
        dest='burnin', default=2000)
    parser.add_argument('--percent-change-clustering', '-pcc', type=float,
        help='Percent of ASVs to update during clustering every time it runs',
        default=1.0, dest='percent_change_clustering')
    parser.add_argument('--leave-out', '-l', type=int,
        help='Which subject/s to leave out of the inference and to use for' \
            ' testing the predictive accuracy.',
        default=-1, dest='leave_out')
    parser.add_argument('--max-n-asvs', '-mo', type=int,
        help='Maximum number of ASVs to truncate to',
        default=-1, dest='max_n_asvs')
    parser.add_argument('--cross-validate', '-cv', type=int,
        help='Whether or not to do a cross validation',
        default=0, dest='cross_validate')
    parser.add_argument('--use-bsub', '-cvbsub', type=int,
        help='Necessary if cross_validate is True. If True, use bsub to do the dispatching. ' \
            'Else just run the jobs sequentially.', default=0, dest='use_bsub')
    parser.add_argument('--continue', '-cont', type=int,
        help='Continue the inference at a specified Gibbs step', default=None, dest='continue_inference')
    
    args = parser.parse_args()
    return args

def main_leave_out_single(params, continue_inference):
    # Constants
    ONLY_PLOT = False

    lparams = config.LoggingConfig()
    pl.seed(params.DATA_SEED)

    ddd = _make_basepath(params=params)
    params.OUTPUT_BASEPATH = ddd['output_basepath']
    basepath = ddd['basepath']
    graph_name = ddd['graph_name']

    chain_result_filename = basepath + config.MCMC_FILENAME
    subjset_filename = basepath + config.SUBJSET_FILENAME
    validate_subjset_filename = basepath + config.VALIDATION_SUBJSET_FILENAME
    graph_filename = basepath + config.GRAPH_FILENAME
    hdf5_filename = basepath + config.HDF5_FILENAME
    tracer_filename = basepath + config.TRACER_FILENAME
    params_filename = basepath + config.PARAMS_FILENAME

    if continue_inference:
        if not os.path.isdir(basepath):
            raise ValueError('You want to continue inference at GIbb step {} but the path ' \
                '{} does not exist'.format(continue_inference, basepath))
    else:
        os.makedirs(basepath, exist_ok=True) # Make the folder

    if not ONLY_PLOT:

        if continue_inference:
            params = config.ModelConfigMDSINE1.load(params_filename)
            subjset = pl.base.SubjectSet.load(subjset_filename)
        else:
            params.save(params_filename)
            subjset = pl.base.SubjectSet.load(params.DATA_FILENAME)
            logging.info('num asvs: {}'.format(len(subjset.asvs)))

            # if params.DELETE_SHORT_DATAS:
            #     if subjset.perturbations is not None:
            #         names = []
            #         for subj in subjset:
            #             if np.max(subj.times) < subjset.perturbations[0].start:
            #                 names.append(subj.name)
            #         logging.info('Deleting {}'.format(names))
            #         subjset.pop_subject(names)
            # else:
            #     raise NotImplementedError('Not implemented')
            if params.MAX_N_ASVS != -1:
                subjset = get_top(subjset, max_n_asvs=params.MAX_N_ASVS)
            if params.DELETE_FIRST_TIMEPOINT:
                subjset.pop_times(0, sids='all')
            # subjset = get_asvs(subjset, asvs=['ASV_2'])
            # subjset.pop_times(times=np.arange(20,70,step=0.5), sids='all')
            # subjset.perturbations = None

            plotbasepath = basepath+'valid_strains/'
            os.makedirs(plotbasepath, exist_ok=True) # Make the folder

        if params.LEAVE_OUT != -1:
            if params.LEAVE_OUT >= len(subjset):
                # Out of range, dont do anything
                return

        # Filter asvs
        if params.DATASET == 'cdiff':
            # Delete the required ASVs
            asvs_to_keep = [
                'Clostridium-hiranonis',
                'Proteus-mirabilis',
                'Bacteroides-ovatus',
                'Bacteroides-vulgatus',
                'Roseburia-hominis',
                'Parabacteroides-distasonis',
                'Akkermansia-muciniphila',
                'Clostridium-difficile',
                'Bacteroides-fragilis',
                'Klebsiella-oxytoca',
                'Clostridium-ramosum',
                'Escherichia-coli',
                'Ruminococcus-obeum',
                'Clostridium-scindens']

            to_delete = []
            for asv in subjset.asvs:
                if asv.name not in asvs_to_keep:
                    to_delete.append(asv.name)
            subjset.pop_asvs(to_delete)

            cdiff = subjset.asvs['Clostridium-difficile']
            cdiff_aidx = cdiff.idx

            # Set the abundance of C. Diff to 0 before day 28 in all of the subjects
            for subj in subjset:
                for t in subj.times:
                    if t < 28:
                        subj.reads[t][cdiff_aidx] = 0
                    print(t, subj.reads[t][cdiff_aidx])

        else:
            raise NotImplementedError('filtering for `{}` not recognized'.format(params.DATASET))

        # Remove subjects if necessary
        if params.LEAVE_OUT != -1:
            if continue_inference:
                validate_subjset = pl.base.SubjectSet.load(validate_subjset_filename)
            else:
                validate_subjset = subjset.pop_subject(params.LEAVE_OUT)
                validate_subjset.save(validate_subjset_filename)
        else:
            validate_subjset = None

        if continue_inference is None:

            # # Plot data
            # logging.info('Plotting Data')
            # for asv in subjset.asvs:
            #     fig = plt.figure()
            #     ax = fig.add_subplot(111)

            #     for sidx, subj in enumerate(subjset):
            #         if sidx == 0:
            #             perts = True
            #         else:
            #             perts = False
            #         pl.visualization.abundance_over_time(subj=subj, dtype='abs', legend=False,
            #             title=asv.name, plot_specific=asv,
            #             set_0_to_nan=False, yscale_log=True, ax=ax, shade_perturbations=perts)
            #     plt.savefig(basepath + 'valid_strains/{}.pdf'.format(asv.name))
            #     plt.close()

            # pl.visualization.abundance_over_time(subj=subjset, dtype='read-depth', yscale_log=False, grid=True)
            # plt.savefig(basepath + 'read_depths.pdf')
            # plt.close()

            # pl.visualization.abundance_over_time(subj=subjset, dtype='qpcr', include_errorbars=False, grid=True)
            # fig = plt.gcf()
            # fig.tight_layout()
            # plt.savefig(basepath + 'qpcr_data.pdf')
            # plt.close()

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
            crash_if_error=True,
            continue_inference=continue_inference)
        chain_result.save(chain_result_filename)
        params.save(params_filename)

    params = config.ModelConfigMDSINE1.load(params_filename)
    chain_result = pl.inference.BaseMCMC.load(chain_result_filename)

    main_base.readify_chain(
        src_basepath=basepath,
        params=params,
        yscale_log=True,
        center_color_for_strength=True,
        run_on_copy=True,
        asv_prefix_formatter='%(index)s: (%(name)s)',
        yticklabels='(%(name)s): %(index)s',
        plot_name_filtering='%(name)s',
        sort_interactions_by_cocluster=True,
        plot_filtering_thresh=False,
        plot_gif_filtering=False)

    # If the validation subjset exists, run the validation function
    if os.path.isfile(validate_subjset_filename):
        main_base.validate(
            src_basepath=basepath, model=chain_result,
            forward_sims=['sim-full'],
            yscale_log=True, run_on_copy=False,
            asv_prefix_formatter='%(index)s: (%(name)s)',
            yticklabels='(%(name)s): %(index)s',
            mp=5, output_dt=1/8, perturbations_additive=params.PERTURBATIONS_ADDITIVE,
            network_topology_metric=pl.metrics.rocauc_posterior_interactions,
            network_topology_metric_kwargs={
                'signed': False,
                'average': 'weighted'},
            traj_error_metric=pl.metrics.logPE,
            pert_error_metric=pl.metrics.RMSE,
            interaction_error_metric=pl.metrics.RMSE,
            growth_error_metric=pl.metrics.RMSE,
            si_error_metric=pl.metrics.RMSE,
            clus_error_metric=pl.metrics.variation_of_information)

def _make_basepath(params):
    
    basepath = params.OUTPUT_BASEPATH
    if basepath[-1] != '/':
        basepath += '/'
    basepath += params.cv_suffix() + '/'
    graph_name = 'graph_'+ params.cv_single_suffix()
    graph_path = basepath + graph_name + '/'

    return {'graph_name':graph_name, 'output_basepath':basepath, 
        'basepath': graph_path}

def dispatch_bsub(params, seeddispatch=None, continue_inference=None):
    '''Use the parameters in `params` to parallelize
    leaving out each one of the subjects 

    If `seeddispatch` is None, then do cross validation.
    If `seeddispatch` is not None, then dispatch on the number of number of seeds.

    Parameters
    ----------
    params : config.ModelConfigMDSINE1
        This is the configuration class for the model
    '''
    params.CROSS_VALIDATE = None

    my_str = '''
        #!/bin/bash
        #BSUB -J {0}
        #BSUB -o {1}_output.out
        #BSUB -e {1}_error.err

        # This is a sample script with specific resource requirements for the
        # **bigmemory** queue with 64GB memory requirement and memory
        # limit settings, which are both needed for reservations of
        # more than 40GB.
        # Copy this script and then submit job as follows:
        # ---
        # cd ~/lsf
        # cp templates/bsub/example_8CPU_bigmulti_64GB.lsf .
        # bsub < example_bigmulti_8CPU_64GB.lsf
        # ---
        # Then look in the ~/lsf/output folder for the script log
        # that matches the job ID number

        # Please make a copy of this script for your own modifications

        #BSUB -q big-multi
        #BSUB -n {2}
        #BSUB -M {3}
        #BSUB -R rusage[mem={3}]

        # Some important variables to check (Can be removed later)
        echo '---PROCESS RESOURCE LIMITS---'
        ulimit -a
        echo '---SHARED LIBRARY PATH---'
        echo $LD_LIBRARY_PATH
        echo '---APPLICATION SEARCH PATH:---'
        echo $PATH
        echo '---LSF Parameters:---'
        printenv | grep '^LSF'
        echo '---LSB Parameters:---'
        printenv | grep '^LSB'
        echo '---LOADED MODULES:---'
        module list
        echo '---SHELL:---'
        echo $SHELL
        echo '---HOSTNAME:---'
        hostname
        echo '---GROUP MEMBERSHIP (files are created in the first group listed):---'
        groups
        echo '---DEFAULT FILE PERMISSIONS (UMASK):---'
        umask
        echo '---CURRENT WORKING DIRECTORY:---'
        pwd
        echo '---DISK SPACE QUOTA---'
        df .
        echo '---TEMPORARY SCRATCH FOLDER ($TMPDIR):---'
        echo $TMPDIR

        # Add your job command here
        # Load module
        module load anaconda
        source activate dispatcher

        cd /data/cctm/darpa_perturbation_mouse_study/perturbation_study/
        python main_real.py -d {4} -i {5} -b {6} -ns {7} -nb {8} -l {10} -mo {11} 
        '''
    if seeddispatch is not None:
        params.LEAVE_OUT = -1
        for seed in range(seeddispatch):
            params.INIT_SEED = seed
            ddd = _make_basepath(params=params)
            basepath = ddd['basepath']

            os.makedirs(basepath, exist_ok=True)

            jobname = 'real{}_MDSINE1_{}'.format(seed, None)
            lsfname = basepath + 'run.lsf'
            
            f = open(lsfname, 'w')
            f.write(my_str.format(
                jobname, 
                basepath + jobname,
                params.N_CPUS, params.N_GBS, params.DATA_SEED,
                params.INIT_SEED, params.OUTPUT_BASEPATH,
                params.N_SAMPLES, params.BURNIN,
                None, -1, params.MAX_N_ASVS))

            if continue_inference is not None:
                f.write(' --continue {}'.format(continue_inference))

            f.close()
            os.system('bsub < {}'.format(lsfname))


    else:
        for lo in range(5):
            params.LEAVE_OUT = lo
            ddd = _make_basepath(params=params)
            basepath = ddd['basepath']

            os.makedirs(basepath, exist_ok=True)

            jobname = 'real{}_MDSINE1_{}'.format(lo, None)
            lsfname = basepath + 'run.lsf'
            
            f = open(lsfname, 'w')
            f.write(my_str.format(
                jobname, 
                basepath + jobname,
                params.N_CPUS, params.N_GBS, params.DATA_SEED,
                params.INIT_SEED, params.OUTPUT_BASEPATH,
                params.N_SAMPLES, params.BURNIN,
                None, lo, params.MAX_N_ASVS))

            f.close()
            os.system('bsub < {}'.format(lsfname))

if __name__ == '__main__':
    '''Build the parameters of the model and run

    1. Set up logging
    2. Load the data
    3. Filter and threshold
    4. qPCR normalization (for numerical stability)
    5. Plot (Optional)
    6. Specify parameters of the model
    7. Run
    '''
    args = parse_args()

    if args.data_seed is None:
        args.data_seed = 9880035
    if args.init_seed is None:
        args.init_seed = 12114738

    if args.n_samples <= args.burnin:
        raise ValueError('`n_samples` ({}) must be larger than burnin ({})'.format(
            args.n_samples, args.burnin))

    lparams = config.LoggingConfig()
    params = config.ModelConfigMDSINE1(
        output_basepath=args.basepath, data_seed=args.data_seed, init_seed=args.init_seed,
        n_samples=args.n_samples, burnin=args.burnin, pcc=args.percent_change_clustering,
        leave_out=args.leave_out, max_n_asvs=args.max_n_asvs, cross_validate=args.cross_validate,
        use_bsub=args.use_bsub)

    if params.CROSS_VALIDATE == 1:
        # Do cross validation.
        if params.USE_BSUB == 1:
            dispatch_bsub(params=params, continue_inference=args.continue_inference)
        else:
            for lo in range(5):
                params.LEAVE_OUT = lo
                main_leave_out_single(params=params, continue_inference=args.continue_inference)

    else:
        # print(params.USE_BSUB)

        if params.USE_BSUB > 0:
            # How many init seeds to make
            dispatch_bsub(params=params, seeddispatch=params.USE_BSUB, continue_inference=args.continue_inference)
        else:
            main_leave_out_single(params=params, continue_inference=args.continue_inference)

    