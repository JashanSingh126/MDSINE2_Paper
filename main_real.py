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
import scipy.stats

import matplotlib.pyplot as plt
import seaborn as sns

import pylab as pl
import config
import posterior
import base
import diversity
from names import STRNAMES
import preprocess_filtering as filtering
from preprocess_filtering import HEALTHY_SUBJECTS, UNHEALTHY_SUBJECTS

def dispatch_docker(params, fparams, mntpath, baseimgname):

    my_str = '''
    FROM python:3.7.3

    WORKDIR /usr/src/app

    COPY ./MDSINE2 ./MDSINE2/
    COPY ./PyLab ./PyLab
    RUN pip install PyLab/.

    WORKDIR MDSINE2
    
    RUN pip install --no-cache-dir -r requirements.txt
    RUN python make_real_subjset.py
    RUN mkdir output

    CMD ["python", "main_real.py", "-d", "{0}", "-i", "0", "-ns", "400", "-nb", "200", "-b", "output/" ]
    '''
    os.makedirs('dockers/', exist_ok=True)
    basepath = 'dockers/'
    for d in range(4):
        path = basepath + 'd{}/'.format(d)
        fname = path + 'Dockerfile'
        os.makedirs(path, exist_ok=True)
        f = open('../Dockerfile', 'w')
        f.write(my_str.format(d))
        f.close()
        os.system('more Dockerfile')
        imgname = baseimgname+'{}'.format(d)
        os.system('docker build -t {} ../'.format(imgname))
        os.rename('../Dockerfile', fname)
        os.system('docker run -d -v {}:/usr/src/app/MDSINE2/output {}'.format(
            mntpath, imgname))

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
    parser.add_argument('--dataset', type=str,
        help='Dataset to do inference on',
        dest='dataset', default='gibson')
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
    parser.add_argument('--checkpoint', '-ckpt', type=int,
        help='How often to save to disk',
        dest='checkpoint', default=200)
    parser.add_argument('--percent-change-clustering', '-pcc', type=float,
        help='Percent of ASVs to update during clustering every time it runs',
        default=1.0, dest='percent_change_clustering')
    parser.add_argument('--healthy', '-hy', type=int,
        help='Whether or not to use the healthy patients or not',
        default=0, dest='healthy')
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
    parser.add_argument('--use-docker', type=int,
        help='Necessary if cross_validate is True. If True, use bsub to do the dispatching. ' \
            'Else just run the jobs sequentially.', default=0, dest='use_docker')
    parser.add_argument('--continue', '-cont', type=int,
        help='Continue inference at the last place where disk is recorded and initialize with the ' \
            'data seed secified in continue', default=None, dest='continue_inference')
    
    args = parser.parse_args()
    return args

def main_leave_out_single(params, fparams, continue_inference):
    # Constants
    ONLY_PLOT = True

    ddd = _make_basepath(params=params, fparams=fparams)
    params.OUTPUT_BASEPATH = ddd['output_basepath']
    basepath = ddd['basepath']
    graph_name = ddd['graph_name']

    config.LoggingConfig() #basepath=basepath)

    chain_result_filename = basepath + config.MCMC_FILENAME
    subjset_filename = basepath + config.SUBJSET_FILENAME
    validate_subjset_filename = basepath + config.VALIDATION_SUBJSET_FILENAME
    graph_filename = basepath + config.GRAPH_FILENAME
    hdf5_filename = basepath + config.HDF5_FILENAME
    tracer_filename = basepath + config.TRACER_FILENAME
    params_filename = basepath + config.PARAMS_FILENAME
    fparams_filename = basepath + config.FPARAMS_FILENAME
    seed_restart_filename = basepath + config.RESTART_INFERENCE_SEED_RECORD

    if continue_inference is not None:
        logging.warning('CONTINUING INFERENCE FROM LAST SAVED PART AT DISK')
        if not os.path.isdir(basepath):
            raise ValueError('You want to continue inference with seed {} but the path ' \
                '{} does not exist'.format(continue_inference, basepath))
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

    else:
        pl.seed(params.DATA_SEED)
        os.makedirs(basepath, exist_ok=True) # Make the folder
        dfnew = pd.DataFrame([[0, params.DATA_SEED]], columns=['Iteration', 'Seed'])
        dfnew.to_csv(seed_restart_filename, sep='\t', index=False)

    # Load the real data and separate the subjects
    subjset = pl.SubjectSet.load(params.DATA_FILENAME)
    if fparams.DATASET == 'gibson':
        if fparams.HEALTHY == -1:
            logging.info('Run the union')
        else:
            if not fparams.HEALTHY == 1:
                sidxs = UNHEALTHY_SUBJECTS
            else:
                sidxs = HEALTHY_SUBJECTS
            subjset.pop_subject(sidxs)

    if params.LEAVE_OUT != -1:
        if params.LEAVE_OUT >= len(subjset):
            # Out of range, dont do anything
            return

    if not ONLY_PLOT:

        if continue_inference is not None:
            params = config.ModelConfigReal.load(params_filename)
            fparams = config.FilteringConfig.load(fparams_filename)
            subjset = pl.base.SubjectSet.load(subjset_filename)
        else:
            params.save(params_filename)
            fparams.save(fparams_filename)

            # Filtering and abundance normalization
            if fparams.DATASET == 'gibson':
                subjset = filtering.consistency(subjset, dtype=fparams.DTYPE,
                    threshold=fparams.THRESHOLD, union_both_consortia=fparams.HEALTHY==-1,
                    min_num_consecutive=fparams.MIN_NUM_CONSECUTIVE,
                    colonization_time=fparams.COLONIZATION_TIME,
                    min_num_subjects=fparams.MIN_NUM_SUBJECTS)
            elif fparams.DATASET == 'mdsine-cdiff':
                subjset = filtering.mdsine_cdiff_preprocess(subjset=subjset)
            else:
                raise ValueError('`dataset` ({}) not recognized'.format(fparams.DATASET))

            f = open(basepath + 'filtering_output.txt', 'w')
            for asv in subjset.asvs:
                f.write('Index: {}\n\tName: {}\n\tTaxonomy: {}\n\tSequence: {}\n'.format(asv.idx, asv.name, 
                    pl.asvname_formatter(format='%(order)s %(family)s, %(genus)s, %(species)s',
                        asv=asv, asvs=subjset.asvs),asv.sequence))
            f.close()

            # f = open('healthy_{}_asvs.txt'.format(fparams.HEALTHY), 'w')
            # for asv in subjset.asvs:
            #     f.write('{}\n'.format(asv.name))
            # f.close()
            # sys.exit()
            
            logging.info('num asvs after preprocess filtering {}'.format(len(subjset.asvs)))

            if params.MAX_N_ASVS != -1:
                subjset = get_top(subjset, max_n_asvs=params.MAX_N_ASVS)
            if params.DELETE_FIRST_TIMEPOINT:
                subjset.pop_times(0, sids='all')
            # subjset = get_asvs(subjset, asvs=['ASV_2'])
            # subjset.pop_times(times=np.arange(20,70,step=0.5), sids='all')
            # subjset.perturbations = None

            plotbasepath = basepath+'valid_asvs/'
            os.makedirs(plotbasepath, exist_ok=True) # Make the folder

        # Remove subjects if necessary
        if params.LEAVE_OUT != -1:
            if continue_inference:
                validate_subjset = pl.base.SubjectSet.load(validate_subjset_filename)
            else:
                validate_subjset = subjset.pop_subject(params.LEAVE_OUT)
                validate_subjset.save(validate_subjset_filename)
        else:
            validate_subjset = None

        # Plot data
        if continue_inference is None:
            matrixes = [subj.matrix()['abs'] for subj in subjset]
            read_depthses = [subj.read_depth() for subj in subjset]
            qpcrses = [np.sum(subj.matrix()['abs'], axis=0) for subj in subjset]

            # logging.info('Plotting Data')
            # for asv in subjset.asvs:
            #     logging.info('{}/{}'.format(asv.idx, len(subjset.asvs)))
            #     fig = plt.figure(figsize=(20,10))
            #     fig = filtering.plot_asv(
            #         subjset=subjset, asv=asv, fparams=fparams, fig=fig,
            #         legend=True, title_format='Subject %(sname)s',
            #         suptitle_format='%(index)s: %(name)s\n%(order)s, %(family)s, %(genus)s',
            #         yscale_log=True, matrixes=matrixes, read_depthses=read_depthses,
            #         qpcrses=qpcrses)
            #     plt.savefig(plotbasepath + '{}.pdf'.format(asv.name))
            #     plt.close()

            subjs = [subj for subj in subjset]
            for i, subj in enumerate(subjs):
                pl.visualization.abundance_over_time(subj=subj, dtype='abs', legend=True,
                    taxlevel='genus', set_0_to_nan=True, yscale_log=True)
                plt.savefig(basepath + 'data{}.pdf'.format(subj.name))
                plt.close()
            
            pl.visualization.abundance_over_time(subj=subjset, dtype='qpcr', include_errorbars=False, grid=True)
            fig = plt.gcf()
            fig.tight_layout()
            plt.savefig(basepath + 'qpcr_data.pdf')
            plt.close()

            pl.visualization.abundance_over_time(subj=subjset, dtype='read-depth', yscale_log=False, grid=True)
            plt.savefig(basepath + 'read_depths.pdf')
            plt.close()

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
        chain_result = base.run(
            params=params,
            graph_name=graph_name,
            data_filename=subjset_filename,
            graph_filename=graph_filename,
            tracer_filename=tracer_filename,
            hdf5_filename=hdf5_filename,
            mcmc_filename=chain_result_filename,
            checkpoint_iter=params.CHECKPOINT,
            crash_if_error=True,
            continue_inference=continue_inference,
            intermediate_validation_t=params.INTERMEDIATE_VALIDATION_T,
            intermediate_validation_kwargs=params.INTERMEDIATE_VALIDATION_KWARGS,
            intermediate_validation_func=base.mdsine2_cv_intermediate_validation_func)
        chain_result.save(chain_result_filename)
        params.save(params_filename)

    params = config.ModelConfigReal.load(params_filename)
    fparams = config.FilteringConfig.load(fparams_filename)
    chain_result = pl.inference.BaseMCMC.load(chain_result_filename)

    base.readify_chain(
        src_basepath=basepath, params=params, yscale_log=True, 
        center_color_for_strength=True, run_on_copy=False,
        asv_prefix_formatter='%(index)s: (%(name)s) %(genus)s %(species)s',
        yticklabels='(%(name)s) %(lca)s: %(index)s',
        plot_name_filtering='%(order)s, %(family)s, %(genus)s', 
        sort_interactions_by_cocluster=True, plot_filtering_thresh=False, 
        plot_gif_filtering=False)

    # # base.readify_chain_fixed_topology(src_basepath=basepath,
    # #     abund_times_start=7, abund_times_end=21,
    # #     piechart_axis_layout='auto', healthy=bool(args.healthy))
    
    # If the validation subjset exists, run the validation function
    if os.path.isfile(validate_subjset_filename):
        base.validate(
            src_basepath=basepath, model=chain_result, dst_basepath=basepath+'validation_RMSE_w_lookahead/', 
            forward_sims=['sim-full'],
            yscale_log=True, run_on_copy=False,
            asv_prefix_formatter='%(index)s: (%(name)s) %(genus)s %(species)s ',
            yticklabels='(%(name)s) %(genus)s %(species)s: %(index)s',
            mp=10, output_dt=1/8, perturbations_additive=params.PERTURBATIONS_ADDITIVE,
            traj_fillvalue=1e5,
            lookaheads=[1,3,7],
            traj_error_metric=pl.metrics.RMSE, #scipy.stats.spearmanr,
            pert_error_metric=pl.metrics.RMSE,
            interaction_error_metric=pl.metrics.RMSE,
            growth_error_metric=pl.metrics.PE,
            si_error_metric=pl.metrics.PE,
            clus_error_metric=pl.metrics.variation_of_information)

        # base.validate(
        #     src_basepath=basepath, model=chain_result, dst_basepath=basepath+'validation_RMSE/',
        #     forward_sims=['sim-full'],
        #     yscale_log=True, run_on_copy=False,
        #     asv_prefix_formatter='%(index)s: (%(name)s) %(genus)s %(species)s ',
        #     yticklabels='(%(name)s) %(genus)s %(species)s: %(index)s',
        #     mp=5, output_dt=1/8, perturbations_additive=params.PERTURBATIONS_ADDITIVE,
        #     traj_fillvalue=1e5,
        #     traj_error_metric=scipy.stats.spearmanr, #pl.metrics.PE,
        #     pert_error_metric=pl.metrics.RMSE,
        #     interaction_error_metric=pl.metrics.RMSE,
        #     growth_error_metric=pl.metrics.PE,
        #     si_error_metric=pl.metrics.PE,
        #     clus_error_metric=pl.metrics.variation_of_information)

def _make_basepath(params, fparams):
    
    basepath = params.OUTPUT_BASEPATH
    if basepath[-1] != '/':
        basepath += '/'
    basepath += fparams.suffix() + '/' + params.cv_suffix() + '/'
    graph_name = 'graph_'+ params.cv_single_suffix()
    graph_path = basepath + graph_name + '/'

    return {'graph_name':graph_name, 'output_basepath':basepath, 
        'basepath': graph_path}

def dispatch_bsub(params, fparams, seeddispatch=None, continue_inference=None):
    '''Use the parameters in `params` and `fparams` to parallelize
    leaving out each one of the subjects 

    If `seeddispatch` is None, then do cross validation.
    If `seeddispatch` is not None, then dispatch on the number of number of seeds.

    Parameters
    ----------
    params : config.ModelConfig.Real
        This is the configuration class for the model
    fparams : config.FilteringConfig
        This is the configuration class for the filtering
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
        python main_real.py -d {4} -i {5} -b {6} -ns {7} -nb {8} -hy {9} -l {10} -mo {11} 
        '''
    if seeddispatch is not None:
        params.LEAVE_OUT = -1
        for seed in range(seeddispatch):
            params.INIT_SEED = seed
            ddd = _make_basepath(params=params, fparams=fparams)
            basepath = ddd['basepath']

            os.makedirs(basepath, exist_ok=True)

            jobname = 'real{}_hy{}'.format(seed, fparams.HEALTHY)
            lsfname = basepath + 'run.lsf'
            
            f = open(lsfname, 'w')
            f.write(my_str.format(
                jobname, 
                basepath + jobname,
                params.N_CPUS, params.N_GBS, params.DATA_SEED,
                params.INIT_SEED, params.OUTPUT_BASEPATH,
                params.N_SAMPLES, params.BURNIN,
                fparams.HEALTHY, -1, params.MAX_N_ASVS))

            if continue_inference is not None:
                f.write(' --continue {}'.format(continue_inference))

            f.close()
            os.system('bsub < {}'.format(lsfname))


    else:
        for lo in range(5):
            params.LEAVE_OUT = lo
            ddd = _make_basepath(params=params, fparams=fparams)
            basepath = ddd['basepath']

            os.makedirs(basepath, exist_ok=True)

            jobname = 'real{}_hy{}'.format(lo, fparams.HEALTHY)
            lsfname = basepath + 'run.lsf'
            
            f = open(lsfname, 'w')
            f.write(my_str.format(
                jobname, 
                basepath + jobname,
                params.N_CPUS, params.N_GBS, params.DATA_SEED,
                params.INIT_SEED, params.OUTPUT_BASEPATH,
                params.N_SAMPLES, params.BURNIN,
                fparams.HEALTHY, lo, params.MAX_N_ASVS))

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
    params = config.ModelConfigReal(
        output_basepath=args.basepath, data_seed=args.data_seed, init_seed=args.init_seed,
        n_samples=args.n_samples, burnin=args.burnin, pcc=args.percent_change_clustering,
        leave_out=args.leave_out, max_n_asvs=args.max_n_asvs, cross_validate=args.cross_validate,
        use_bsub=args.use_bsub, dataset=args.dataset, checkpoint=args.checkpoint)
    fparams = config.FilteringConfig(healthy=args.healthy, dataset=args.dataset)

    if args.use_docker:
        dispatch_docker(params=params, fparams=fparams, mntpath='/mnt/disks/data', 
            baseimgname='real-test-dispatch-')
        sys.exit()

    if params.CROSS_VALIDATE == 1:
        # Do cross validation.
        if params.USE_BSUB == 1:
            dispatch_bsub(params=params, fparams=fparams, continue_inference=args.continue_inference)
        else:
            for lo in range(5):
                params.LEAVE_OUT = lo
                main_leave_out_single(params=params, fparams=fparams, continue_inference=args.continue_inference)

    else:
        # print(params.USE_BSUB)

        if params.USE_BSUB > 0:
            # How many init seeds to make
            dispatch_bsub(params=params, fparams=fparams, seeddispatch=params.USE_BSUB, continue_inference=args.continue_inference)
        else:
            main_leave_out_single(params=params, fparams=fparams, continue_inference=args.continue_inference)

    

