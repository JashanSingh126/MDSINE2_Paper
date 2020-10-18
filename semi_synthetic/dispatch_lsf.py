import numpy as np
import os
import shutil
import argparse
import sys
import pandas as pd
import time
import datetime
import re
import logging

import matplotlib.pyplot as plt
import seaborn as sns

import config
import pylab as pl

logging.basicConfig(level=logging.INFO)


priority_queues = ['vlong', 'medium', 'long', 'normal', 'big']
seed_record_fmt = '{basepath}{jobname}/' + config.RESTART_INFERENCE_SEED_RECORD
intermediate_validation_fmt = '{basepath}{jobname}/' + config.INTERMEDIATE_RESULTS_FILENAME
max_jobs_per_queue = 10

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument('--dispatch-jobs', '-dj', type=int,
        help='If 1, use bsub to submit the jobs. Else just monitor',
        dest='dispatch_jobs', default=1)
parser.add_argument('--n-samples', '-ns', type=int,
        help='Total number of Gibbs steps to do',
        dest='n_samples', default=6000)
parser.add_argument('--burnin', '-nb', type=int,
    help='Total number of burnin steps',
    dest='burnin', default=3000)
parser.add_argument('--checkpoint', '-ckpt', type=int,
    help='How often to save to disk',
    dest='checkpoint', default=200)
parser.add_argument('--basepath', '-b', type=str,
    help='Basepath to save the output', default=None,
    dest='basepath')
parser.add_argument('--data-path', '-db', type=str,
    help='Folder to lead the data from', dest='data_path')
parser.add_argument('--monitor-path', '-mb', type=str,
    help='Folder to monitor the runs', dest='monitor_path')
parser.add_argument('--monitor-time', '-mt', type=float,
    help='How often to monitor the runs (in hours)', dest='monitor_time')
parser.add_argument('--n-cpus', '-nc', type=int, 
    help='Number of CPUs to reserve', dest='n_cpus', default=1)
parser.add_argument('--n-mbs', '-nmb', type=int, 
    help='Number og MBs to reserve for the job', dest='n_mbs', default=10000)
parser.add_argument('--run-make-subjects', '-rms', dest='run_make_subjsets', default=1,
    help='Whether or not to be run make_subjsets.py', type=int)
args = parser.parse_args()

basepath = args.basepath

def make_lsf_script(jobname, logging_loc, n_cpus, queue, n_mbs, mn, pv, d, i, b, 
    burnin, n_samples, nr, co, nt, db, us, continue_str):
    my_str = '''
    #!/bin/bash
    #BSUB -J {jobname}
    #BSUB -o {logging_loc}_output.out
    #BSUB -e {logging_loc}_error.err

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

    #BSUB -q {queue}
    #BSUB -n {n_cpus}
    #BSUB -M {n_mbs}
    #BSUB -R rusage[mem={n_mbs}]

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
    source activate dispatcher_pylab301

    cd /data/cctm/darpa_perturbation_mouse_study/MDSINE2_data/MDSINE2/semi_synthetic/
    python main_mcmc.py --job-name {jobname} -m {mn} -p {pv} -d {d} -i {i} -b {b} -nb {burnin} -ns {n_samples} -nr {nr} -c {co} -nt {nt} -db {db} -us {us} {continue_str}
    '''
    if continue_str != '':
        continue_str = '--continue {}'.format(continue_str)
    return my_str.format(
        jobname=jobname,
        logging_loc=logging_loc,
        n_cpus=n_cpus,
        queue=queue,
        n_mbs=n_mbs,
        mn=mn,
        pv=pv,
        d=d,
        i=i,
        b=b,
        burnin=burnin,
        n_samples=n_samples,
        nr=nr,
        co=co,
        nt=nt,
        db=db,
        us=us,
        continue_str=continue_str)

def _outer_boxplot(df, only, x):
    fig = plt.figure(figsize=(20,10))
    ax = _inner_boxplot(df=df, only=only, x=x, y='rmse_growth',
        ax=fig.add_subplot(2,3,1), ylabel='RMSE', yscale='linear',
        title='Growth Error')
    ax = _inner_boxplot(df=df, only=only, x=x, y='rmse_interactions',
        ax=fig.add_subplot(2,3,2), ylabel='RMSE', yscale='linear',
        title='Interactions Error')
    ax = _inner_boxplot(df=df, only=only, x=x, y='topology',
        ax=fig.add_subplot(2,3,3), ylabel='AUCROC', yscale='linear',
        title='Topology Error')
    ax = _inner_boxplot(df=df, only=only, x=x, y='rmse_perturbations',
        ax=fig.add_subplot(2,3,4), ylabel='RMSE', yscale='linear',
        title='Average Perturbation error')
    ax = _inner_boxplot(df=df, only=only, x=x, y='clustering',
        ax=fig.add_subplot(2,3,5), ylabel='Normalized Mutual Information', yscale='linear',
        title='Clustering Error')
    fig.suptitle(x, fontsize=22, fontweight='bold')
    fig.tight_layout()
    fig.subplots_adjust(top=0.87)
    return fig

def _inner_boxplot(df, only, x, y, ax, ylabel, yscale, title):
    dftemp = df
    if only is not None:
        for col, val in only.items():
            dftemp = dftemp[dftemp[col] == val]

        # print(df.columns)
        # print(dftemp['Measurement Noise'])
        # sys.exit()
    
    sns.boxplot(data=dftemp, x=x, y=y, ax=ax)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_yscale(yscale)
    # ax.get_legend().remove()

    return ax


os.makedirs(basepath, exist_ok=True)

# Make dir for lsf files, outputs and error files
lsfdir = basepath + 'lsfs/'
logdir = basepath + 'logs/'
os.makedirs(lsfdir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)

meshes = config.SEMI_SYNTHETIC_MESHES

arguments_global = []

agg_repliates = set([])
agg_times = set([])
agg_measurement_noise = set([])
max_dataseeds = -1
agg_process_variances = set([])

for mesh in meshes:
    n_replicates = mesh[0]
    n_timepoints = mesh[1]
    n_data_seeds = mesh[2]
    n_init_seeds = mesh[3]
    measurement_noises = mesh[4]
    process_variances = mesh[5]
    clustering_ons = mesh[6]
    uniform_sampling = mesh[7]
    boxplot_type = mesh[8]

    for d in range(n_data_seeds):
        if d > max_dataseeds:
            max_dataseeds = d
        for i in range(n_init_seeds):
            for nr in n_replicates:
                agg_repliates.add(str(nr))
                for nt in n_timepoints:
                    agg_times.add(str(nt))
                    for mn in measurement_noises:
                        agg_measurement_noise.add(str(mn))
                        for pv in process_variances:
                            agg_process_variances.add(str(pv))
                            for co in clustering_ons:
                                arr = [nr, nt, d, i, mn, pv, uniform_sampling, boxplot_type]
                                arguments_global.append(arr)
lst_replicates = ' '.join(agg_repliates)
lst_measurement_noises = ' '.join(agg_measurement_noise)
lst_times = ' '.join(agg_times)
lst_process_variances = ' '.join(agg_process_variances)
if args.run_make_subjsets:
    command = 'python make_subjsets.py -b {basepath} -nr {nrs} -m {mns} -p {pvs} -d {nd} -dset semi-synthetic -nt {nts}'.format(
        basepath=args.data_path, nrs=lst_replicates, mns=lst_measurement_noises,
        pvs=lst_process_variances, nd=max_dataseeds+1, nts=lst_times)
    logging.info('EXECUTING: {}'.format(command))
    os.system(command)

job_names_master = {}
jobs_left = []

# Dispatch the jobs
cnt_queues = np.zeros(len(priority_queues))
for mesh in arguments_global:
    nr = mesh[0]
    nt = mesh[1]
    d = mesh[2]
    i = mesh[3]
    mn = mesh[4]
    pv = mesh[5]
    co = 1
    us = mesh[6]
    boxplot_type = mesh[7]

    # Make name
    if boxplot_type == 0:
        # Do measurement 
        jobname = 'MC{}m{}'.format(d,mn)
    elif boxplot_type == 1:
        # Do replicates
        jobname = 'MC{}r{}'.format(d,nr)
    else:
        # Do number of timepoints
        jobname = 'MC{}t{}'.format(d,nt)

    # Get the queue
    i_queue = 0
    while cnt_queues[i_queue] > max_jobs_per_queue:
        i_queue += 1
    if i_queue == len(priority_queues):
        raise ValueError('Too many jobs for how many queues you have. Add more queues ({}) or increase limit ({})'.format(
            len(priority_queues), max_jobs_per_queue))
    cnt_queues[i_queue] += 1
    queue = priority_queues[i_queue]
    lsfname = lsfdir + jobname + '.lsf'
    kwargs_to_save = {
        'jobname':jobname, 
        'logging_loc':logdir + jobname,
        'n_cpus':args.n_cpus, 'queue':queue,
        'n_mbs':args.n_mbs, 
        'mn':mn, 'pv':pv, 'd':d, 'i':i, 'b':basepath,
        'burnin':args.burnin, 'n_samples':args.n_samples, 'nr':nr, 
        'co':co, 'nt':nt, 'db':args.data_path, 'us':uniform_sampling,
        'continue_str':''}
    
    job_names_master[jobname] = kwargs_to_save
    jobs_left.append(jobname)

    f = open(lsfname, 'w')
    f.write(make_lsf_script(**kwargs_to_save))
    f.close()
    cmd = 'bsub < {}'.format(lsfname)
    logging.info(cmd)
    # if args.dispatch_jobs:
    #     os.system(cmd)

# Start monitoring 
# ----------------
# We monitor two different things:
#   (1) If jobs are finished but the inference is not finished
#   (2) Intermediate validation
# 
# We can tell if the jobs are done if they do not appear when we do bjobs. If
# it does not appear in the list then we check to see if it is done. We do this
# by checking if there are any nans in the seed table 
delete_date = r'(\s+Oct.+)'
delete_date = re.compile(delete_date)
submit_time_delete = r'(\s+SUBMIT_TIME)'
submit_time_delete = re.compile(submit_time_delete)
make_tabs = r'(\ ){1,10}'
make_tabs = re.compile(make_tabs)

monitor_basepath = args.monitor_path
if monitor_basepath[-1] != '/':
    monitor_basepath += '/'
os.makedirs(monitor_basepath, exist_ok=True)

start_new_seed = 10000
wait_time_seconds = 5 ##int(args.monitor_time * 60 *60)

while len(jobs_left) > 0:
    logging.info('starting to sleep')
    time.sleep(wait_time_seconds)

    now = datetime.datetime.now()
    date_time = now.strftime("%m.%d.%Y-%H.%M.%S")
    logging.info(date_time)

    # Make intermediate path
    monitor_path = monitor_basepath + date_time + '/'
    os.makedirs(monitor_path, exist_ok=True)

    # Get intermediate validation data
    df_master = None
    n_not_done = 0
    for job in job_names_master:
        path = intermediate_validation_fmt.format(
            basepath=basepath, jobname=job)
        try:
            df_temp = pd.read_csv(path, sep='\t')
        except Exception as e:
            df_temp = None
            n_not_done += 1
        if df_temp is not None:
            if df_master is None:
                df_master = df_temp
            else:
                df_master = df_master.append(df_temp.iloc[-1,:])

    if df_master is not None:

        # Make histogram of samples
        samples = df_master['sample_iter'].to_numpy().ravel()
        if n_not_done > 0:
            samples = np.append(samples, np.zeros(n_not_done))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(samples, bins=15)
        ax.set_title('Progress of inference')
        ax.set_xlabel('Sample iteration')
        ax.set_ylabel('Number of jobs')
        plt.savefig(monitor_path + 'sample_progress_of_inference.pdf')

        # make boxplots of intermediate results
        # Measurement noise
        try:
            fig = _outer_boxplot(df=df_master, only={'Number of Timepoints': 55, 'Number of Replicates': 5, 
                'Uniform Samples': False}, x='Measurement Noise')
            plt.savefig(monitor_path + 'noise.pdf')
            plt.close()
        except Exception as e:
            logging.info('Failed on measurement noise')
            logging.info(e)

        # # Number of replicates
        # try:
        #     fig = _outer_boxplot(df=df_master, only={'Number of Timepoints': 55, 'Measurement Noise': 0.3, 
        #         'Uniform Samples': False}, x='Number of Replicates')
        #     plt.savefig(monitor_path + 'replicates.pdf')
        #     plt.close()
        # except Exception as e:
        #     print('Failed on number of replicates')
        #     print(e)

        # # Measurement noise
        # try:
        #     fig = _outer_boxplot(df=df_master, only={'Measurement Noise': 0.3, 'Number of Replicates': 4, 
        #         'Uniform Samples': True}, x='Measurement Noise')
        #     plt.savefig(monitor_path + 'timepoints.pdf')
        #     plt.close()
        # except Exception as e:
        #     print('Failed on timepoints')
        #     print(e)



    # Check if jobs need to be restarted
    fname = monitor_path + 'bjobs.txt'
    fname_tabbed = monitor_path + 'bjobs_tabbed.txt'
    cmd = 'bjobs > {}'.format(fname)
    os.system(cmd)

    f = open(fname, 'r')
    txt = f.read()
    f.close()

    txt = delete_date.sub('', txt)
    txt = submit_time_delete.sub('', txt)
    txt = make_tabs.sub(',', txt)

    f = open(fname_tabbed, 'w')
    f.write(txt)
    f.close()

    df_bjobs = pd.read_csv(fname_tabbed, sep=',')

    # Get only the jobs that start with MC
    jobs_active_ = df_bjobs['JOB_NAME']
    jobs_active = []
    for job in jobs_active_:
        if 'MC' in job:
            jobs_active.append(job)

    if len(jobs_active) != len(jobs_left):
        print('Number of jobs active ({}) != number of jobs left ({})'.format(
            len(jobs_active), len(jobs_left)))
        for job in jobs_left:
            if job not in jobs_active:
                # This job was killed in the process and we need to check if we need to restart
                path = seed_record_fmt.format(basepath=basepath, jobname=job)
                df = pd.read_csv(path, sep='\t')
                data = df.to_numpy()
                if np.any(np.isnan(data)):
                    # This means that we do not need to restart it and we can take it off of jobs left
                    jobs_left.remove(job)
                else:
                    print('{} not found - needs to restart'.format(job))
                    # restart it with the designated dataseed, it will record it automatically
                    args = job_names_master[job]
                    args['continue_str'] = start_new_seed
                    start_new_seed += 11

                    # figure out the queue it needs to go in
                    for q in priority_queues:
                        if len(df_bjobs[df_bjobs['QUEUE'] == q]) < max_jobs_per_queue:
                            args['queue'] = q
                            break


                    lsfname = lsfdir + jobname + '_cont{}.lsf'.format(args['continue_str'])
                    f = open(lsfname, 'w')
                    f.write(make_lsf_script(**args))
                    f.close()
                    os.system('bsub < {}'.format(lsfname))
