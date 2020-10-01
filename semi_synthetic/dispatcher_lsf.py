import numpy as np
import os
import shutil
import argparse

import config

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument('--n-samples', '-ns', type=int,
        help='Total number of Gibbs steps to do',
        dest='n_samples', default=6000)
parser.add_argument('--burnin', '-nb', type=int,
    help='Total number of burnin steps',
    dest='burnin', default=3000)
parser.add_argument('--checkpoint', '-ckpt', type=int,
    help='How often to save to disk',
    dest='checkpoint', default=200)
parser.add_argument('--queue', '-q', type=str,
    help='Which queue to submit to',
    dest='queue', default='big')
parser.add_argument('--basepath', '-b', type=str,
    help='Basepath to save the output', default=None,
    dest='basepath')
parser.add_argument('--data-path', '-db', type=str,
    help='Folder to lead the data from', dest='data_path')
parser.add_argument('--n-cpus', '-nc', type=int, 
    help='Number of CPUs to reserve', dest='n_cpus', default=1)
parser.add_argument('--n-mbs', '-nmb', type=int, 
    help='Number og MBs to reserve for the job', dest='n_mbs', default=10000)
parser.add_argument('--run-make-subjects', '-rms', dest='run_make_subjsets', default=1,
    help='Whether or not to be run make_subjsets.py', type=int)
args = parser.parse_args()

basepath = args.basepath

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
module load anaconda
source activate dispatcher

cd /data/cctm/darpa_perturbation_mouse_study/MDSINE2/semi_synthetic/
python main_mcmc.py -m {mn} -p {pv} -d {d} -i {i} -b {b} -nb {burnin} -ns {n_samples} -nr {nr} -c {co} -nt {nt} -db {db} -us {us}
'''
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
    print('EXECUTING:', command)
    os.system(command)

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

    name = 'd{d}_i{i}_ns{ns}_nb{nb}_nr{nr}_m{m}_p{p}_co{co}_nt{nt}_us{us}'.format(
        d=d,i=i, ns=args.n_samples, nb=args.burnin, nr=nr, m=mn, p=pv, 
        co=co, nt=nt, us=uniform_sampling)
    lsfname = lsfdir + name + '.lsf'
    f = open(lsfname, 'w')
    f.write(my_str.format(
        jobname=jobname, 
        logging_loc=logdir + name,
        n_cpus=args.n_cpus, queue=args.queue,
        n_mbs=args.n_mbs, 
        mn=mn, pv=pv, d=d, i=i, b=basepath,
        burnin=args.burnin, n_samples=args.n_samples, nr=nr, 
        co=co, nt=nt, db=args.data_path, us=uniform_sampling))
    f.close()
    os.system('bsub < {}'.format(lsfname))




                        

