import numpy as np
import os
import shutil
import argparse


parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument('--n-asvs', '-o',
    help='Number of ASVs', dest='n_asvs', default=None)
parser.add_argument('--n-samples', '-ns', type=int,
        help='Total number of Gibbs steps to do',
        dest='n_samples', default=6000)
parser.add_argument('--burnin', '-nb', type=int,
    help='Total number of burnin steps',
    dest='burnin', default=3000)
parser.add_argument('--basepath', '-b', type=str,
    help='Basepath to save the output', default=None,
    dest='basepath')
parser.add_argument('--data-path', '-db', type=str,
    help='Folder to lead the data from', dest='data_path')
parser.add_argument('--n-cpus', '-nc', type=int, 
    help='Number of CPUs to reserve', dest='n_cpus', default=8)
parser.add_argument('--n-gbs', '-ng', type=int, 
    help='Number og GBs to reserve for the job', dest='n_gbs', default=10000)
args = parser.parse_args()

basepath = args.basepath

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

cd /data/cctm/darpa_perturbation_mouse_study/perturbation_study/semi_synthetic/
python main_mcmc.py -m {4} -p {5} -d {6} -i {7} -b {8} -n {9} -nb {10} -ns {11} -nr {12} -c {13} -nt {14} -db {15} -us {16}
'''
os.makedirs(basepath, exist_ok=True)

# Make dir for lsf files, outputs and error files
lsfdir = basepath + 'lsfs/'
logdir = basepath + 'logs/'
os.makedirs(lsfdir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)

meshes = [
    ([5], [55], 10, 1, [0.1, 0.2, 0.3, 0.4], [0.05], [1], 0, 0),    
    ([3,4,5], [55], 10, 1, [0.3], [0.05], [1], 0, 1), 
    ([4], [35, 45, 55, 65], 10, 1, [0.3], [0.05], [1], 1, 2)]

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
        for i in range(n_init_seeds):
            for nr in n_replicates:
                for nt in n_timepoints:
                    for mn in measurement_noises:
                        for pv in process_variances:
                    
                            for co in clustering_ons:
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

                                name = 'n{}_d{}_i{}_ns{}_nb{}_nr{}_m{}_p{}_co{}_nt{}_us{}'.format(
                                    args.n_asvs, d,i, args.n_samples, args.burnin, nr, mn, pv, 
                                    co, nt, uniform_sampling)
                                lsfname = lsfdir + name + '.lsf'
                                f = open(lsfname, 'w')
                                f.write(my_str.format(
                                    jobname, 
                                    logdir + name,
                                    args.n_cpus,
                                    args.n_gbs, 
                                    mn, pv, d, i, basepath,
                                    args.n_asvs, args.burnin, 
                                    args.n_samples, nr, 
                                    co, nt, args.data_path, uniform_sampling))
                                f.close()
                                os.system('bsub < {}'.format(lsfname))




                        

