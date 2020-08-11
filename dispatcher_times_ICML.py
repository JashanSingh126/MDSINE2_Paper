'''This is a function to run many different instances on the cluster
'''
import numpy as np
import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n-asvs', '-o', type=int,
    help='Number of ASVs', dest='n_asvs', default=100)
parser.add_argument('--n-data-seeds', '-d', type=int,
    help='Number of data seeds for each noise level', 
    dest='n_data_seeds', default=5)
parser.add_argument('--n-init-seeds', '-i', type=int,
    help='Number of initialization seeds for each data seed', 
    dest='n_init_seeds', default=1)
parser.add_argument('--n-samples', '-ns', type=int,
        help='Total number of Gibbs steps to do',
        dest='n_samples', default=6000)
parser.add_argument('--burnin', '-nb', type=int,
    help='Total number of burnin steps',
    dest='burnin', default=3000)
parser.add_argument('--times', '-t', type=float,
    help='Time difference between each sample.',
    dest='times', default=[0.5,1,2], nargs='+')
parser.add_argument('--measurement-noise', '-m', type=float,
    help='Measurement noise for counts and qPCR',
    dest='measurement_noise', default=0.1)
parser.add_argument('--process-variance', '-p', type=float,
    help='Process variance', dest='process_variance', default=0.05)
parser.add_argument('--basepath', '-b', type=str,
    help='Basepath to save the output', default=None,
    dest='basepath')
parser.add_argument('--aggregate-results', '-ar', type=bool,
    help='Aggregate the results of a previous run with the above settings. Makes boxplots.',
    dest='aggregate_results', default=False)
parser.add_argument('--n-replicates', '-nr', type=int,
    help='How many replicates of data to run with.', dest='n_replicates',
    default=[5], nargs='+')
parser.add_argument('--n-cpus', '-nc', type=int, 
    help='Number of CPUs to reserve', dest='n_cpus', default=6)
parser.add_argument('--n-gbs', '-ng', type=int, 
    help='Number og GBs to reserve for the job', dest='n_gbs', default=10000)
args = parser.parse_args()

data_seed = np.arange(0,args.n_data_seeds)
init_seed = np.arange(0,args.n_init_seeds)
if args.basepath is None:
    basepath = 'times_ASVs{}/'.format(args.n_asvs)
else:
    basepath = args.basepath

my_str = '''
#!/bin/bash
#BSUB -J times_{2}_{3}
#BSUB -o {9}{3}_output.out
#BSUB -e {9}{3}_error.err

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
#BSUB -n {12}
#BSUB -M {13}
#BSUB -R rusage[mem={13}]

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
conda activate dispatcher

cd /data/cctm/darpa_perturbation_mouse_study/perturbation_study
python main_ICML.py -m {0} -p {1} -d {2} -i {3} -b {4} -n {5} -q {0} -nb {6} -ns {7} -t {8} -nr {10} -c {11}
'''

# Make the directories to store the information
if not args.aggregate_results:
    try:
        os.mkdir(basepath)
    except OSError:
        # Remove the directory and then make one
        shutil.rmtree(basepath)
        os.mkdir(basepath)

    for nr in args.n_replicates:
        replicatefolder = basepath + 'n_replicates{}/'.format(nr)
        os.mkdir(replicatefolder)

        for clustering_on in [1, 0]:
            clusteringfolder = replicatefolder + 'clustering_on{}/'.format(bool(clustering_on))
            os.mkdir(clusteringfolder)

            for t in args.times:
                timesfolder = clusteringfolder + 'time_{}/'.format(str(t).replace('.','_'))
                os.mkdir(timesfolder)

                for ds in data_seed:
                    dsfolder = timesfolder + 'data_seed_{}/'.format(ds)
                    os.mkdir(dsfolder)
                    logname = dsfolder + 'logs/'
                    os.mkdir(logname)

                    for i in init_seed:

                        fname = dsfolder + 'init_seed_{}.lsf'.format(i)
                        

                        f = open(fname, 'w')
                        f.write(my_str.format(
                            args.measurement_noise, 
                            args.process_variance,
                            ds, i, dsfolder, args.n_asvs,
                            args.burnin,
                            args.n_samples,
                            t, logname, nr, clustering_on,
                            args.n_cpus, args.n_gbs))
                        f.close()

                        os.system('bsub < {}'.format(fname))
else:
    # Check if the basepath exists
    if not os.path.isdir(basepath):
        raise ValueError('`basepath` ({}) not found'.format(basepath))
    
