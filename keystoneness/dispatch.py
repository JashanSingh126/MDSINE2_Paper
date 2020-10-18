'''Save traces and dispatch all the paths in basepaths.txt with lsf files to run on the cluster.
This file will parse the chains and then dispatch the jobs

The input file should be a text file with the following format:
jobName,chain_baspath

jobName
    This is what the job will look like when running on the cluster. In addition to this we
    append the name '_{n_days}_{start}' where `n_days` is how far we are looking ahead and `start`
    is the time point we start from. Additionally, this is how we make a job sub-folder.
    `jobName/data/` is where we save the traces and `jobName/output/` is where we save the output
chain_basepath
    This path is where the output for a leave-one-out forward simulation is from main_real.
    This folder should contain 2 files:
        mcmc.pkl : This is the pylab.inference.BaseMCMC object that stores the traces
        validate_subjset.pkl : This subjectset contains the subject we use for validation

Example:
python dispatch.py --queue short --input input/basepaths.txt --n-cpus 1 --n-mbs 4000 --n-days 8 --run-jobs 1
'''
import argparse
import os
import sys
import numpy as np
import pylab as pl

sys.path.append('..')
import names

lsf_format = '''#!/bin/bash
#BSUB -J {jobname}
#BSUB -o {input_basepath}{jobname}_output.out
#BSUB -e {input_basepath}{jobname}_error.err

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

cd /data/cctm/darpa_perturbation_mouse_study/MDSINE2_data/MDSINE2/keystoneness
python keystoneness.py --type {type} --input-basepaths {input_basepath} --output-basepath {output_basepath} --simulation-dt {dt} --leave-out {leave_out} --leave-out-table {leave_out_table}
'''

if __name__ == '__main__':
    # Parse the input
    # ---------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--queue', '-q', type=str, dest='queue',
        help='This is the queue we are submitting to', default='short')
    parser.add_argument('--leave-out-table', type=str, dest='leave_out_table',
        help='This is where we are getting the data from')
    parser.add_argument('--type', type=str, dest='type',
        help='What kind of keystoneness')
    parser.add_argument('--run-jobs', type=int, dest='run_jobs',
        help='Run the jobs using lsf', default=0)
    parser.add_argument('--n-cpus', type=int, dest='n_cpus',
        help='Number of CPUs for each job', default=1)
    parser.add_argument('--n-mbs', type=int, dest='n_mbs',
        help='Number of megabytes to save for the job', default=4000)
    # parser.add_argument('--job-folder', type=str, dest='job_folder',
    #     help='Where the jobs are')
    parser.add_argument('--mcmc-paths', type=str, dest='mcmc_paths',
        help='Where the chains are')
    parser.add_argument('--output-basepath', type=str, dest='output_basepath',
        help='Base folder for the output')
    
    args = parser.parse_args()

    # Parse the chains
    # ----------------
    f = open(args.mcmc_paths, 'r')
    txt = f.read()
    f.close()

    output_basepath = args.output_basepath
    os.makedirs(output_basepath, exist_ok=True)
    if output_basepath[-1] != '/':
        output_basepath += '/'


    for line in txt.split('\n'):
        try:
            dset, path = line.split(',')
        except:
            print('line ({}) failed. skipping'.format(line))
            continue

        # Make the data folder
        dset_basepath = output_basepath + dset + '/'
        dset_input_basepath = dset_basepath + 'input/'
        dset_output_basepath = dset_basepath + 'output/'
        os.makedirs(dset_basepath, exist_ok=True)
        os.makedirs(dset_input_basepath, exist_ok=True)
        os.makedirs(dset_output_basepath, exist_ok=True)
        

        # Get the traces
        mcmc = pl.inference.BaseMCMC.load(path + 'mcmc.pkl')
        mcmc.tracer.filename = '../' + mcmc.tracer.filename

        growth = mcmc.graph[names.STRNAMES.GROWTH_VALUE].get_trace_from_disk(section='posterior')
        np.save(dset_input_basepath+'growth.npy', growth)
        si = mcmc.graph[names.STRNAMES.SELF_INTERACTION_VALUE].get_trace_from_disk(section='posterior')
        np.save(dset_input_basepath+'self_interactions.npy', si)

        interactions = mcmc.graph[names.STRNAMES.INTERACTIONS_OBJ].get_trace_from_disk(section='posterior')
        interactions[np.isnan(interactions)] = 0
        np.save(dset_input_basepath+'interactions.npy', interactions)

        perturbations = mcmc.graph.perturbations
        for pidx, pert in enumerate(perturbations):
            pert_ = pert.get_trace_from_disk(section='posterior')
            pert_[np.isnan(pert_)] = 0
            np.save(dset_input_basepath+'perturbation{}.npy'.format(pidx), pert_)

        subjset = mcmc.graph.data.subjects
        subjset.save(dset_input_basepath + 'subjset.pkl')

        # Run the chains
        # --------------
        onlyfiles = [f for f in os.listdir(dset_basepath) if os.path.isfile(dset_basepath+f)]

        for fname in onlyfiles:
            print(fname)
            fpath = fname.replace('.txt', '/')
            jobpath = dset_basepath + fpath
            lsfpath = dset_basepath + 'lsfs/'
            os.makedirs(jobpath, exist_ok=True)
            os.makedirs(lsfpath, exist_ok=True)

            f = open(dset_basepath + fname, 'r')
            lll = f.read()
            f.close()
            n_jobs = len(lll.split('\n'))

            # Calculate base
            lsfname = lsfpath + '{}_{}.lsf'.format(fname.replace('.txt', ''), None)
            f = open(lsfname, 'w')
            f.write(lsf_format.format(
                jobname=fpath.replace('/', '{}'.format(None)),
                input_basepath=dset_input_basepath,
                queue=args.queue, n_cpus=args.n_cpus, n_mbs=args.n_mbs,
                type=args.type, dt=0.1, leave_out=None,
                output_basepath=jobpath,
                leave_out_table=dset_basepath+fname))
            f.close()
            command = 'bsub < {}'.format(lsfname)
            os.system(command)

            # Calculate all others
            for i in range(n_jobs):
                lsfname = lsfpath + '{}_{}.lsf'.format(fname.replace('.txt', ''), i)
                f = open(lsfname, 'w')
                f.write(lsf_format.format(
                    jobname=fpath.replace('/', '{}'.format(i)),
                    input_basepath=dset_input_basepath,
                    queue=args.queue, n_cpus=args.n_cpus, n_mbs=args.n_mbs,
                    type=args.type, dt=0.01, leave_out=i,
                    output_basepath=jobpath,
                    leave_out_table=dset_basepath+fname))
                f.close()

                command = 'bsub < {}'.format(lsfname)
                os.system(command)


               