'''Make the lsf file for cross validation.

WARNING: THIS FILE ONLY WORKS IF THE OS CONTAINS AN LSF JOB SUBMISSION SYSTEM.
THIS IS A INTERNAL DOCUMENT. FOR IDENTICAL RESULTS THAT DO NOT REQUIRE RUNNING 
LSF, RUN THE SCRIPT `MDSINE2/gibson_dataset/run_cv.sh`.
'''

lsfstr = '''#!/bin/bash
#BSUB -J {jobname}
#BSUB -o {lsf_files}{jobname}.out
#BSUB -e {lsf_files}{jobname}.err

#BSUB -q big-multi
#BSUB -n 4
#BSUB -M 12000
#BSUB -R rusage[mem=12000]

# 
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


# Run the job
module load anaconda/default
source activate {environment_name}
cd {code_basepath}

python ../../run_cross_validation.py \
    --dataset {dset_fileloc} \
    --cv-basepath {cv_basepath} \
    --dset-basepath {dset_basepath} \
    --negbin-run {negbin_run} \
    --seed {seed} \
    --burnin {burnin} \
    --n-samples {n_samples} \
    --checkpoint {checkpoint} \
    --multiprocessing {mp} \
    --leave-out-subject {leave_out_subject}
'''

import mdsine2 as md2
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, dest='dataset',
        help='This is the Gibson dataset we want to do cross validation on')
    parser.add_argument('--cv-basepath', '-o', type=str, dest='output_basepath',
        help='This is the basepath to save the output')
    parser.add_argument('--dset-basepath', '-db', type=str, dest='input_basepath',
        help='This is the basepath to load and save the cv datasets')
    parser.add_argument('--lsf-basepath', '-l', type=str, dest='lsf_basepath',
        help='This is the basepath to save the lsf files', default='lsf_files/')
    parser.add_argument('--leave-out-subject', '-lo', type=str, dest='leave_out_subj',
        help='This is the subject to leave out')
    parser.add_argument('--negbin-run', type=str, dest='negbin',
        help='This is the MCMC object that was run to learn a0 and a1')
    parser.add_argument('--seed', '-s', type=int, dest='seed',
        help='This is the seed to initialize the inference with')
    parser.add_argument('--burnin', '-nb', type=int, dest='burnin',
        help='How many burn-in Gibb steps for Markov Chain Monte Carlo (MCMC)')
    parser.add_argument('--n-samples', '-ns', type=int, dest='n_samples',
        help='Total number Gibb steps to perform during MCMC inference')
    parser.add_argument('--checkpoint', '-c', type=int, dest='checkpoint',
        help='How often to write the posterior to disk. Note that `--burnin` and ' \
             '`--n-samples` must be a multiple of `--checkpoint` (e.g. checkpoint = 100, ' \
             'n_samples = 600, burnin = 300)')
    parser.add_argument('--multiprocessing', '-mp', type=int, dest='mp',
        help='If 1, run the inference with multiprocessing. Else run on a single process',
        default=0)
    parser.add_argument('--environment-name', dest='environment_name', type=str,
        help='Name of the conda environment to activate when the job starts')
    parser.add_argument('--code-basepath', type=str, dest='code_basepath',
        help='Where the `run_cross_validation` script is located')

    args = parser.parse_args()
    study = md2.Study.load(args.dataset)
    dset = study.name

    # Make directory for lsf files
    os.makedirs(args.lsf_basepath, exist_ok=True)
    lsf_basepath = os.path.join(args.lsf_basepath, dset)
    os.makedirs(lsf_basepath, exist_ok=True)

    jobname = dset + '-cv' + args.leave_out_subj

    lsfname = os.path.join(lsf_basepath, jobname + '.lsf')
    f = open(lsfname, 'w')
    f.write(lsfstr.format(
        jobname=jobname, lsf_files=lsf_basepath,
        environment_name=args.environment_name, 
        code_basepath=args.code_basepath,
        dset_fileloc=args.dataset, cv_basepath=args.output_basepath,
        dset_basepath=args.input_basepath, negbin_run=args.negbin, 
        seed=args.seed, burnin=args.burnin, n_samples=args.n_samples,
        checkpoint=args.checkpoint, mp=args.mp, 
        leave_out_subject=args.leave_out_subj))
    f.close()
    os.system('bsub < {}'.format(lsfname))