'''Run time lookahead prediction and a full trajectory prediction given the chain 
and data. Submit a job for every forward prediction task.

This is called after cross validation is done for the fold and is automatically
called from the lsf script in `run_cv.py`
'''

lsfstr = '''#!/bin/bash
#BSUB -J {jobname}
#BSUB -o {stdout_loc}{jobname}.out
#BSUB -e {stderr_loc}{jobname}.err

#BSUB -q {queue}
#BSUB -n {cpus}
#BSUB -M {mem}
#BSUB -R rusage[mem={mem}]

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


# Load the environment
module load anaconda/default
source activate {environment_name}
cd {code_basepath}

# Run time lookahead
python forward_sim.py \
    --chain {chain} \
    --validation {validation_path} \
    --simulation-dt {sim_dt} \
    --start {start} \
    --n_days {n_days} \
    --limit-of-detection {lim_of_detection} \
    --sim-max {sim_max} \
    --output-basepath {basepath} \
    --save-intermediate-times {save_intermed_times}
'''

import mdsine2 as md2
import argparse
import os
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chain', '-c', type=str, dest='chain',
        help='This is the path of the chain for inference or the folder that contains ' \
             'numpy arrays of the traces for the different parameters')
    parser.add_argument('--validation', type=str, dest='validation',
        help='Data to do inference with')
    parser.add_argument('--simulation-dt', type=float, dest='simulation_dt',
        help='Timesteps we go in during forward simulation', default=0.01)
    parser.add_argument('--start', type=str, dest='start',
        help='Day to start on', default=None)
    parser.add_argument('--n-days', type=str, dest='n_days',
        help='Number of days to simulate for', default=None)
    parser.add_argument('--limit-of-detection', dest='limit_of_detection',
        help='If any of the taxas have a 0 abundance at the start, then we ' \
            'set it to this value.',default=1e5)
    parser.add_argument('--sim-max', dest='sim_max',
        help='Maximum value', default=1e20)
    parser.add_argument('--output-basepath', '-o', type=str, dest='basepath',
        help='This is where you are saving the posterior renderings')

    # ErisOne parameters
    parser.add_argument('--environment-name', dest='environment_name', type=str,
        help='Name of the conda environment to activate when the job starts')
    parser.add_argument('--code-basepath', type=str, dest='code_basepath',
        help='Where the `run_cross_validation` script is located')
    parser.add_argument('--queue', '-q', type=str, dest='queue',
        help='ErisOne queue this job gets submitted to')
    parser.add_argument('--memory', '-mem', type=str, dest='memory',
        help='Amount of memory to reserve on ErisOne')
    parser.add_argument('--n-cpus', '-cpus', type=str, dest='cpus',
        help='Number of cpus to reserve on ErisOne')
    parser.add_argument('--lsf-basepath', '-l', type=str, dest='lsf_basepath',
        help='This is the basepath to save the lsf files', default='lsf_files/')

    args = parser.parse_args()
    n_days =args.n_days
    basepath = args.basepath
    os.makedirs(basepath, exist_ok=True)

    # Get all the union timepoints within this study object
    study = md2.Study.load(args.validation)
    times = []
    for subj in study:
        times.append(subj.times)
    times = np.sort(np.unique(times))

    lsfdir = args.lsf_basepath
    os.makedirs(lsfdir, exist_ok=True)
    script_path = os.path.join(lsfdir, 'scripts')
    stdout_loc = os.path.join(lsfdir, 'stdout')
    stderr_loc = os.path.join(lsfdir, 'stderr')
    os.makedirs(script_path, exist_ok=True)
    os.makedirs(stdout_loc, exist_ok=True)
    os.makedirs(stderr_loc, exist_ok=True)

    # Do time lookahead
    for start in times[:-1]:
        jobname = study.name + '-{}-{}'.format(start, n_days)

        lsfname = os.path.join(script_path, jobname + '.lsf')
        f = open(lsfname, 'w')
        f.write(lsfstr.format(
            jobname=jobname, stdout_loc=stdout_loc, stderr_loc=stderr_loc,
            queue=args.queue, cpus=args.cpus, mem=args.memory,
            environment_name=args.environment_name, code_basepath=args.code_basepath,
            chain=args.chain, validate_path=args.validation, sim_dt=args.simulation_dt,
            start=start, n_days=n_days, lim_of_detection=args.limit_of_detection,
            sim_max=args.sim_max, basepath=basepath, save_intermed_times=1))
        f.close()
        command = 'bsub < {}'.format(lsfname)
        print(command)
        os.system(command)

    # Do full simulation
    jobname = study.name + '-full'
    lsfname = os.path.join(script_path, jobname + '.lsf')
    f = open(lsfname, 'w')
    f.write(lsfstr.format(
        jobname=jobname, stdout_loc=stdout_loc, stderr_loc=stderr_loc,
        queue=args.queue, cpus=args.cpus, mem=args.memory,
        environment_name=args.environment_name, code_basepath=args.code_basepath,
        chain=args.chain, validate_path=args.validation, sim_dt=args.simulation_dt,
        start=None, n_days=None, lim_of_detection=args.limit_of_detection,
        sim_max=args.sim_max, basepath=basepath, save_intermed_times=0))
    f.close()
    command = 'bsub < {}'.format(lsfname)
    print(command)
    os.system(command)



