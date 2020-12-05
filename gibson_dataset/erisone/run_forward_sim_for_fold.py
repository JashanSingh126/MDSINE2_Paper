'''Run time lookahead prediction and a full trajectory prediction given the chain 
and data. Submit a job for every forward prediction task.

This is called after cross validation is done for the fold and is automatically
called from the lsf script in `run_cv.py`
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


# Load the environment
module load anaconda/default
source activate {environment_name}

# Run time lookahead
'''

import mdsine2 as md2
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chain', '-c', type=str, dest='chain',
        help='This is the path of the chain for inference.')