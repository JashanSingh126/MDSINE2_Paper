import mdsine2 as md2
import argparse
import logging
import os
import sys

lsfstr = '''#!/bin/bash
#BSUB -J {dset}
#BSUB -o {lsf_files}{dset}.out
#BSUB -e {lsf_files}{dset}.err

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
#BSUB -n 4
#BSUB -M 12000
#BSUB -R rusage[mem=12000]

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
module load anaconda/default

source activate mdsine2_403
cd /data/cctm/darpa_perturbation_mouse_study/MDSINE2
python gibson_4_mdsine2_inference.py \
    --input {input_dataset} \
    --negbin-run gibson_output/output/negbin/replicates/mcmc.pkl \
    --seed 0 \
    --burnin  5000 \
    --n-samples 15000 \
    --checkpoint 100 \
    --basepath {output_basepath}''' 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, dest='dataset',
        help='This is the Gibson dataset we want to do cross validation on')
    parser.add_argument('--output-basepath', '-o', type=str, dest='output_basepath',
        help='This is the basepath to save the output',
        default='gibson_output/output/mdsine2/cv/')
    parser.add_argument('--input-basepath', '-i', type=str, dest='input_basepath',
        help='This is the basepath to load and save the cv datasets',
        default='gibson_output/datasets/cv/')
    parser.add_argument('--lsf-basepath', '-l', type=str, dest='lsf_basepath',
        help='This is the basepath to save the lsf files',
        default='lsf_files/')
    args = parser.parse_args()

    md2.config.LoggingConfig(level=logging.INFO)

    input_basepath = args.input_basepath
    output_basepath = args.output_basepath
    lsf_basepath = args.lsf_basepath
    os.makedirs(input_basepath, exist_ok=True)
    os.makedirs(lsf_basepath, exist_ok=True)
    os.makedirs(output_basepath, exist_ok=True)

    logging.info('Loading dataset {}'.format(args.dataset))
    study_master = md2.Study.load(args.dataset)

    for subj in study_master:
        logging.info('Leave out {}'.format(subj.name))
        study = md2.Study.load(args.dataset)
        val_study = study.pop_subject(subj.name)
        study.name = study.name + '-cv{}'.format(subj.name)

        print('Study |||', study.name, '||| ', study.names())

        logging.info('Save the validation Study')
        val_study.name = study.name + '-validate'

        print('val_study |||', val_study.name, '||| ', val_study.names())

        input_dataset = input_basepath + study.name + '.pkl'
        study.save(input_dataset)
        val_study.save(input_basepath + val_study.name + '.pkl')

        logging.info('Make the lsf file')
        lsfname = lsf_basepath + study.name + '.lsf'
        f = open(lsfname, 'w')
        f.write(lsfstr.format(
            dset=study.name,
            lsf_files=lsf_basepath,
            input_dataset=input_dataset,
            output_basepath=output_basepath))
        f.close()

        os.system('bsub < {}'.format(lsfname))
