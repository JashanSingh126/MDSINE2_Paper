import numpy as np
import logging
import sys
import scipy.stats
import scipy.sparse
import scipy.spatial
from scipy.cluster.hierarchy import linkage
import scipy
import sklearn
import numba
import time
import collections
import pandas as pd
import h5py
import inspect
import random
import copy
import os
import shutil
import math
import argparse
import itertools
import re
import pickle
import datetime 

from multiprocessing import Pool

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as plticker
import matplotlib.patches as patches

print('import pylab')
import pylab as pl
print('import synthetic')
import synthetic
print('import diversity')
import diversity
print('import config')
import config
print('import filtering')
import preprocess_filtering as filtering
print('import model')
import model
print('import names')
import names
print('import main_base')
import main_base
print('import metrics')
import metrics
print('import util')
import util as MDSINE2_util

import ete3
import Bio
from Bio import Phylo
from Bio import SeqIO, AlignIO

# import ray
import psutil
import scipy.signal
# import torch
# import main_base

import io
from sklearn.metrics.cluster import normalized_mutual_info_score

config.LoggingConfig()
logging.basicConfig(level=logging.INFO)
####################################################
# Submit keystoneness jobs 
####################################################
my_str = '''
#!/bin/bash
#BSUB -J {RRR}_{consortium}_{start}_{end}
#BSUB -o {basepath}{consortium}_{start}_{end}_output.out
#BSUB -e {basepath}{consortium}_{start}_{end}_error.err

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

cd /data/cctm/darpa_perturbation_mouse_study/MDSINE2_data/MDSINE2/
python keystoneness.py --type leave-one-out --model {chain_fname} --data {input_fname} --output-tbl {basepath}{consortium}_{start}_{end}.tsv --compute-base {computebase}
'''

chains = {
    'healthy': 'output_real/runs/healthy1_5_0.0001_rel_2_5/ds0_is0_b5000_ns15000_mo-1_logTrue_pertsmult/graph_leave_out-1/mcmc.pkl',
    'uc': 'output_real/runs/healthy0_5_0.0001_rel_2_5/ds0_is1_b5000_ns15000_mo-1_logTrue_pertsmult/graph_leave_out-1/mcmc.pkl'}

# Clusters
chains_cluster = {
    'healthy': 'output_real/runs/fixed_top/healthy1_5_0.0001_rel_2_5/ds0_is0_b5000_ns15000_mo-1_logTrue_pertsmult/graph_leave_out-1/mcmc.pkl',
    'uc': 'output_real/runs/fixed_top/healthy0_5_0.0001_rel_2_5/ds0_is3_b5000_ns15000_mo-1_logTrue_pertsmult/graph_leave_out-1/mcmc.pkl'}
tmp_folder = 'tmp/keystone_data/'
output_basepath = 'tmp/keystone_clusters/'
os.makedirs(tmp_folder, exist_ok=True)
os.makedirs(output_basepath, exist_ok=True)

fname_fmt = tmp_folder + '{consortium}_clusters.txt'
tsv_fmt = output_basepath + '{consortium}_clusters_{start}_{end}.tsv'
for consortium in chains_cluster:
    # Make the data folder
    chain = pl.inference.BaseMCMC.load(chains_cluster[consortium])
    asvs = chain.graph.data.asvs
    clustering = chain.graph[names.STRNAMES.CLUSTERING_OBJ]
    s = ''
    for cluster in clustering:
        s = s +','.join([asvs.names.order[aaa] for aaa in cluster.members]) + '\n'
    input_fname = fname_fmt.format(consortium=consortium)
    f = open(input_fname, 'w')
    f.write(s)
    f.close()

    # Make tsv fname
    basepath = basepath = input_fname.replace('.txt', '/')
    os.makedirs(basepath, exist_ok=True)
    start = 0
    n_asvs_per_job = 1

    f = open(input_fname, 'r')
    args = f.read().split('\n')\

    compute_base = 1
    while start < len(args):
        end = start+n_asvs_per_job
        if end > len(args):
            end = len(args)

        # Make input data
        input_fname = basepath + 'data_{}_{}.txt'.format(start,end)
        f = open(input_fname, 'w')
        f.write('\n'.join(args[start:end]))
        f.close()

        # Make lsf file
        lsf_fname = basepath + 'job_{}_{}.lsf'.format(start,end)
        f = open(lsf_fname, 'w')
        f.write(my_str.format(
            consortium=consortium, start=start, end=end, queue='short', n_cpus=1, n_mbs=7000,
            chain_fname=chains[consortium], input_fname=input_fname, basepath=basepath,
            RRR='cluster', computebase=compute_base))
        f.close()

        # Submit the job
        command = 'bsub < {}'.format(lsf_fname)
        print(command)
        os.system(command)
        time.sleep(90)
        if compute_base == 1:
            compute_base = 0
        start = end
# sys.exit()

# Chains and cycles
fnames_ddd = {
    'healthy': [
        'tmp/keystone_data/healthy_chain_2.txt',
        'tmp/keystone_data/healthy_chain_3.txt',
        'tmp/keystone_data/healthy_cycle_2.txt',
        'tmp/keystone_data/healthy_cycle_3.txt',
        'tmp/keystone_data/healthy_chain_1.txt'],
    'uc': [
        'tmp/keystone_data/uc_chain_2.txt',
        'tmp/keystone_data/uc_chain_3.txt',
        'tmp/keystone_data/uc_cycle_2.txt',
        'tmp/keystone_data/uc_cycle_3.txt',
        'tmp/keystone_data/uc_chain_1.txt']}
# If agglomerate is True, condense all of the separate tsv files together
agglomerate = False
tbl_format = '{basepath}{consortium}_{start}_{end}.tsv'

if agglomerate:
    df_master = None

for consortium in chains:
    chain_fname = chains[consortium]
    for input_fname in fnames_ddd[consortium]:
        print('\n\nInput fname: {}'.format(input_fname))
        basepath = basepath = input_fname.replace('.txt', '/')
        os.makedirs(basepath, exist_ok=True)
        start = 0
        n_asvs_per_job = 1

        f = open(input_fname, 'r')
        args = f.read().split('\n')
        f.close()
        save_the_table = True
        
        compute_base = 1
        while start < len(args):
            end = start+n_asvs_per_job
            if end > len(args):
                end = len(args)

            if not agglomerate:

                # Make input data
                input_fname = basepath + 'data_{}_{}.txt'.format(start,end)
                f = open(input_fname, 'w')
                f.write('\n'.join(args[start:end]))
                f.close()

                # Make lsf file
                lsf_fname = basepath + 'job_{}_{}.lsf'.format(start,end)
                f = open(lsf_fname, 'w')
                f.write(my_str.format(
                    consortium=consortium, start=start, end=end, queue='short', n_cpus=1, n_mbs=7000,
                    chain_fname=chain_fname, input_fname=input_fname, basepath=basepath,
                    RRR='R', computebase=compute_base))
                f.close()

                # Submit the job
                command = 'bsub < {}'.format(lsf_fname)
                print(command)
                os.system(command)
                time.sleep(90)
                if compute_base == 1:
                    compute_base = 0
            else:
                tbl_fname = tbl_format.format(basepath=basepath, consortium=consortium,
                    start=start, end=end)
                try:
                    df = pd.read_csv(tbl_fname, sep='\t', index_col=0)
                    print('found')
                except Exception as e:
                    print('NO FILE ', tbl_fname)
                    print(e)
                    save_the_table = False
                    start = end
                    continue
                if df_master is None:
                    df_master = df
                else:
                    df = df.drop('base', axis='index')
                    df_master = df_master.append(df)
                    
            start = end
        
        if agglomerate and save_the_table:
            # print(df_master)
            print(df_master.index)
            df_master.to_csv(basepath + 'master_tbl.tsv', sep='\t', index=True, header=True)



# Each ASV
for consortium in ['uc', 'healthy']:
    start = 0
    n_asvs_per_job = 1
    chain_fname = chains[consortium]
    chain = pl.inference.BaseMCMC.load(chain_fname)
    asv_names = chain.graph.data.subjects.asvs.names.order
    basepath = 'tmp/keystone_{}/'.format(consortium)
    os.makedirs(basepath, exist_ok=True)

    compute_base = 1
    while start < len(asv_names):

        end = start+n_asvs_per_job
        if end > len(asv_names):
            end = len(asv_names)

        # Make input data
        input_fname = basepath + 'data_{}_{}.txt'.format(start,end)
        f = open(input_fname, 'w')
        f.write('\n'.join(asv_names[start:end]))
        f.close()

        # Make lsf file
        lsf_fname = basepath + 'job_{}_{}.lsf'.format(start,end)
        f = open(lsf_fname, 'w')
        f.write(my_str.format(
            consortium=consortium, start=start, end=end, queue='short', n_cpus=1, n_mbs=7000,
            chain_fname=chain_fname, input_fname=input_fname, basepath=basepath, RRR='',
            computebase=compute_base))
        f.close()

        # Submit the job
        command = 'bsub < {}'.format(lsf_fname)
        print(command)
        os.system(command)
        time.sleep(90)
        if compute_base == 1:
            compute_base = 0
        start = end

sys.exit()