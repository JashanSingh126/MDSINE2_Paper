'''This file is a temporary file that is used for experimentation. It has no
utility within perturbation_study.
'''

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

import pylab as pl
import synthetic
import diversity
import config
import preprocess_filtering as filtering
import model
import names
import main_base
import metrics
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
pl.seed(1)

HEALTHY_SUBJECTS = ['2','3','4','5']
UC_SUBJECTS = ['6','7','8','9','10']

subjset_real = pl.base.SubjectSet.load('pickles/real_subjectset.pkl')


s1 = 'time_look_ahead/Healthy_subject2/output/MDSINE2-subj2-tla6.0-start1.0-pred.npy'
s2 = 'time_look_ahead/Healthy_subject2/output/MDSINE2-subj2-tla6.0-start1.0-truth.npy'


a = np.arange(4).reshape(2,2)

c = np.arange(24, step=2).reshape(3,2,2)

print(a)
print()
print(c)
print()
print(c-a)
print()
print(np.mean(c-a, axis=1))

sys.exit()

####################################################
# PERMANOVA
####################################################
import skbio.stats.distance
import skbio.diversity

colonization = 5

# a = skbio.stats.distance.permanova()

labels = []
reads = []
grouping = []
i_uc = 0
i_he = 0
for subj in subjset_real:
    print(subj.name)
    for t in subj.times:
        if t < colonization:
            continue
        if t >= subjset_real.perturbations[0].start:
            continue
        reads.append(subj.reads[t])
        if subj.name in HEALTHY_SUBJECTS:
            label = 'Healthy{}'.format(i_he)
            i_he += 1
        else:
            label = 'UC{}'.format(i_uc)
            i_uc += 1
        labels.append(label)
        grouping.append('UC' if subj.name not in HEALTHY_SUBJECTS else 'Healthy')

print(labels)

bc_dm = skbio.diversity.beta_diversity(counts=np.asarray(reads), ids=labels, metric="braycurtis")
# print(bc_dm)

result = skbio.stats.distance.permanova(
    distance_matrix=bc_dm, grouping=grouping)
print(result)

sys.exit()

####################################################
# Wilcoxon signed-rank tests
####################################################
colonization = 5

healthy = {}
uc = {}

for subj in subjset_real:
    print(len(subj.times))
    for t in subj.times:
        aaa = diversity.alpha.normalized_entropy(subj.reads[t])
        if subj.name in HEALTHY_SUBJECTS:
            if t not in healthy:
                healthy[t] = []
            healthy[t].append(aaa)
        else:
            if t not in uc:
                uc[t] = []
            uc[t].append(aaa)

times_healthy = list(healthy.keys())
median_healthy = []
for t in times_healthy:
    if t < colonization:
        continue
    if t >= subjset_real.perturbations[0].start:
        continue
    median_healthy.append(np.median(healthy[t]))

times_uc = list(uc.keys())
median_uc = []
for t in times_uc:
    if t < colonization:
        continue
    if t >= subjset_real.perturbations[0].start:
        continue
    median_uc.append(np.median(uc[t]))

print(len(median_uc))
print(len(median_healthy))
print(scipy.stats.wilcoxon(median_healthy, median_uc))


sys.exit()

# ####################################################
# # Check bjobs
# ####################################################
# fname = 'tmp/bjobs_output.txt'
# fname_tabed = 'tmp/bjobs_output_tabed.txt'
# f = open(fname, 'r')
# txt = f.read()
# f.close()

# delete_date = r'(\s+Oct.+)'
# delete_date = re.compile(delete_date)
# submit_time_delete = r'(\s+SUBMIT_TIME)'
# submit_time_delete = re.compile(submit_time_delete)
# make_tabs = r'(\ ){1,10}'
# make_tabs = re.compile(make_tabs)

# delete_date = r'(\s+Oct.+)'
# delete_date = re.compile(delete_date)
# submit_time_delete = r'(\s+SUBMIT_TIME)'
# submit_time_delete = re.compile(submit_time_delete)
# make_tabs = r'(\ ){1,10}'
# make_tabs = re.compile(make_tabs)

# a = delete_date.sub('', txt)
# a = submit_time_delete.sub('', a)
# a = make_tabs.sub(',', a)
# f = open(fname_tabed, 'w')
# f.write(a)
# f.close()

# df = pd.read_csv(fname_tabed, sep=',')
# print(df)
# print(df[df['STAT']=='PEND']['JOB_NAME'])

# # Check if any job names are missing, if they are then send them to a new queue

# sys.exit()

# ####################################################
# # Adjust metric on metrics
# ####################################################
# path = 'output_real/pred/healthy0_5_0.0001_rel_2_5/ds0_is0_b5000_' \
#     'ns15000_mo-1_logTrue_pertsmult/graph_leave_out0/validation/results.pkl'

# mmm = metrics.Metrics.load(path)
# results = mmm.results
# sid = list(results.keys())[0]
# simtype = list(results[sid].keys())[0]
# d = results[sid][simtype]

# pred_traj = d['pred-traj']
# subj_traj = d['subj-traj']

# print(pred_traj.shape)
# print(subj_traj.shape)
# print(d['error-metric'])
# print(d['subj-times'])
# print()
# print(d['pred-times'])
# print(d['error-total'].shape)

# # print(sid, simtype)

# sys.exit()

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
    n_asvs_per_job = 2

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
        n_asvs_per_job = 5

        f = open(input_fname, 'r')
        args = f.read().split('\n')
        f.close()
        save_the_table = True
        
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
                    consortium=consortium, start=start, end=end, queue='normal', n_cpus=1, n_mbs=7000,
                    chain_fname=chain_fname, input_fname=input_fname, basepath=basepath,
                    RRR='R'))
                f.close()

                # Submit the job
                command = 'bsub < {}'.format(lsf_fname)
                print(command)
                os.system(command)
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



# # Each ASV
# start = 0
# n_asvs_per_job = 5
# consortium = 'uc'
# chain_fname = chains[consortium]
# chain = pl.inference.BaseMCMC.load(chain_fname)
# asv_names = chain.graph.data.subjects.asvs.names.order
# basepath = 'tmp/keystone_{}/'.format(consortium)
# os.makedirs(basepath, exist_ok=True)
# while start < len(asv_names):

#     end = start+n_asvs_per_job
#     if end > len(asv_names):
#         end = len(asv_names)

#     # Make input data
#     input_fname = basepath + 'data_{}_{}.txt'.format(start,end)
#     f = open(input_fname, 'w')
#     f.write('\n'.join(asv_names[start:end]))
#     f.close()

#     # Make lsf file
#     lsf_fname = basepath + 'job_{}_{}.lsf'.format(start,end)
#     f = open(lsf_fname, 'w')
#     f.write(my_str.format(
#         consortium=consortium, start=start, end=end, queue='medium', n_cpus=1, n_mbs=7000,
#         chain_fname=chain_fname, input_fname=input_fname, basepath=basepath, RRR=''))
#     f.close()

#     # Submit the job
#     command = 'bsub < {}'.format(lsf_fname)
#     print(command)
#     os.system(command)
#     start = end

sys.exit()


# ####################################################
# # Correct mdsine1 data
# ####################################################

# asvs = subjset_real.asvs
# for asv in asvs:
#     print(asv)

# asv = asvs['Clostridium-hiranonis']
# asv.taxonomy['kingdom'] = 'Bacteria'
# asv.taxonomy['phylum'] = 'Firmicutes'
# asv.taxonomy['class'] = 'Clostridia'
# asv.taxonomy['order'] = 'Closdtridiales'
# asv.taxonomy['family'] = 'Clostridiaceae'
# asv.taxonomy['genus'] = 'Clostridium'
# asv.taxonomy['species'] = 'hiranonis'

# asv = asvs['Clostridium-difficile']
# asv.taxonomy['kingdom'] = 'Bacteria'
# asv.taxonomy['phylum'] = 'Firmicutes'
# asv.taxonomy['class'] = 'Clostridia'
# asv.taxonomy['order'] = 'Closdtridiales'
# asv.taxonomy['family'] = 'Clostridiaceae'
# asv.taxonomy['genus'] = 'Clostridium'
# asv.taxonomy['species'] = 'difficile'

# asv = asvs['Proteus-mirabilis']
# asv.taxonomy['kingdom'] = 'Bacteria'
# asv.taxonomy['phylum'] = 'Proteobacteria'
# asv.taxonomy['class'] = 'Gammaproteobacteria'
# asv.taxonomy['order'] = 'Enterobacterales'
# asv.taxonomy['family'] = 'Morganellaceae'
# asv.taxonomy['genus'] = 'Proteus'
# asv.taxonomy['species'] = 'mirabilis'

# asv = asvs['Clostridium-scindens']
# asv.taxonomy['kingdom'] = 'Bacteria'
# asv.taxonomy['phylum'] = 'Firmicutes'
# asv.taxonomy['class'] = 'Clostridia'
# asv.taxonomy['order'] = 'Closdtridiales'
# asv.taxonomy['family'] = 'Clostridiaceae'
# asv.taxonomy['genus'] = 'Clostridium'
# asv.taxonomy['species'] = 'scindens'

# asv = asvs['Ruminococcus-obeum']
# asv.taxonomy['kingdom'] = 'Bacteria'
# asv.taxonomy['phylum'] = 'Firmicutes'
# asv.taxonomy['class'] = 'Clostridia'
# asv.taxonomy['order'] = 'Closdtridiales'
# asv.taxonomy['family'] = 'Ruminococcaceae'
# asv.taxonomy['genus'] = 'Ruminococcus'
# asv.taxonomy['species'] = 'obeum'

# asv = asvs['Clostridium-ramosum']
# asv.taxonomy['kingdom'] = 'Bacteria'
# asv.taxonomy['phylum'] = 'Firmicutes'
# asv.taxonomy['class'] = 'Clostridia'
# asv.taxonomy['order'] = 'Closdtridiales'
# asv.taxonomy['family'] = 'Clostridiaceae'
# asv.taxonomy['genus'] = 'Clostridium'
# asv.taxonomy['species'] = 'ramosum'

# asv = asvs['Bacteroides-ovatus']
# asv.taxonomy['kingdom'] = 'Bacteria'
# asv.taxonomy['phylum'] = 'Bacteroidetes'
# asv.taxonomy['class'] = 'Bacteroidia'
# asv.taxonomy['order'] = 'Bacteroidales'
# asv.taxonomy['family'] = 'Bacteroidaceae'
# asv.taxonomy['genus'] = 'Bacteroides'
# asv.taxonomy['species'] = 'ovatus'

# asv = asvs['Akkermansia-muciniphila']
# asv.taxonomy['kingdom'] = 'Bacteria'
# asv.taxonomy['phylum'] = 'Verrucomicrobia'
# asv.taxonomy['class'] = 'Verrucomicrobiae'
# asv.taxonomy['order'] = 'Verrucomicrobiales'
# asv.taxonomy['family'] = 'Akkermansiaceae'
# asv.taxonomy['genus'] = 'Akkermansia'
# asv.taxonomy['species'] = 'muciniphila'

# asv = asvs['Parabacteroides-distasonis']
# asv.taxonomy['kingdom'] = 'Bacteria'
# asv.taxonomy['phylum'] = 'Bacteroidetes'
# asv.taxonomy['class'] = 'Bacteroidia'
# asv.taxonomy['order'] = 'Bacteroidales'
# asv.taxonomy['family'] = 'Porphyromonadaceae'
# asv.taxonomy['genus'] = 'Parabacteroides'
# asv.taxonomy['species'] = 'distasonis'

# asv = asvs['Bacteroides-fragilis']
# asv.taxonomy['kingdom'] = 'Bacteria'
# asv.taxonomy['phylum'] = 'Bacteroidetes'
# asv.taxonomy['class'] = 'Bacteroidia'
# asv.taxonomy['order'] = 'Bacteroidales'
# asv.taxonomy['family'] = 'Bacteroidaceae'
# asv.taxonomy['genus'] = 'Bacteroides'
# asv.taxonomy['species'] = 'fragilis'

# asv = asvs['Bacteroides-vulgatus']
# asv.taxonomy['kingdom'] = 'Bacteria'
# asv.taxonomy['phylum'] = 'Bacteroidetes'
# asv.taxonomy['class'] = 'Bacteroidia'
# asv.taxonomy['order'] = 'Bacteroidales'
# asv.taxonomy['family'] = 'Bacteroidaceae'
# asv.taxonomy['genus'] = 'Bacteroides'
# asv.taxonomy['species'] = 'vulgatus'

# asv = asvs['Klebsiella-oxytoca']
# asv.taxonomy['kingdom'] = 'Bacteria'
# asv.taxonomy['phylum'] = 'Proteobacteria'
# asv.taxonomy['class'] = 'Gammaproteobacteria'
# asv.taxonomy['order'] = 'Enterobacterales'
# asv.taxonomy['family'] = 'Enterobacteriaceae'
# asv.taxonomy['genus'] = 'Klebsiella'
# asv.taxonomy['species'] = 'oxytoca'

# asv = asvs['Roseburia-hominis']
# asv.taxonomy['kingdom'] = 'Bacteria'
# asv.taxonomy['phylum'] = 'Firmicutes'
# asv.taxonomy['class'] = 'Clostridia'
# asv.taxonomy['order'] = 'Closdtridiales'
# asv.taxonomy['family'] = 'Lachnospiraceae'
# asv.taxonomy['genus'] = 'Roseburia'
# asv.taxonomy['species'] = 'hominis'

# asv = asvs['Escherichia-coli']
# asv.taxonomy['kingdom'] = 'Bacteria'
# asv.taxonomy['phylum'] = 'Proteobacteria'
# asv.taxonomy['class'] = 'Gammaproteobacteria'
# asv.taxonomy['order'] = 'Enterobacterales'
# asv.taxonomy['family'] = 'Enterobacteriaceae'
# asv.taxonomy['genus'] = 'Escherichia'
# asv.taxonomy['species'] = 'coli'

# subjset_real.save('pickles/subjset_cdiff.pkl')


# ####################################################
# # Stem plots
# ####################################################
# import matplotlib.image as mpimg
# from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox

# subjset_real = pl.base.SubjectSet.load('pickles/real_subjectset.pkl')
# times = []
# for subj in subjset_real:
#     times = np.append(times, subj.times)
# times = np.sort(np.unique(times))
# print(times)

# def remove_edges(ax):
#     ax.spines['top'].set_visible(False)
#     ax.spines['bottom'].set_visible(False)
#     ax.spines['left'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.xaxis.set_major_locator(plt.NullLocator())
#     ax.xaxis.set_minor_locator(plt.NullLocator())
#     ax.yaxis.set_major_locator(plt.NullLocator())
#     ax.yaxis.set_minor_locator(plt.NullLocator())
#     return ax

# y = [-0.1 for t in times]

# fig = plt.figure(figsize=(34,10))
# ax = fig.add_subplot(311)
# ax = remove_edges(ax)
# markerline, stemlines, baseline = ax.stem(times, y, linefmt='none')
# baseline.set_color('black')
# markerline.set_color('black')
# markerline.set_markersize(5)


# x = np.arange(0, np.max(times), step=10)
# labels = ['Day {}'.format(int(t)) for t in x]
# for ylim in [0.12, -0.12]:
#     y = [ylim for t in x]
#     markerline, stemlines, baseline = ax.stem(x, y)
#     stemlines = [stemline.set_color('black') for stemline in stemlines]
#     baseline.set_color('black')
#     markerline.set_color('none')
# for i in range(len(labels)):
#     label = labels[i]
#     xpos = x[i] 
#     ax.text(xpos, 0.-.15, label, horizontalalignment='center')
# x = np.arange(0,np.max(times),2)
# for ylim in [0.07, -0.07]:
#     y = [ylim for t in x]
#     markerline, stemlines, baseline = ax.stem(x, y)
#     stemlines = [stemline.set_color('black') for stemline in stemlines]
#     baseline.set_color('black')
#     markerline.set_color('none')

# for perturbation in subjset_real.perturbations:
#     name = perturbation.name
#     # lines += [start,end]
#     # ax.text(start, 0.33, 'Start {}'.format(name), horizontalalignment='center')
#     ax.text((perturbation.end+perturbation.start)/2, 0.15, name.capitalize(), horizontalalignment='center')

#     # ax.arrow(start, 0.3, 0, -0.27, length_includes_head=True, head_width=0.25, head_length=0.05)
#     # ax.arrow(end, 0.3, 0, -0.27, length_includes_head=True, head_width=0.25, head_length=0.05)
#     starts = np.asarray([perturbation.start])
#     ends = np.asarray([perturbation.end])
#     ax.barh(y=[0 for i in range(len(starts))], width=ends-starts, height=0.1, left=starts)

# ax.text(0, 0.2, 'Colonization', horizontalalignment='center')
# ax.arrow(0, 0.17, 0, -0.14, length_includes_head=True, head_width=0.25, head_length=0.05)


# # stool collection
# xpos = np.max(times)* 1.05
# y = 0.05
# ax.scatter([xpos], [y], c='black', s=25)
# ax.text(xpos+1, y, 'Stool Collection', horizontalalignment='left', fontsize=18, 
#     verticalalignment='center')
# ax.set_ylim(-0.15,0.4)
# plt.show()
# sys.exit()

# ####################################################
# # Parse data like MDSINE1
# ####################################################

# chains = {
#     'healthy': 'output_real/pylab24/real_runs/strong_priors/healthy1_5_0.0001_rel_2_5/ds0_is0_b5000_ns15000_mo-1_logTrue_pertsmult/graph_leave_out-1/mcmc.pkl',
#     'uc': 'output_real/pylab24/real_runs/strong_priors/healthy0_5_0.0001_rel_2_5/ds0_is1_b5000_ns15000_mo-1_logTrue_pertsmult/graph_leave_out-1/mcmc.pkl'}

# for dtype in ['healthy', 'uc']:
#     chainname = chains[dtype]
#     chain = pl.inference.BaseMCMC.load(chainname)
#     subjset = chain.graph.data.subjects
#     MDSINE2_util.make_data_like_mdsine1(subjset, basepath='tmp/{}_mdsine1_like/'.format(dtype))

# ####################################################
# # Calculate keystoneness
# ####################################################
# # from multiprocessing import freeze_support

# chains = {
#     'healthy': 'output_real/pylab24/real_runs/strong_priors/healthy1_5_0.0001_rel_2_5/ds0_is0_b5000_ns15000_mo-1_logTrue_pertsmult/graph_leave_out-1/mcmc.pkl',
#     'uc': 'output_real/pylab24/real_runs/strong_priors/healthy0_5_0.0001_rel_2_5/ds0_is1_b5000_ns15000_mo-1_logTrue_pertsmult/graph_leave_out-1/mcmc.pkl'}

# fnames = {
#     'healthy': [
#         'tmp/keystone_proposal/healthy_chain_2.txt',
#         'tmp/keystone_proposal/healthy_chain_3.txt',
#         'tmp/keystone_proposal/healthy_cycle_2.txt',
#         'tmp/keystone_proposal/healthy_cycle_3.txt'],
#     'uc': [
#         # 'tmp/keystone_proposal/uc_chain_2.txt',
#         # 'tmp/keystone_proposal/uc_chain_3.txt']} #,
#         # 'tmp/keystone_proposal/uc_cycle_2.txt',
#         'tmp/keystone_proposal/uc_cycle_3.txt']}

# for key in ['uc']: #chains.keys():
#     fnames_temp = fnames[key]
#     for fname in fnames_temp:
#         MDSINE2_util.keystoneness(chains[key], fname=fname, outfile=fname.replace('.txt', '_keystoneness_full_posterior.txt'),
#             max_posterior=None, mp=None)
# sys.exit()

# ####################################################
# # Get interaction traces
# ####################################################
# # chain_loc = 'output_real/pylab24/real_runs/strong_priors/healthy1_5_0.0001_rel_2_5/ds0_is0_b5000_ns15000_mo-1_logTrue_pertsmult/graph_leave_out-1/mcmc.pkl'
# chain_loc = 'output_real/pylab24/real_runs/strong_priors/healthy0_5_0.0001_rel_2_5/ds0_is1_b5000_ns15000_mo-1_logTrue_pertsmult/graph_leave_out-1/mcmc.pkl'

# chain = pl.inference.BaseMCMC.load(chain_loc)
# interactions = chain.graph[names.STRNAMES.INTERACTIONS_OBJ]

# arr = interactions.get_trace_from_disk()
# np.save('asv_asv_interactions.npy', arr)


# sys.exit()

# ####################################################
# # Make family level plots of the ASVs in the phylogenetic trees
# ####################################################
# chain_locs = [
#     'output_real/pylab24/real_runs/strong_priors/healthy1_5_0.0001_rel_2_5/ds0_is0_b5000_ns15000_mo-1_logTrue_pertsmult/graph_leave_out-1/mcmc.pkl',
#     'output_real/pylab24/real_runs/strong_priors/healthy0_5_0.0001_rel_2_5/ds0_is1_b5000_ns15000_mo-1_logTrue_pertsmult/graph_leave_out-1/mcmc.pkl']

# # tree_loc = 'raw_data/phylogenetic_tree_w_reference_seq.nhx'
# os.makedirs('tmp', exist_ok=True)
# os.makedirs('tmp/subtrees_125percentmedian', exist_ok=True)
# asvnames = set([])
# for chainloc in chain_locs:
#     chain = pl.inference.BaseMCMC.load(chainloc)
#     asvs = chain.graph.data.asvs

#     for asv in asvs:
#         asvnames.add(asv.name)

# asvnames = list(asvnames)
# set_asvnames = set(asvnames)

# totalnames = copy.deepcopy(asvnames)
# # tree = ete3.Tree(tree_loc)
# # for name in tree.get_leaf_names():
# #     if 'OTU_' not in name:
# #         totalnames.append(name)

# # # print(totalnames)

# # # tree.prune(totalnames, preserve_branch_length=True)
# # # tree.write(outfile='tmp/tree_temp.nhx')
# # # # sys.exit()
# tree_name = 'tmp/tree_temp.nhx'
# tree = ete3.Tree(tree_name)

# d = {}
# for i, aname in enumerate(asvnames):
#     print('{}/{}'.format(i,len(asvnames)))
#     asv = subjset_real.asvs[aname]
#     if asv.tax_is_defined('family'):
#         family = asv.taxonomy['family']
#         if family in d:
#             continue
#         else:
#             d[family] = []
#             for iii, bname in enumerate(asvnames):
#                 if bname == aname:
#                     continue
#                 basv = subjset_real.asvs[bname]
#                 if basv.taxonomy['family'] != family:
#                     continue
#                 d[family].append(tree.get_distance(aname, bname))

# f = open('tmp/subtrees_125percentmedian/family_dists_asv.txt', 'w')
# for family in d:
#     f.write(family + '\n')
#     arr = np.asarray(d[family])
#     summ = pl.variables.summary(arr)
#     for k,v in summ.items():
#         f.write('\t{}: {}\n'.format(k,v))
# f.write('total\n')
# arr = []
# for ele in d.values():
#     arr += ele
# arr = np.asarray(arr)
# summ = pl.variables.summary(arr)
# for k,v in summ.items():
#     f.write('\t{}:{}\n'.format(k,v))


# # Set the radius to 125% the median
# radius = summ['median'] * 1.25
# f.write('Radius set to 125% of median ({}): {}'.format(summ['median'], radius))
# f.close()

# # Make the distance matrix
# print(len(tree))
# names = tree.get_leaf_names()
# names.sort()

# with open('tmp/phylo_dist.pkl', 'rb') as handle:
#     df = pickle.load(handle)


# i = 0
# f = open('tmp/subtrees_125percentmedian/table.tsv', 'w')
# for asvname in asvnames:
#     asv = subjset_real.asvs[asvname]
#     if not asv.tax_is_defined('species'):
#         i += 1
#         print('\n\nLooking at {}, {}'.format(i,asv))
#         print('-------------------------')

#         row = df[asv.name].to_numpy()
#         idxs = np.argsort(row)

#         iii = 0
#         for idx in idxs:
#             if names[idx] not in set_asvnames:
#                 iii = idx 
#                 break

#         cnr = names[iii]
#         f.write('{}\n'.format(asv.name))

#         tree = ete3.Tree(tree_name)
#         # Get the all elements within `radius`
#         names_to_keep = []
#         row = df[asv.name].to_numpy()
#         idxs = np.argsort(row)

#         for idx in idxs:
#             if row[idx] > radius:
#                 break
#             if names[idx] in set_asvnames:
#                 continue
#             names_to_keep.append(names[idx])

#         print(names_to_keep)

#         # Make subtree of just these names
#         names_to_keep.append(asv.name)
#         tree.prune(names_to_keep, preserve_branch_length=False)

#         for node in tree.traverse():
#             node.name = node.name.replace('OTU','ASV')

#             if 'ASV' in node.name:
#                 face = ete3.TextFace(node.name)
#                 face.background.color='LightGreen'
#                 node.add_face(face, column=0)

#         ts = ete3.TreeStyle()
#         ts.title.add_face(ete3.TextFace('{}, Phylogenetic radius: {:.4f}'.format(
#             asv.name.replace('OTU', 'ASV'), radius), fsize=15), column=1)
#         tree.render('tmp/subtrees_125percentmedian/{}.pdf'.format(asv.name.replace('OTU','ASV')), tree_style=ts)
# f.close()
# sys.exit()

# ####################################################
# # Make plots of the ASVs
# ####################################################
# chain_locs = [
#     'output_real/pylab24/real_runs/strong_priors/healthy1_5_0.0001_rel_2_5/ds0_is0_b5000_ns15000_mo-1_logTrue_pertsmult/graph_leave_out-1/mcmc.pkl',
#     'output_real/pylab24/real_runs/strong_priors/healthy0_5_0.0001_rel_2_5/ds0_is1_b5000_ns15000_mo-1_logTrue_pertsmult/graph_leave_out-1/mcmc.pkl']
# folder_base = ['./healthy/', './unhealthy/']

# for iii, chain_loc in enumerate(chain_locs):

#     basepath = folder_base[iii]
#     os.makedirs(basepath, exist_ok=True)
    

#     chain = pl.inference.BaseMCMC.load(chain_loc)
#     subjset = chain.graph.data.subjects
#     asvs = subjset.asvs

#     subj_matrices = [subj.matrix()['abs'] for subj in subjset]
#     subj_times = [subj.times for subj in subjset]
#     subjnames = [subj.name for subj in subjset]

#     titlefmt = '%(name)s\n%(family)s, %(genus)s, %(species)s'

#     for asv in asvs:
#         aidx = asv.idx

#         print('{}/{}'.format(aidx,len(asvs)))

#         fig = plt.figure()
#         ax = fig.add_subplot(111)
#         for i, M in enumerate(subj_matrices):
#             times = subj_times[i]
            
#             ax.plot(times, M[aidx,:], marker='o', label=subjnames[i])

#         pl.visualization.shade_in_perturbations(ax, perturbations=subjset.perturbations)
        
#         ax.set_yscale('log')
#         title = pl.asvname_formatter(titlefmt, asv=asv, asvs=asvs, lca=False).replace('OTU', 'ASV')
#         ax.set_title(title, fontsize=15, fontweight='bold')
#         ax.set_xlabel('Days', fontsize=12)
#         ax.set_ylabel('CFUs/g', fontsize=12)
#         fig.tight_layout()

#         name = asv.name.replace('OTU', 'ASV')

#         plt.savefig(basepath + name + '.pdf')
#         plt.close()
    

# ####################################################
# # 16S v4 gapless sequences and taxonomy
# ###################################################

# seqs = SeqIO.parse('raw_data/align_seqs_v4.sto', 'stockholm')
# seqs = SeqIO.to_dict(seqs)

# asvs = {}
# chains = [
#     'output_real/pylab24/real_runs/strong_priors/healthy1_5_0.0001_rel_2_5/ds0_is0_b5000_ns15000_mo-1_logTrue_pertsmult/graph_leave_out-1/mcmc.pkl',
#     'output_real/pylab24/real_runs/strong_priors/healthy0_5_0.0001_rel_2_5/ds0_is1_b5000_ns15000_mo-1_logTrue_pertsmult/graph_leave_out-1/mcmc.pkl']
# for i, fname in enumerate(chains):
#     chain = pl.inference.BaseMCMC.load(fname)
#         if asvname not in asvs:
#             asvs[asvname] = None

# print(len(asvs))
# d = {}
# l = None
# for k,v in seqs.items():
#     if k in asvs:
#         d[k] = v
#     l = len(v.seq)
# seqs = d

# asvs = list(asvs.keys())

# M = np.zeros(shape=(len(seqs), l), dtype=bool)
# Mseqs = np.zeros(shape=(len(seqs), l), dtype=str)
# for i, (k,v) in enumerate(seqs.items()):
#     Mseqs[i] = np.asarray(list(str(v.seq)))
#     M[i] = Mseqs[i] == '-'
#     # print(str)

# cols_to_keep = []
# for col in range(M.shape[1]):
#     n = np.sum(M[:, col])
#     if n == 0:
#         cols_to_keep.append(col)
#     else:
#         print('column {} has {} gaps'.format(col, n))

# print('{}/{} left'.format(len(cols_to_keep), M.shape[1]))

# Mseqs = Mseqs[:, cols_to_keep]
# # print(Mseqs)
# # print(Mseqs.shape)

# lll = []
# taxonomies = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
# d = {}
# for tax in taxonomies:
#     d[tax] = []
# asvset = subjset_real.asvs
# for i in range(Mseqs.shape[0]):
#     # print()
#     # print(asvs[i])
#     asvname = asvs[i]
#     asv = asvset[asvname]
#     for taxa in taxonomies:
#         d[taxa].append('NA')
#         if asv.tax_is_defined(taxa):
#             if taxa == 'species':
#                 p = asv.taxonomy[taxa].split('/')
#                 if len(p) > 3:
#                     continue
#                 else:
#                     d[taxa][-1] = asv.taxonomy[taxa]
#             else:
#                 d[taxa][-1] = asv.taxonomy[taxa]
#     lll.append(''.join(list(Mseqs[i])))

# ddd = [d[taxa] for taxa in taxonomies]
# df = pd.DataFrame([asvs,lll] + ddd).T
# df.columns=['name', 'sequences'] + taxonomies
# df = df.set_index('name')
# print(df)
# print(df.shape)
# df.to_csv('16S_v4_aligned_gapless_sequences_and_taxonomy.tsv', sep='\t')
# # print(df)
# # print(df.T)

# ####################################################
# # Coclustering trace save
# ###################################################
# chains = 'output_real/pylab24/real_runs/strong_priors/healthy1_5_0.0001_rel_2_5/ds0_is0_b5000_ns15000_mo-1_logTrue_pertsmult/graph_leave_out-1/mcmc.pkl'
# mcmc = pl.inference.BaseMCMC.load(chains)
# clustering = mcmc.graph[names.STRNAMES.CLUSTERING_OBJ]
# coclusters = clustering.coclusters.get_trace_from_disk(section='posterior')
# np.save('coclusters_trace.npy', coclusters)

# print(coclusters.shape)
# sys.exit()

# ####################################################
# # Percent Identity
# ###################################################

# chains = [
#     'output_real/pylab24/real_runs/strong_priors/healthy1_5_0.0001_rel_2_5/ds0_is0_b5000_ns15000_mo-1_logTrue_pertsmult/graph_leave_out-1/mcmc.pkl',
#     'output_real/pylab24/real_runs/strong_priors/healthy0_5_0.0001_rel_2_5/ds0_is1_b5000_ns15000_mo-1_logTrue_pertsmult/graph_leave_out-1/mcmc.pkl'
# ]

# def combine_asvs_union(chains):
#     asvs_total = set([])
#     for chain in chains:
        
#         chain = pl.inference.BaseMCMC.load(chain)
#         subjset = chain.graph.data.subjects

#         asvs = []
#         for asv in subjset.asvs:
#             asvs.append(asv.name)
#         asvs = set(asvs)
#         asvs_total = asvs_total.union(asvs)
#     return asvs_total

# asvs = combine_asvs_union(chains)

# seqs = SeqIO.parse('raw_data/seqs_temp/RDP_trim1600_mingaplen1_noccur5_aligned_align_seqs.sto', 'stockholm')
# seqs = SeqIO.to_dict(seqs)
# temp = {}

# for k,v in seqs.items():
#     if k in asvs:
#         temp[k] = str(v.seq)
# seqs = temp
# print(seqs.keys())

# l = len(seqs['OTU_0'])
# M = np.zeros(shape=(len(seqs), l), dtype=bool)
# seqs_arr = np.zeros(shape=(len(seqs), l), dtype=str)
# for i, (k,v) in enumerate(seqs.items()):
#     seqs_arr[i] = np.array(list(v)) 
#     M[i] = seqs_arr[i] != '-'

# sums = np.sum(M, axis=0)
# print(sums)
# cols_keep = sums > 0
# M = M[:, cols_keep]
# seqs_arr = seqs_arr[:, cols_keep]
# print(M.shape)

# def percent_identity(seq1, seq2):
#     return np.sum(seq1 == seq2)/len(seq1)

# def print_clustering(labels, asvs):
#     d = {}
#     for i,label in enumerate(labels):
#         a = pl.asvname_formatter(asv=i, asvs=subjset_real.asvs, 
#             format='%(name)s %(order)s %(family)s %(genus)s %(species)s')
#         a = a.replace('OTU', 'ASV')
#         if label not in d:
#             d[label] = [a]
#         else:
#             d[label].append(a)
#     for i,v in enumerate(d.values()):
#         print(i)
#         for ele in v:
#             print('\t{}'.format(ele))

# print('seq1\n{}'.format(seqs_arr[0]))
# print('seq2\n{}'.format(seqs_arr[1]))
# print(percent_identity(seqs_arr[0], seqs_arr[1]))

# PI = np.diag(np.ones(len(seqs), dtype=float))
# for i in range(PI.shape[0]):
#     for j in range(i):
#         PI[i,j] = percent_identity(seqs_arr[i], seqs_arr[j])
#         PI[j,i] = PI[i,j]

# n_clusters = []
# vals = []
# step = 0.0001
# distances = np.arange(step, 1, step)

# for i in distances:
#     c = sklearn.cluster.AgglomerativeClustering(n_clusters=None, distance_threshold=i, affinity='precomputed',
#         linkage='average').fit(PI)
#     n_clusters.append(c.n_clusters_)
#     vals.append(i)
#     if n_clusters[-1] == 1:
#         break
# # print(c)
# # print(c.labels_)
# # keys = list(seqs.keys())
# print_clustering(sklearn.cluster.AgglomerativeClustering(n_clusters=None, distance_threshold=0.1, affinity='precomputed',
#         linkage='average').fit(1-PI).labels_, subjset_real.asvs)

# plt.plot(vals, n_clusters)
# plt.xlabel('Distance Threshold')
# plt.ylabel('Number of Clusters')
# plt.title('Agglomerative Clustering')
# plt.show()


# percentiles = [0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92]
# clus99th = int(.99*PI.shape[0])


# print(np.nanmin(PI))
# print(np.nanmax(PI))
# # print(PI)
# asvs = chain.graph.data.subjects.asvs
# keys = np.random.permutation(np.array(list(seqs.keys())))[:10]
# for i,seqi in enumerate(keys):
#     for j,seqj in enumerate(keys):
#         if i == j:
#             continue
#         print()
#         print(pl.asvname_formatter(asv=i, asvs=asvs, 
#             format='%(order)s %(family)s %(genus)s %(species)s'))
        # print(pl.asvname_formatter(asv=j, asvs=asvs, 
        #     format='%(order)s %(family)s %(genus)s %(species)s'))
#         print(PI[i,j])
# sys.exit()

# ####################################################
# # Make df of the cluster interactions
# ###################################################
# fnames = [
# #     # 'output_real/pylab24/real_runs/strong_priors/healthy1_5_0.0001_rel_2_5/ds0_is0_b5000_ns15000_mo-1_logTrue_pertsmult/graph_leave_out-1/mcmc.pkl',
#     'output_real/pylab24/real_runs/strong_priors/fixed_top/healthy0_5_0.0001_rel_2_5/ds0_is3_b5000_ns15000_mo-1_logTrue_pertsmult/graph_leave_out-1/mcmc.pkl'
# ]
# names_loop = [
#     # 'healthy',
#     'ulcerative_colitis'
# ]

# for i, fname in enumerate(fnames):
#     print('here')
#     chain = pl.inference.BaseMCMC.load(fname)
    
#     interactions = chain.graph[names.STRNAMES.INTERACTIONS_OBJ]
#     clustering = chain.graph[names.STRNAMES.CLUSTERING_OBJ]
#     asvs = clustering.items

#     asv_interaction_trace = interactions.get_trace_from_disk(section='posterior')
#     clus_interactions = main_base._condense_interactions(asv_interaction_trace, clustering=clustering)
#     expected_interactions = pl.variables.summary(clus_interactions, only='mean', set_nan_to_0=True)['mean']

#     bf_asvs = interactions.generate_bayes_factors_posthoc(
#         prior=chain.graph[names.STRNAMES.CLUSTER_INTERACTION_INDICATOR].prior,
#         section='posterior')
#     bf_clus = main_base._condense_interactions(bf_asvs, clustering=clustering)

#     np.save('interactions_over_gibbs_{}.npy'.format(names_loop[i]), clus_interactions)
#     np.save('bayes_factors_{}.npy'.format(names_loop[i]), bf_clus)
#     np.save('expected_interactions_{}.npy'.format(names_loop[i]), expected_interactions)


#     # mask_clus = bf_clus < 5
#     # clus_interactions[mask_clus] = 0
    
#     # cluster_names = ['Cluster {}'.format(cidx + 1) for cidx in range(clus_interactions.shape[-1])]
#     # df = pd.DataFrame(clus_interactions, columns=cluster_names, index=cluster_names)
#     # df.to_csv('raw_data/diffuse_uc_cluster_interactions_bf5{}.tsv'.format(i), sep='\t' )

# sys.exit()



#     n = 0
#     for asv in asvs:
#         if not pl.isstr(asv.taxonomy['genus']):
#             continue
#         # if 'Clostridium' in asv.taxonomy['genus']:
#         #     print(asv.name, asv.taxonomy)
#         #     n += 1
#         if 'Butyricicoccus' in asv.taxonomy['genus']:
#             print(asv.name, asv.taxonomy)
#             n += 1

#     print('n clostridiales', n)


# # print(subjset_real.asvs['OTU_220'].taxonomy)
# for asv in subjset_real.asvs:
#     if asv.taxonomy['genus'] == 'Butyricicoccus':
#         print(asv.name)

# sys.exit()

# d = {}

# subjset = pl.base.SubjectSet.load('pickles/real_subjectset.pkl')
# classes = {}

# for asv in subjset.asvs:
#     k = asv.taxonomy['kingdom']
#     p = asv.taxonomy['phylum']
#     c = asv.taxonomy['class']
#     o = asv.taxonomy['order']

#     if type(k) == float:
#         k = 'na'
#     if type(p) == float:
#         p = 'na'
#     if type(c) == float:
#         c = 'na'
#     if type(o) == float:
#         o = 'na'

#     if k not in classes:
#         classes[k] = {}
#     if p not in classes[k]:
#         classes[k][p] = {}
#     if c not in classes[k][p]:
#         classes[k][p][c] = {}
#     if o not in classes[k][p][c]:
#         classes[k][p][c][o] = 0
#     classes[k][p][c][o] += 1

# for kingdom,d_ in classes.items():
#     print(kingdom)
#     for phylum, d__ in d_.items():
#         print('\t', phylum)
#         for class_, d___ in d__.items():
#             print('\t\t', class_)
#             for order, n in d___.items():
#                 print('\t\t\t{}:{}'.format(order, n))

# rdp_f = 'raw_data/seqs_temp/unaligned RDP seqs/rdp_reference_unaligned_seqs.fa'
# seqs_hmm = SeqIO.parse(hmm_f, 'stockholm')
# seqs_rdp = SeqIO.parse(rdp_f, 'fasta')

# for record in seqs_hmm:
#     print(len(record.seq))
#     break

# for record in seqs_rdp:
#     print(len(record.seq))
#     break

# f = 'raw_data/seqs_temp/RDP_alignment/align_seqs.sto'
# seqs = SeqIO.parse(hmm_f, 'stockholm')

# ###########################################
# # Net repression/promotion
# ####################################################

# subjset_real = pl.base.SubjectSet.load('pickles/real_subjectset.pkl')

# fnames = [
#     'output_real/pylab24/real_runs/strong_priors/fixed_top/healthy1_5_0.0001_rel_2_5/ds0_is0_b5000_ns15000_mo-1_logTrue_pertsmult/graph_leave_out-1/mcmc.pkl',
#     'output_real/pylab24/real_runs/strong_priors/fixed_top/healthy0_5_0.0001_rel_2_5/ds0_is1_b5000_ns15000_mo-1_logTrue_pertsmult/graph_leave_out-1/mcmc.pkl'
# ]

# net_effects = []
# asv_set = []

# for fname in fnames:
#     chain = pl.inference.BaseMCMC.load(fname)

#     interactions = chain.graph[names.STRNAMES.INTERACTIONS_OBJ]
#     clustering = chain.graph[names.STRNAMES.CLUSTERING_OBJ]
#     asvs = clustering.items

#     asv_interactions = pl.variables.summary(interactions, only='mean', set_nan_to_0=True)['mean']
#     bf_asvs = interactions.generate_bayes_factors_posthoc(
#         prior=chain.graph[names.STRNAMES.CLUSTER_INTERACTION_INDICATOR].prior,
#         section='posterior')
#     mask_asvs = bf_asvs < 10
#     asv_interactions[mask_asvs] = 0

#     df = np.sum(asv_interactions, axis=1)
#     df = pd.DataFrame(df, index=asvs.names.order)

#     net_effects.append(df)
#     asv_set.append(asvs.names.order)

# # Condense to only the shared asvs
# asvs1 = set(list(asv_set[0]))
# asvs2 = set(list(asv_set[1]))
# asvs_both = list(asvs1.intersection(asvs2))

# asvs_both_idx = []
# for asv in asvs_both:
#     asvs_both_idx.append(subjset_real.asvs[asv].idx)
# idxs = np.argsort(asvs_both_idx)
# asvs_both = np.asarray(asvs_both)[idxs]



# df_healthy = net_effects[0]
# df_uc = net_effects[1]

# df_healthy = df_healthy.loc[asvs_both]
# df_uc = df_uc.loc[asvs_both]

# M_healthy = df_healthy.values.ravel()
# M_uc = df_uc.values.ravel()

# asvs = subjset_real.asvs
# f = open('sparse_net_effects.txt', 'w')
# for i, aname in enumerate(df_healthy.index):
#     asv = asvs[aname]
#     f.write('\n')
#     f.write(pl.asvname_formatter('%(name)s %(genus)s %(species)s', asv=asv, asvs=asvs))
#     f.write('\n-----------------------------\n')
#     f.write('healthy: {:.3E}\nulcerative colitis: {:.3E}\n'.format(
#         M_healthy[i], M_uc[i]))

# f.close()


# ####################################################
# # Filter RDP sequences
# ####################################################
# rdp_f = 'raw_data/seqs_temp/unaligned RDP seqs/rdp_reference_unaligned_seqs.fa'
# seqs = SeqIO.parse(rdp_f, 'fasta')
# min_l = None
# max_l = None
# i = 0
# i_long = 0
# lens = []

# seqs_keep = []

# trunc_thresh = 1600

# for record in seqs:
#     lens.append(len(record.seq))
#     if len(record.seq) < trunc_thresh:
#         seqs_keep.append(record)
        
#     if len(record.seq) > 1700:
#         i_long += 1

# SeqIO.write(
#     seqs_keep, 
#     'raw_data/seqs_temp/unaligned RDP seqs/seqs_trunc{}.fa'.format(trunc_thresh), 
#     format='fasta')

# plt.hist(x=lens, bins=50)
# plt.yscale('log')
# plt.title('Sequence Lengths')
# plt.xlabel('Lengths')
# plt.ylabel('Count')
# plt.savefig('raw_data/seqs_temp/unaligned RDP seqs/count.pdf')
# plt.close()

# ####################################################
# # Check gram negative or positive for different 
# ####################################################
# def is_gram_negative(asv):
#     '''Return true if the asv is gram - or gram positive
#     '''
#     if not asv.tax_is_defined('phylum'):
#         return None
#     if asv.taxonomy['phylum'].lower() == 'bacteroidetes':
#         return True
#     if asv.taxonomy['phylum'].lower() == 'firmicutes':
#         return False
#     if asv.taxonomy['phylum'].lower() == 'verrucomicrobia':
#         return True
#     if asv.taxonomy['phylum'].lower() != 'proteobacteria':
#         print(asv)
#         print('Not included')
#         return None

#     # Deltaproteobacteria are all gram -
#     return True

# chain_fnames = [
#     'output_real/pylab24/real_runs/strong_priors/fixed_top/healthy0_5_0.0001_rel_2_5/ds0_is1_b5000_ns15000_mo-1_logTrue_pertsmult/graph_leave_out-1/mcmc.pkl',
#     'output_real/pylab24/real_runs/strong_priors/fixed_top/healthy1_5_0.0001_rel_2_5/ds0_is0_b5000_ns15000_mo-1_logTrue_pertsmult/graph_leave_out-1/mcmc.pkl',
#     'output_real/pylab24/real_runs/perts_mult/fixed_top/healthy0_5_0.0001_rel_2_5/ds0_is1_b5000_ns15000_mo-1_logTrue_pertsmult/graph_leave_out-1/mcmc.pkl',
#     'output_real/pylab24/real_runs/perts_mult/fixed_top/healthy1_5_0.0001_rel_2_5/ds0_is0_b5000_ns20000_mo-1_logTrue_pertsmult/graph_leave_out-1/mcmc.pkl',
# ]
# for filename in chain_fnames:
#     chain = pl.inference.BaseMCMC.load(filename)
#     clustering = chain.graph[names.STRNAMES.CLUSTERING_OBJ]
#     asvs = clustering.items

#     print('\n\n')
#     print(filename)
#     print('--------------')
#     for cidx, cluster in enumerate(clustering):
#         print('cidx {}'.format(cidx+1))
#         totalness = [0,0,0]
#         for aidx in cluster.members:
#             asv = asvs[aidx]
#             a = is_gram_negative(asv)
#             if a is None:
#                 totalness[0] += 1
#             if a:
#                 totalness[1] += 1
#             else:
#                 totalness[2] += 1
#         print('\t NA, Gram neg, Gram pos', totalness)
# 

# ####################################################
# # Cluster interaction mutual networks
# ####################################################

# subjset_real = pl.base.SubjectSet.load('pickles/real_subjectset.pkl')

# fnames = [
#     'output_real/pylab24/real_runs/perts_mult/fixed_top/healthy1_5_0.0001_rel_2_5/ds0_is0_b5000_ns20000_mo-1_logTrue_pertsmult/graph_leave_out-1/mcmc.pkl',
#     'output_real/pylab24/real_runs/perts_mult/fixed_top/healthy0_5_0.0001_rel_2_5/ds0_is1_b5000_ns15000_mo-1_logTrue_pertsmult/graph_leave_out-1/mcmc.pkl'
# ]

# Ms = []
# asv_set = []

# for fname in fnames:
#     chain = pl.inference.BaseMCMC.load(fname)

#     interactions = chain.graph[names.STRNAMES.INTERACTIONS_OBJ]
#     clustering = chain.graph[names.STRNAMES.CLUSTERING_OBJ]
#     asvs = clustering.items

#     asv_interaction_trace = interactions.get_trace_from_disk(section='posterior')
#     asv_interactions = pl.variables.summary(asv_interaction_trace, only='mean', set_nan_to_0=True)['mean']
#     clus_interactions = main_base._condense_interactions(asv_interactions, clustering=clustering)

#     bf_asvs = interactions.generate_bayes_factors_posthoc(
#         prior=chain.graph[names.STRNAMES.CLUSTER_INTERACTION_INDICATOR].prior,
#         section='posterior')
#     bf_clus = main_base._condense_interactions(bf_asvs, clustering=clustering)

#     mask_clus = bf_clus < 10
#     mask_asvs = bf_asvs < 10

#     asv_interactions[mask_asvs] = 0
#     clus_interactions[mask_clus] = 0

#     asv_interactions[asv_interactions < 0] = -1
#     asv_interactions[asv_interactions > 0] = 1

#     df = pd.DataFrame(asv_interactions, 
#         index=asvs.names.order,
#         columns=asvs.names.order)

#     asv_set.append(asvs.names.order)
#     Ms.append(df)

# d = {'Ms': Ms, 'asv_set': asv_set}
# with open('dict_compare_diffuse_temp.pkl', 'wb') as handle:
#     pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

# fmt = '%(genus)s %(species)s %(name)s'

# with open('dict_compare_diffuse_temp.pkl', 'rb') as handle:
#     d = pickle.load(handle)
# asv_set = d['asv_set']
# Ms = d['Ms']


# # Condense to only the shared asvs
# asvs1 = set(list(asv_set[0]))
# asvs2 = set(list(asv_set[1]))
# asvs_both = list(asvs1.intersection(asvs2))

# asvs_both_idx = []
# for asv in asvs_both:
#     asvs_both_idx.append(subjset_real.asvs[asv].idx)
# idxs = np.argsort(asvs_both_idx)
# asvs_both = np.asarray(asvs_both)[idxs]

# df_healthy = Ms[0]
# df_uc = Ms[1]

# df_healthy = df_healthy.loc[asvs_both]
# df_healthy = df_healthy[asvs_both]
# df_uc = df_uc.loc[asvs_both]
# df_uc = df_uc[asvs_both]

# total_num_possible = np.sum(
#     np.logical_or(np.absolute(df_healthy.values), np.absolute(df_uc.values))
# )

# # Get the same and opposite - additionally condense teh dataframes to
# # only nonzeros for either the row or columns
# df_same = df_uc == df_healthy
# df_same[df_uc == 0] = False
# asvs_keep = []
# for asv in df_same:
#     if np.sum(df_same[asv]) + np.sum(df_same.loc[asv]) > 0:
#         asvs_keep.append(asv)
# df_same = df_same.loc[asvs_keep]
# df_same = df_same[asvs_keep]
# print('percent interactions the same sign', 
#     np.sum(df_same.values)/total_num_possible)

# labels_same = []
# ASVS = subjset_real.asvs
# for aidx, asv in enumerate(df_same.columns):
#     labels_same.append(pl.asvname_formatter(
#         format=fmt, asv=asv, asvs=ASVS, lca=False) + ' {}'.format(aidx))


# df_opp = df_uc == -df_healthy
# df_opp[df_uc == 0] = False
# asvs_keep = []
# for asv in df_opp:
#     if np.sum(df_opp[asv]) + np.sum(df_opp.loc[asv]) > 0:
#         asvs_keep.append(asv)
# df_opp = df_opp.loc[asvs_keep]
# df_opp = df_opp[asvs_keep]
# print('percent interactions the opposite sign', 
#     np.sum(df_opp.values)/total_num_possible)

# labels_opp = []
# ASVS = subjset_real.asvs
# for aidx, asv in enumerate(df_opp.columns):
#     labels_opp.append(pl.asvname_formatter(
#         format=fmt, asv=asv, asvs=ASVS, lca=False) + ' {}'.format(aidx))

# df_xor = np.logical_xor(np.absolute(df_uc.values), np.absolute(df_healthy.values))
# df_xor = pd.DataFrame(df_xor, columns=df_uc.columns, index=df_uc.index)
# asvs_keep = []
# for asv in df_xor:
#     if np.sum(df_xor[asv]) + np.sum(df_xor.loc[asv]) > 0:
#         asvs_keep.append(asv)
# df_xor = df_xor.loc[asvs_keep]
# df_xor = df_xor[asvs_keep]
# print('percent interactions xor sign', 
#     np.sum(df_xor.values)/total_num_possible)

# labels_xor = []
# ASVS = subjset_real.asvs
# for aidx, asv in enumerate(df_xor.columns):
#     labels_xor.append(pl.asvname_formatter(
#         format=fmt, asv=asv, asvs=ASVS, lca=False) + ' {}'.format(aidx))


# sns.heatmap(df_healthy)
# plt.title('healthy')
# plt.figure()
# sns.heatmap(df_uc)
# plt.title('ulcerative colitis')
# plt.figure()
# sns.heatmap(df_same, xticklabels=np.arange(df_same.shape[0]), yticklabels=labels_same,
#     cbar=False, cmap='Blues', linewidths=0.1, linecolor='lightgrey')
# plt.title('Same interaction signs')
# plt.figure()
# sns.heatmap(df_opp, xticklabels=np.arange(df_opp.shape[0]), yticklabels=labels_opp,
#     cbar=False, cmap='Blues', linewidths=0.1, linecolor='lightgrey')
# plt.title('Opposite interaction signs')
# plt.figure()
# sns.heatmap(df_xor, xticklabels=np.arange(df_xor.shape[0]), yticklabels=labels_xor,
#     cbar=False, cmap='Blues')
# plt.title('xor')
# plt.show()


# ####################################################
# # Cluster interaction unique networks
# ####################################################

# subjset_real = pl.base.SubjectSet.load('pickles/real_subjectset.pkl')

# fnames = [
#     'output_real/pylab24/real_runs/strong_priors/fixed_top/healthy1_5_0.0001_rel_2_5/ds0_is0_b5000_ns15000_mo-1_logTrue_pertsmult/graph_leave_out-1/mcmc.pkl',
#     'output_real/pylab24/real_runs/strong_priors/fixed_top/healthy0_5_0.0001_rel_2_5/ds0_is1_b5000_ns15000_mo-1_logTrue_pertsmult/graph_leave_out-1/mcmc.pkl'
# ]

# Ms = []
# asv_set = []
# clusterings = []

# for fname in fnames:
#     chain = pl.inference.BaseMCMC.load(fname)

#     interactions = chain.graph[names.STRNAMES.INTERACTIONS_OBJ]
#     clustering = chain.graph[names.STRNAMES.CLUSTERING_OBJ]
#     asvs = clustering.items

#     cs = []
#     for cluster in clustering:
#         cs.append([])
#         for aidx in cluster.members:
#             asvname = asvs[aidx].name
#             cs[-1].append(asvname)

#     asv_interaction_trace = interactions.get_trace_from_disk(section='posterior')
#     asv_interactions = pl.variables.summary(asv_interaction_trace, only='mean', set_nan_to_0=True)['mean']
#     clus_interactions = main_base._condense_interactions(asv_interactions, clustering=clustering)

#     bf_asvs = interactions.generate_bayes_factors_posthoc(
#         prior=chain.graph[names.STRNAMES.CLUSTER_INTERACTION_INDICATOR].prior,
#         section='posterior')
#     bf_clus = main_base._condense_interactions(bf_asvs, clustering=clustering)

#     mask_clus = bf_clus < 10
#     mask_asvs = bf_asvs < 10

#     asv_interactions[mask_asvs] = 0
#     clus_interactions[mask_clus] = 0

#     asv_interactions[asv_interactions < 0] = -1
#     asv_interactions[asv_interactions > 0] = 1

#     df = pd.DataFrame(asv_interactions, 
#         index=asvs.names.order,
#         columns=asvs.names.order)

#     asv_set.append(asvs.names.order)
#     Ms.append(df)
#     clusterings.append(cs)

# d = {'Ms': Ms, 'asv_set': asv_set, 'clusterings': clusterings}
# with open('dict_compare_strong_temp.pkl', 'wb') as handle:
#     pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

# fmt = '%(genus)s %(species)s %(name)s'

# with open('dict_compare_strong_temp.pkl', 'rb') as handle:
#     d = pickle.load(handle)
# asv_set = d['asv_set']
# Ms = d['Ms']
# clusterings = d['clusterings']

# df_healthy = Ms[0]
# clustering_healthy = clusterings[0]
# df_uc = Ms[1]
# clustering_uc = clusterings[1]

# # Condense to only the unique asvs
# asvs_healthy = set(list(df_healthy.columns))
# asvs_uc = set(list(df_uc.columns))

# asvs_both = asvs_healthy.intersection(asvs_uc)
# asvs_healthy_set = asvs_healthy - asvs_both
# asvs_uc_set = asvs_uc - asvs_both

# asvs_healthy = []
# for cluster in clustering_healthy:
#     for name in cluster:
#         if name in asvs_healthy_set:
#             asvs_healthy.append(name)

# asvs_uc = []
# for cluster in clustering_uc:
#     for name in cluster:
#         if name in asvs_uc_set:
#             asvs_uc.append(name)

# df_healthy = df_healthy.loc[asvs_healthy]
# df_healthy = df_healthy[asvs_healthy]
# df_uc = df_uc.loc[asvs_uc]
# df_uc = df_uc[asvs_uc]


# labels_healthy = []
# ASVS = subjset_real.asvs
# for aidx, asv in enumerate(df_healthy.columns):
#     labels_healthy.append(pl.asvname_formatter(
#         format=fmt, asv=asv, asvs=ASVS, lca=False) + ' {}'.format(aidx))
# labels_uc = []
# ASVS = subjset_real.asvs
# for aidx, asv in enumerate(df_uc.columns):
#     labels_uc.append(pl.asvname_formatter(
#         format=fmt, asv=asv, asvs=ASVS, lca=False) + ' {}'.format(aidx))

# sns.heatmap(df_healthy, xticklabels=np.arange(df_healthy.shape[0]), yticklabels=labels_healthy,
#     cmap='RdBu', linewidths=0.1, linecolor='lightgrey')
# plt.title('healthy')
# plt.figure()
# sns.heatmap(df_uc, xticklabels=np.arange(df_uc.shape[0]), yticklabels=labels_uc,
#     cmap='RdBu', linewidths=0.1, linecolor='lightgrey')
# plt.title('ulcerative colitis')


# plt.show()




# ####################################################
# # Alignment for E. Coli
# ####################################################
# # f = 'raw_data/seqs_temp/RDP11_Seqs_MITRE/alignment/aligned_RDP-11-1_TS_MinLen1200_Curated_cd-hit100.fa'
# # f = 'raw_data/seqs_temp/ECOLI_no_gaps/arb-silva.de_2020-08-05_id864575_tax_silva.fasta'
# f = 'raw_data/seqs_temp/align_seqs.sto'

# # seqs = SeqIO.parse(f, 'fasta')
# # for record in seqs:
# #     print('\n', record.id)
# #     print(record.seq)
# #     print(len(record.seq))

# seqs = AlignIO.read(f, 'stockholm')


# f = open('raw_data/seqs_temp/ecoli_align.txt', 'w')
# for i, record in enumerate(seqs):
#     # if i > 100:
#     #     break
#     seq = str(record.seq)
#     a = ''
#     i = 0
#     while i+80 < len(seq):
#         a += seq[i:i+80] + '\n'
#         i += 80
#     a += seq[i:] + '\n'
    

#     f.write('\n>{}\n{}\n'.format(record.id, a))
# f.close()



# syn = synthetic.make_semisynthetic(
#     chain='output_real/pylab24/real_runs/strong_priors/fixed_top/healthy1_5_0.0001_rel_2_5/ds0_is0_b5000_ns15000_mo-1_logTrue_pertsmult/graph_leave_out-1/mcmc.pkl',
#     min_bayes_factor=10, set_times=False, init_dist_timepoint=2)

# syn.save('semi_synthetic/preprocessed_semi_synthetic_healthy_strong_sparse.pkl')



# ####################################################
# # Eigan value heatmap
# ####################################################
# uc_filename = 'pickles/uc_diagrA.npy'
# healthy_filename = 'pickles/healthy_diagrA.npy'
# arr = np.load(uc_filename)
# print(arr.shape)

# X = []
# Y = []

# for i in range(arr.shape[0]):
#     print('{}/{}'.format(i,arr.shape[0]))
#     a = np.linalg.eigvals(arr[i])
#     # print(a)
#     for l in a:
#         real = l.real
#         imag = l.imag

#         # if real < 0:
#         #     real = -np.log10(-real)
#         # else:
#         #     real = np.log10(real)

#         # if imag < 0:
#         #     imag = -np.log10(-imag)
#         # else:
#         #     imag = np.log10(imag)

#         X.append(real)
#         Y.append(imag)
#         # print('\t', l)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# a = ax.hexbin(X,Y, bins='log', cmap='inferno')
# ax.set_xlabel('log10')
# ax.set_ylabel('log10')
# cb = fig.colorbar(a, ax=ax)
# plt.savefig('eiganvalue_heatmap.pdf')

# plt.show()

# ####################################################
# # Parse MDSINE data
# ####################################################
# from util import parse_mdsine1_diet, parse_mdsine1_cdiff

# subjset_cdiff = parse_mdsine1_cdiff('raw_data/mdsine1/data_cdiff/')
# subjset_diet = parse_mdsine1_diet('raw_data/mdsine1/data_diet/')

# os.makedirs('output_mdsine1/', exist_ok=True)
# subjset_cdiff.save('pickles/subjset_cdiff.pkl')
# subjset_diet.save('pickles/subjset_diet.pkl')

# # for asv in subjset_diet.asvs.names.order:
# #     if asv not in ['Strain26', 'Strain28', 'Strain29', 'Strain15', 'Strain27', 'Strain4']:
# #         continue
# #     fig = plt.figure()
# #     ax = fig.add_subplot(111)
# #     for subj in subjset_diet:
# #         pl.visualization.abundance_over_time(subj=subj, dtype='raw', yscale_log=False, 
# #             ax=ax, ylabel='ng strain DNA/ug total fecal DNA', title=asv, legend=False,
# #             plot_specific=asv)

# for subj in subjset_diet:
#     pl.visualization.abundance_over_time(subj=subj, dtype='raw', yscale_log=False)

# plt.show()


# sns.palplot(sns.dark_palette('Blue'))
# plt.show()



# ####################################################
# # Make heatmaps
# ####################################################
# chainname = 'output_real/pylab24/real_runs/strong_priors/healthy0_5_0.0001_rel_2_5/' \
#     'ds0_is1_b5000_ns15000_mo-1_logTrue_pertsmult/graph_leave_out-1/mcmc.pkl'
# chain = pl.inference.BaseMCMC.load(chainname)

# clustering = chain.graph[names.STRNAMES.CLUSTERING_OBJ]
# clustering.generate_cluster_assignments_posthoc(n_clusters=23)

# print('number of clusters', len(clustering))
# print(clustering)

# asvs = chain.graph.data.asvs

# # Get the abunances (oidx -> abundance)
# subjset = chain.graph.data.subjects
# abundances = np.zeros(len(asvs))
# for subj in subjset:
#     M = subj.matrix()['abs']
#     tidx_start = np.searchsorted(subj.times, 7)
#     tidx_end = np.searchsorted(subj.times, 21)
#     abundances += np.sum(M[:,tidx_start:tidx_end], axis=1)

# # Make the cluster map
# matrix = np.zeros(shape=(len(asvs), len(clustering)))
# for cidx, cluster in enumerate(clustering):
#     for oidx in cluster:
#         matrix[oidx, cidx] = abundances[oidx]

# d = {}
# taxas = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus']
# for tax in taxas:
#     df = pl.condense_matrix_with_taxonomy(matrix, asvs=asvs, 
#         fmt='%(' + tax + ')s')
#     print('\n', tax)
#     print(df.shape)
#     d[tax] = df

# with open('cluster_heatmap_abundance_d.pkl', 'wb') as handle:
#     pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

# perturbations = chain.graph.perturbations
# M = np.zeros(shape=(len(perturbations), len(clustering)))
# for pidx, pert in enumerate(perturbations):
#     pert_trace = pert.get_trace_from_disk(section='posterior')
#     pert_trace = main_base._condense_perturbations(pert_trace, clustering)
#     pert_vals = pl.variables.summary(pert_trace)['mean']

#     for cidx in range(len(clustering)):
#         oidx = list(clustering[clustering.order[cidx]].members)[0]
#         bf = main_base.perturbation_bayes_factor(pert, oidx)
#         if bf >= 10:
#             M[pidx, cidx] = pert_vals[cidx]

# pert_names = [pert.name for pert in perturbations]
# clusternames = ['Cluster {}'.format(cidx+1) for cidx in range(len(clustering))]
# df = pd.DataFrame(M, index=pert_names, columns=clusternames)
# # print(df)
# with open('perturbation_heatmap.pkl', 'wb') as handle:
#     pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)



# ####################################################
# # Continue
# ####################################################
# chainname = 'output_real/pylab24/joint_run/healthy-1_5_0.0001_rel_2_5/' \
#     'ds0_is0_b5000_ns15000_mo-1_logTrue_pertsmult/graph_leave_out-1/mcmc.pkl'

# chain = pl.inference.BaseMCMC.load(chainname)
# print(chain.sample_iter)

# graph = chain.graph[names.STRNAMES.GROWTH_VALUE].get_trace_from_disk

####################################################
# Test semisynthetic
####################################################
# chainname = 'output_real/pylab24/real_runs/strong_priors/fixed_top/healthy1_5_0.0001_rel_2_5/' \
#     'ds0_is0_b5000_ns15000_mo-1_logTrue_pertsmult/graph_leave_out-1/mcmc.pkl'

# chain = pl.inference.BaseMCMC.load(chainname)
# real_subjectset = chain.graph.data.subjects

# synth = synthetic.make_semisynthetic(chain=chainname, min_bayes_factor=10, 
#     init_dist_timepoint=1.5)

# synth.save('raw_data/temp_semi_synth.pkl')
# synth = synthetic.SyntheticData.load('raw_data/temp_semi_synth.pkl')

# times = synth.master_times
# required = [times[0], times[-1]]
# for perturbation in synth.dynamics.perturbations:
#     required.append(perturbation.start)
#     required.append(perturbation.end)
# synth.set_timepoints(synthetic.subsample_timepoints(times, int(0.6*len(times)),required))

# pv = model.MultiplicativeGlobal(asvs=synth.asvs)
# pv.value = 0.2**2

# synth.generate_trajectories(dt=0.005, processvar=pv)
# synth.generate_trajectories(dt=0.005, processvar=pv)
# subjset = synth.simulateRealRegressionDataNegBinMD(a0=config.NEGBIN_A0, 
#     a1=config.NEGBIN_A1, qpcr_noise_scale=0.34, subjset=real_subjectset)

# pl.visualization.abundance_over_time(subjset.iloc(0), dtype='abs', yscale_log=True,
#     legend=False, clustering=synth.dynamics.clustering, color_code_clusters=True)
# pl.visualization.abundance_over_time(subjset.iloc(1), dtype='abs', yscale_log=True,
#     legend=False, clustering=synth.dynamics.clustering, color_code_clusters=True)
# pl.visualization.abundance_over_time(real_subjectset.iloc(0), dtype='abs', yscale_log=True,
#     legend=False, clustering=synth.dynamics.clustering, color_code_clusters=True)
# pl.visualization.abundance_over_time(real_subjectset.iloc(1), dtype='abs', yscale_log=True,
#     legend=False, clustering=synth.dynamics.clustering, color_code_clusters=True)
# plt.show()

# ####################################################
# # Phylogeny
# ####################################################

# class Mer:

#     def __init__(self, subjset, tree):
#         self.subjset = subjset
#         self.tree = tree

#     def set_colors(self, taxlevel):
#         self.taxlevel = taxlevel
#         taxonomies = {}
#         for asv in self.subjset.asvs.names.order:
#             tax = self.subjset.asvs[asv].taxonomy[self.taxlevel]
#             # print()
#             # print(tax, type(tax))
#             if type(tax) == float:
#                 # print('here')
#                 tax = 'nan'
                
#             if tax not in taxonomies:
#                 taxonomies[tax] = set()
#             taxonomies[tax].add(asv)
#         print('N taxas', len(taxonomies))


#         cs_ = ['red', 'blue', 'gray', 'green', 'yellow', 
#             'coral', 'pink', 'purple', 'black', 'darkolivegreen', 'navy', 'lightgreen', 'lightblue',
#             'cyan']
#         self.colors = {}
#         for i,tax in enumerate(taxonomies):
#             self.colors[tax] = cs_[i]
#             print('{}: {}'.format(tax, cs_[i]))


#     def node_layout(self, node):
#         if node.is_leaf():
#             name = node.name
#             tax = self.subjset.asvs[name].taxonomy[self.taxlevel]
#             if type(tax) == float:
#                 tax = 'nan'
#             color = self.colors[tax]
#             N = ete3.AttrFace('name', fsize=50, fgcolor=color)
#             ete3.faces.add_face_to_node(N, node, column=0, position='aligned')
#             nstyle = ete3.NodeStyle()
#             nstyle['fgcolor'] = color
#             nstyle['size'] = 30
#             node.set_style(nstyle)
        

# asvs_to_keep = 'healthy_1_asvs.txt'
# subjset = pl.SubjectSet.load('pickles/real_subjectset.pkl')
# tree = ete3.Tree('raw_data/phylogenetic_tree_branch_len_preserved.nhx')

# # Delete asvs that dont pass filtering
# f = open(asvs_to_keep, 'r')
# asvs = []
# for line in f:
#     asvs.append(line.replace('\n', ''))

# asvs_to_delete = []
# for asv in subjset.asvs.names.order:
#     if asv not in asvs:
#         asvs_to_delete.append(asv)
# subjset.pop_asvs(asvs_to_delete)
# tree.prune(asvs, True)


# # for name in asvs:
# #     print('{}: {}'.format(name, tree.get_distance(name)))

# # threshold = 0.5
# # tree,b = pl.coarsen_phylogenetic_tree(tree, threshold)

# # for leaf in tree.get_leaf_names():
# #     print('{}: {}'.format(leaf, tree.get_distance(leaf)))

# # print(tree)
# # for k,v in b.items():
# #     print('{}:{}'.format(k,v))


# # sys.exit()

# TAXLEVEL = 'order'

# obj = Mer(subjset=subjset, tree=tree)
# obj.set_colors(taxlevel=TAXLEVEL)


# # for node in tree.traverse():
# #     node.img_style['size'] = 0
# #     if node.is_leaf():
# #         tax = None
# #         for a in taxonomies:
# #             if node.name in taxonomies[a]:
# #                 tax = a
# #                 break
# #         if tax is None:
# #             raise Exception('Get here :(')
# #         color = colors[tax]
# #         name_face = ete3.TextFace(node.name, fgcolor=color, fsize=50)
# #         node.add_face(name_face, column=0)

# # Color the sub-branches
# colored = set()
# for node in tree.traverse():
#     # # Check if parent is colored
#     # name = node.name
#     # parents = node.get_ancestors()
#     # there = False
#     # for parent in parents:
#     #     if parent.name in colored:
#     #         there = True
#     #         break
#     # if there:
#     #     continue

#     children = node.get_leaf_names()
#     taxs = set([obj.subjset.asvs[name].taxonomy[obj.taxlevel] for name in children])
#     if len(taxs) == 1:
#         tax = list(taxs)[0]
#         if type(tax) == float:
#             tax = 'nan'
#         # Color in the whole thing
#         style = ete3.NodeStyle()
#         style['size'] = 0
#         style["vt_line_color"] = obj.colors[tax]
#         style["hz_line_color"] = obj.colors[tax]
#         style["vt_line_width"] = 10
#         style["hz_line_width"] = 10
#         style["vt_line_type"] = 0 # 0 solid, 1 dashed, 2 dotted
#         style["hz_line_type"] = 0
#         node.set_style(style)

#         colored.add(node.name)

# ts = TreeStyle()
# # ts.show_leaf_name = False
# # # ts.show_branch_length = True
# ts.mode = 'c'
# ts.arc_start = -180
# ts.arc_span = 270
# # ts.scale = 10
# # ts.legend.add_face(ete3.TextFace(obj.taxlevel.capitalize()), column=0)
# # ts.legend.add_face(ete3.TextFace("0.5 support"), column=1)
# # tree.show(tree_style=ts)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# for k,v in obj.colors.items():
#     ax.plot([],[],color=v, label=k)
# ax.legend(title=TAXLEVEL.capitalize())

# ax.spines['top'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.xaxis.set_major_locator(plt.NullLocator())
# ax.xaxis.set_minor_locator(plt.NullLocator())
# ax.yaxis.set_major_locator(plt.NullLocator())
# ax.yaxis.set_minor_locator(plt.NullLocator())

# plt.show()

# ####################################################
# # Regex
# ####################################################
# mno = 'abcdefghijklmnop'
# txt = '%(genus)s %(species)s %(species10)s 4%(species3)s'
# _SPECIESX_SEARCH = re.compile('\%\(species[0-9]+\)s')

# print(txt)

# X = _SPECIESX_SEARCH.search(txt)
# if X is not None:
#     while True:
#         X = X[0]
#         print()
#         print(type(X))
#         print(X)
#         n = int(X.replace('%(species', '').replace(')s',''))
#         txt = txt.replace(X, mno[:n])
#         X = _SPECIESX_SEARCH.search(txt)
#         if X is None:
#             break

# print(txt)

# ####################################################
# # Display phylogenetic trees
# ####################################################
# from Bio import Phylo

# tree = Phylo.read('output_real/RDP-11-1_TS_Processed_seqs_tog_speciesName.xml', 'phyloxml')
# tree.ladderize()
# Phylo.draw_ascii(tree)
# plt.savefig('output_real/phylo_tree.pdf')
# # plt.show()


# ####################################################
# # Get counts at different filtering stages
# ####################################################

# df = pd.read_csv('raw_data/counts.tsv', sep='\t')
# print(df.head())
# print(df.columns)
# print(df['sample_Forward reads'])
# df = df.set_index('sample_Forward reads')
# print(df.head)


# ####################################################
# # Time sparse matrix operations
# ####################################################

# d1=55875
# d2 = 157

# n = 1000

# Cdiag = np.random.rand(d1)
# C = scipy.sparse.dia_matrix((Cdiag, [0]), shape=(len(Cdiag), len(Cdiag))).tocsc()

# # Make random D
# size = int(.0375*d1*d2)
# D = np.random.rand(size)
# rows = np.random.randint(low=0, high=d1, size=size)
# cols = np.random.randint(low=0, high=d2, size=size)

# D = scipy.sparse.csc_matrix((D,(rows,cols)), shape=(d1,d2))

# print(C.shape)
# print(D.shape)

# # Cdense = pl.toarray(C)
# # Ddense = pl.toarray(D)

# # start = time.time()
# # for i in range(n):
# #     l = Ddense.T * Cdiag
# # t = time.time() - start
# # print('time dense', t/n)

# # start = time.time()
# # for i in range(n):
# #     l = D.T @ C
# # t = time.time() - start
# # print('time @', t/n)

# # start = time.time()
# # for i in range(n):
# #     l = D.T.dot(C)
# # t = time.time() - start
# # print('time dot', t/n)
# print(np.sum(((D.T.dot(C)) -(D.T @ C)).toarray()))

# # print((C @ D).toarray())
# # print(C.dot(D).toarray())

# # Cdiag = np.ones(5)
# # C = scipy.sparse.dia_matrix((Cdiag, [0]), shape=(len(Cdiag), len(Cdiag))).tocsc()
# # Cdense = pl.toarray(C)

# # a = np.arange(5).reshape(-1,1)

# # print(C.dot(a))



# ####################################################
# # Count asvs overlap
# ####################################################
# f0 = open('healthy_0_asvs.txt', 'r')

# a0 = f0.read()
# a0 = a0.split('\n')
# a0 = a0[:-1]
# a0 = set(a0)

# print(a0)
# f1 = open('healthy_1_asvs.txt', 'r')

# a1 = f1.read()
# a1 = a1.split('\n')
# a1 = a1[:-1]
# a1 = set(a1)
# print(a1)

# i = 0
# for ele in a0:
#     if ele in a1:
#         i += 1
# print(i)

# ####################################################
# # Testing going from cocluster to toarray
# ####################################################

# # @numba.jit(nopython=True, cache=True)
# def toarray_from_cocluster(coclusters):
#     aa = np.full(coclusters.shape[0], -1)
#     i = 0
#     a = []
#     for j in range(coclusters.shape[0]):
#         if aa[j] == -1:
#             l = set([])
#             for k in range(j,coclusters.shape[0]):
#                 if coclusters[j,k] == 1:
#                     l.add(k)
#                     aa[k] = i
#             a.append(l)
#             i += 1
#     ret = [list(l) for l in a]
#     return ret

# def mer(coclusters):

#     a = np.full(coclusters.shape[0], -1)
#     i = 0
#     for j in range(coclusters.shape[0]):
#         if a[j] == -1:
#             for k in range(coclusters.shape[0]):
#                 if coclusters[j,k] == 1:
#                     a[k] = i
#             i += 1
#     ret = [[] for _ in range(np.max(a)+1)]
#     for i, cidx in enumerate(a):
#         ret[cidx].append(i)
#     return ret
# coclusters = np.diag(np.ones(6))
# coclusters[0,1] = 1
# coclusters[1,0] = 1
# coclusters[0,5] = 1
# coclusters[5,0] = 1

# coclusters[4,3] = 1
# coclusters[3,4] = 1
# print(coclusters)
# print(toarray_from_cocluster(coclusters))
# start = time.time()
# for i in range(100000):
#     toarray_from_cocluster(coclusters)
# end = time.time()
# print('time numba:', (end-start)/1000)
# start = time.time()
# for i in range(100000):
#     mer(coclusters)
# end = time.time()
# print('time reg:', (end-start)/1000)

# ####################################################
# # Testing making replicates for synthetic
# ####################################################

# perturbations = [ 
#     (20,26,0.2, '1', [0.2, 0.4, 0.4], [0.5, 1, 2], 0.1),
#     (34,42,0.2, '2', [0.2, 0.4, 0.4], [0.5, 1, 2], 0.1),
#     (50,56,0.2, '1', [0.2, 0.4, 0.4], [0.5, 1, 2], 0.1),
# ]
# subjset_real = pl.SubjectSet.load('pickles/real_subjectset.pkl')
# syndata = synthetic.SyntheticData(log_dynamics=True, n_days=62)
# syndata.icml_topology(n_asvs=13, max_abundance=1e8)
# for pert in perturbations:
#     syndata.sample_single_perturbation(*pert)
# processvar = model.MultiplicativeGlobal(asvs=syndata.asvs)
# processvar.value = 0.1
# init_dist = pl.variables.Uniform(low=5e6, high=5e7)

# for N in [30,40,50,60,70]:
#     syndata.set_times(N=N)
#     for _ in range(1):
#         syndata.generate_trajectories(init_dist=init_dist, 
#             dt=0.001, 
#             processvar=processvar)


# subjset = syndata.simulateRealRegressionDataNegBinMD(
#     a0=1e-10, a1=1e-5, 
#     qpcr_noise_scale=1e-5, subjset=subjset_real)
# for i, subj in enumerate(subjset):
#     ax = pl.visualization.abundance_over_time(subj=subj, dtype='abs', legend=True,
#         taxlevel=None, set_0_to_nan=True, yscale_log=True, 
#         color_code_clusters=True, clustering=syndata.dynamics.clustering)

# plt.show()

# ####################################################
# # Making boxplots
# ####################################################

# df = main_base.make_df(basepath='output_ICML/boxplots/')
# df = main_base.make_df(basepath='output_mlcrr/boxplots/', df=df)

# print(df)
# print(df.columns)

# fig = plt.figure(figsize=(10,10))
# # for i in range(1,5):
# #     ax = main_base.make_boxplots(df, y='RMSE-interactions', x='Replicates',
# #         only={'Replicates': i}, yscale='log', 
# #         # hue='Model', 
# #         ax=fig.add_subplot(2,2,i))
# for i,mn in enumerate([0.05, 0.1, 0.2, 0.3]):
#     ax = main_base.make_boxplots(df, y='Variation of Information', x='Replicates',
#             only={'Measurement Noise': mn}, yscale='linear', 
#             # hue='Model', 
#             ax=fig.add_subplot(2,2,i+1))
# fig.suptitle('Variation of Information', size=20)
# plt.savefig('figure_1/variation_of_information.pdf')
# plt.close()



# ####################################################
# # Test cuda
# ####################################################

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# a = torch.DoubleTensor(np.arange(15).reshape(5,3)).to(device)
# b = torch.DoubleTensor(np.arange(30,45).reshape(5,3)).to(device) 

# print(torch.cat([a,b], 1))


# print(type(a))


# ####################################################
# # Plot abundances
# ####################################################

# subjset = pl.SubjectSet.load('pickles/real_subjectset.pkl')


# os.makedirs('plotting1/', exist_ok=True)

# fparams = config.FilteringConfig(healthy=False)
# subjset = filtering.consistency(subjset, dtype=fparams.DTYPE,
#     threshold=fparams.THRESHOLD,
#     min_num_consecutive=fparams.MIN_NUM_CONSECUTIVE,
#     colonization_time=fparams.COLONIZATION_TIME,
#     min_num_subjects=fparams.MIN_NUM_SUBJECTS)
# subj = subjset['7']


# label_format = '%(name)s: %(family)s, %(genus)s'
# ax = pl.visualization.abundance_over_time(
#     subj=subj, taxlevel=None, dtype='rel',
#     yscale_log=False, plot_specific=[1,4,19],
#     legend=True, title=None,# ylim=(1e3, 1e12),
#     include_errorbars=True)
# # label = pl.asvname_formatter(format=label_format, asv=asv,
# #     asvs=subj.asvs)
# ax.set_title('Relative Abundance', size=15)
# fig = plt.gcf()
# fig.tight_layout()
# plt.savefig('plotting1/rel.png')
# plt.close()

# ####################################################
# # Inspect
# ####################################################
# def whoami():
#     return inspect.stack()[1][3]
# def whosdaddy():
#     return inspect.stack()[2][3]

# def getstacktrace():
#     stack = inspect.stack()
#     # print(stack)
#     trace = [(stack[1][1], stack[1][2], stack[1][3])]
#     for i in range(2, len(stack)):
#         trace.append((stack[i][1], stack[i][2], stack[i][3]))
#     return trace


# def foo():
#     bar()

# @pl.util.inspect_trace()
# def bar():
#     raise IndexError('')


# # bar()
# # print('\n\n\n')
# foo()

# ####################################################
# # in/out degree
# # ####################################################

# a1 = np.array([[[True,False,False], [True,True,False], [False,False,True]], [[True,False,False], [True,True,False], [False,False,True]]])

# print(a1)
# print('in')
# print(np.sum(a1,axis=2))
# print('out')
# print(np.sum(a1, axis=1))


# ####################################################
# # Copy the difference between two folders into a new one
# ####################################################

# path1 = 'output_real/february/week2/graph_ds9880021_is11114500_b2000_ns4000_healthy5_0.00025_rel_2_3_True/valid_asvs/'
# path2 = 'output_real/february/week2/graph_ds9880021_is11114500_b2000_ns4000_healthy5_0.00025_rel_all_7_True/valid_asvs/'
# respath = 'output_real/february/week2/diff_folder_healthy_True/'

# files1 = os.listdir(path1)
# files2 = os.listdir(path2)

# os.makedirs(respath, exist_ok=True)

# for f in files1:
#     if f not in files2:
#         shutil.copyfile(path1+f, respath+f)

# for f in files2:
#     if f not in files1:
#         shutil.copyfile(path2+f, respath+f)

# ####################################################
# # Argparse
# ####################################################
# parser = argparse.ArgumentParser()
# parser.add_argument('--burnin', '-nb', type=int,
#     help='Total number of burnin steps',
#     dest='burnin', default=3000)
# parser.add_argument('--times', '-t', type=float,
#     dest='times', default=[0.5,1,2], nargs='+')

# args = parser.parse_args()
# print(args)

# ####################################################
# # Plot prior ontop of histogram
# ####################################################

# shape = 2
# scale = 0.5277010396076822

# l = 0.02418557
# h = 0.29225711

# xs = np.arange(l,h,step=(h-l)/100)

# ys = scipy.stats.gamma.pdf(xs, a=shape, scale=scale)

# fig = plt.figure()
# ax = fig.add_subplot(111)

# ax.plot(xs,ys)
# plt.show()


# ####################################################
# # Loading the array that crashed to cholesky
# ####################################################
# arr = np.load('this_is_what_made_cholesky_crash_44320.npy')
# plt.imshow(np.isnan(np.log(arr)))
# plt.show()


# ####################################################
# # Plot the noise in inference for the runs
# ####################################################
# percentchanges = [25] #, 50, 75, 100]
# measurement_noises = [0.01, 0.05, 0.1, 0.15]
# process_variances = [0.01]
# n_asvs = [20, 25, 30, 35]

# data_seed = np.arange(0,5)
# init_seed = np.arange(0,1)

# base_basepath = '/Volumes/dek15/perturbation_study/'

# paths = ['percentchange25/', 'pc25o25/', 'pc25o30/', 'pc25o25/']

# df = pd.DataFrame(columns=['n_asvs', 'measurement_noise', 'rmse_interactions', 'variation_of_information', 'rmse_trajectories'])

# for pcc in percentchanges:
#     # d[pcc] = {}
#     basepath = base_basepath + 'percentchange{}/'.format(pcc)
#     for mn in measurement_noises:
#         for iii in range(len(n_asvs)):

#             N_ASVS = n_asvs[iii]
#             basepath = base_basepath + paths[iii]

#             # d[pcc][mn] = {}
#             mnfolder = basepath + 'measurement_noise_{}/'.format(int(100*mn))
#             for p in process_variances:
#                 pfolder = mnfolder + 'process_variance_{}/'.format(int(100*p))
#                 # d[pcc][mn]['RMSE_interactions'] = []
#                 # d[pcc][mn]['RMSE_trajectories'] = []
#                 # d[pcc][mn]['variation_of_information'] = []
#                 for ds in data_seed:
#                     dsfolder = pfolder + 'data_seed_{}/'.format(ds)

#                     dirs = os.listdir(dsfolder)

#                     for l in dirs:
#                         if l == 'logs':
#                             continue
#                         lll = 'no{}'.format(N_ASVS)
#                         if lll not in l:
#                             continue
                        
#                         dd = dsfolder + l
#                         if os.path.isdir(dd):
#                             fname = dd + '/comparison_with_synthetic/errors.txt'
#                             try:
#                                 f = open(fname, 'r')
#                             except:
#                                 continue

#                             a = [N_ASVS, mn]
#                             for line in f:
#                                 line = line.strip()
#                                 if 'RMSE Interaction Matrix: ' in line:
#                                     n = float(line.replace('RMSE Interaction Matrix: ', ''))
#                                     a.append(n)
#                                 if 'Variation of Information: ' in line:
#                                     n = float(line.replace('Variation of Information: ', ''))
#                                     a.append(n)
#                                 if 'RMSE Trajectories: ' in line:
#                                     n = float(line.replace('RMSE Trajectories: ', ''))
#                                     a.append(n)

#                                 #make dataframe
#                                 try:
#                                     df_temp = pd.DataFrame(data=[a], columns=['n_asvs', 'measurement_noise', 'rmse_interactions', 'variation_of_information', 'rmse_trajectories'])
#                                 except:
#                                     continue
#                                 df = df.append(df_temp)

# for data in ['rmse_trajectories', 'variation_of_information', 'rmse_interactions']:
#     fig = plt.figure()
#     ax = fig.add_subplot(1,1,1)
#     ax = sns.boxplot(x='n_asvs', y=data, hue='measurement_noise', data=df, ax=ax)
#     ax.set_title(data)
# plt.show()

# ####################################################
# # Testing DASW
# ####################################################
# class Worker(pl.multiprocessing.PersistentWorker):
#     def __init__(self, first):
#         self.first = first

#     def initialize1(self, hpam1):
#         self.hpam1 = hpam1

#     def initialize2(self, hpam2):
#         self.hpam2 = hpam2

#     def run(self, x, y):

#         a = x * self.hpam1
#         b = y * self.hpam2
#         return (x,y), a + b + self.first


# first = 1
# hpam1 = 10
# hpam2 = 1000


# pool = pl.multiprocessing.PersistentPool(ptype='dasw')
# for i in range(10):
#     pool.add_worker(Worker(first=first))

# args = [{'hpam1': hpam1}]*10
# pool.map('initialize1', args)

# args = [{'hpam2': hpam2}]*10
# pool.map('initialize2', args)

# pool.staged_map_start(func='run')
# for i in range(100):
#     time.sleep(0.01)
#     x = i
#     y = i+2
#     pool.staged_map_put(args={'x':x, 'y':y})

# ret = pool.staged_map_get()

# for (x,y), ans in ret:
#     print('\n(x,y)', (x,y))
#     print(ans)
#     print(x*hpam1 + y*hpam2 + first)

# ####################################################
# # Variation of Information metric
# ####################################################
# def variation_of_information(X, Y, n):
#     sigma = 0.0
#     for x in X:
#         p = len(x) / n
#         for y in Y:
#             q = len(y) / n
#             r = len(set(x) & set(y)) / n
#             if r > 0.0:
#                 sigma += r * (math.log(r / p) + math.log(r / q))
#     return abs(sigma)

# X = [[0,1,2,3], [4,5,6,7]]
# Y = [[0,1,2,3], [4,5,6,7]]
# print('\nsame')
# print(variation_of_information(X,Y,8))

# X = [[0,1,2,3], [4,5,6,7]]
# Y = [[4,5,6,7], [0,1,2,3]]
# print('\ndifferent order')
# print(variation_of_information(X,Y,8))

# X = [[0,1] ,[2,3], [4,5,6,7]]
# Y = [[4,5,6,7], [0,1,2,3]]
# print('\nbroken up')
# print(variation_of_information(X,Y,8))

# X = [[0,1] ,[2,3,4,5,6,7]]
# Y = [[4,5,6,7], [0,1,2,3]]
# print('\nslightly different')
# print(variation_of_information(X,Y,8))

# X = [[0],[1] ,[2],[3],[4],[5],[6],[7]]
# Y = [[4,5,6], [7,0,1,2,3]]
# print('\ntotally different')
# print(variation_of_information(X,Y,8))

# print(math.log(8))

# a = scipy.sparse.csc_matrix((np.ones(4), (np.arange(4), np.arange(4))), shape=(4,4))
# print(a)

# b = np.arange(4).reshape(-1,1)
# print(b)

# print(a.dot(b).shape)

# #####################################################
# # Log normal
# #####################################################

# # def plot_

# healthy_patients = True
# filename = 'pickles/real_subjectset.pkl'
# ss = 0.150703479879726 * 2

# subjset_real = pl.SubjectSet.load(filename)
# if not healthy_patients:
#     sidxs = ['2','3','4','5']
# else:
#     sidxs = ['6','7','8','9','10']
# for sidx in sidxs:
#     subjset_real.pop_subject(sidx)



# init_data = []
# means = []
# for subj in subjset_real:
#     for t in subj.qpcr:
#         init_data.append(subj.qpcr[t].data)
#         means.append([subj.qpcr[t].mean]*len(init_data[-1]))

# # print(data)
# data = np.asarray(init_data * 10).ravel()

# fig = plt.figure()
# ax = fig.add_subplot(111)

# hist, bins = np.histogram(data, bins=50)
# hist = hist/np.sum(hist)
# # print('hist', hist)
# # print('bins', bins)
# logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))

# ax.hist(data, bins=logbins, alpha=0.4)
# ax.set_xscale('log')

# y = [np.exp(np.log(x) + ss * np.random.normal()) for x in data]


# hist, bins = np.histogram(y, bins=50)
# logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))

# ax.hist(y, bins=logbins, alpha = 0.4)


# plt.show()

# print(pl.random.truncnormal.sample(mean=mu, std=std, low=0, high=float('inf')))



# #####################################################
# # Numpy slicing testing
# #####################################################
# a = np.random.randn(87500, int(2500 * .1))
# rows = np.random.randint(a.shape[0], size=int(a.shape[0] * 0.85))
# cols = np.random.randint(a.shape[1], size=int(a.shape[1] * 0.85))
# # cols = np.arange(a.shape[1])

# start_time = time.time()
# for _ in range(10):
#     # a[rows][:, cols]
#     a[rows, :]
# dt_slow = time.time()-start_time
# print('slow time:', dt_slow)

# start_time = time.time()
# for _ in range(10):
#     # a[rows][:, cols]
#     aaa = np.take(a, indices=rows, axis=0)
#     # print('aaa.shape', aaa.shape)
# dt_slow = time.time()-start_time
# print('slow time:', dt_slow)

# start_time = time.time()
# for _ in range(10):
#     (a.ravel()[(cols + (rows * a.shape[1]).reshape((-1,1))).ravel()]).reshape(rows.size, cols.size)
# dt_fast = time.time()-start_time
# print('fast time:', dt_fast)
# print('faster', dt_slow/dt_fast)

# #####################################################
# # Scipy slicing testing
# #####################################################

# # Big matrix
# data = np.arange(500000)
# rows = np.random.randint(low=0, high=100, size=500000)
# cols = np.random.randint(low=0, high=5000, size=500000)
# X = scipy.sparse.coo_matrix((data,(rows,cols)),
#             shape=(70000, 40000)).tocsc()

# data = np.ones(600)
# rows = np.random.randint(low=0, high=40000, size=600)
# cols = np.random.randint(low=0, high=56, size=600)
# M = scipy.sparse.coo_matrix((data,(rows,cols)),
#             shape=(40000, 56)).tocsc()
# start = time.time()
# for i in range(5000):
#     a = X @ M
# print('time', (time.time()-start)/5000)

# indices = np.arange(5) #np.append(np.arange(10), np.arange(15, 31))
# data = np.ones(300)
# rows = np.random.randint(low=0, high=40000, size=300)
# cols = np.random.randint(low=0, high=5, size=300)
# M1 = scipy.sparse.coo_matrix((data,(rows,cols)),
#             shape=(40000, 5)).tocsc()
# start = time.time()
# print(a.shape)

# for i in range(5000):
#     a[:, indices] = X @ M1
# print('time2', (time.time()-start)/5000)

# print(X.shape)
# print(M.shape)
# print(a.shape)

# #####################################################
# # Make abundance plots for the filtering
# #####################################################

# healthy = True
# n_consecutive = 3
# threshold = 0.00025
# dtype = 'rel'
# n_subjects = 2
# colonization_time = 5
# subjset = pl.SubjectSet.load('pickles/real_subjectset.pkl')

# if not healthy:
#     sidxs = ['2','3','4','5']
# else:
#     sidxs = ['6','7','8','9','10']
# for sidx in sidxs:
#     subjset.pop_subject(sidx)

# subjset = filtering.consistency(subjset, dtype, threshold, n_consecutive, colonization_time,
#     n_subjects)

# print('left', len(subjset.asvs))

# basepath = 'figures/filt_{}consec_{}thresh_{}_{}subj_{}colonize_healthy{}/'.format(
#     n_consecutive, str(threshold).replace('.','dot'),
#     dtype, n_subjects, colonization_time, healthy)
# if os.path.isdir(basepath):
#     shutil.rmtree(basepath)
# os.makedirs(basepath, exist_ok=True)
# subjset.save(basepath + 'subjset.pkl')

# c_m = 5e6
# min_counts = 1
# figsize = (10,5)

# for subj in subjset:

#     plot_name_formatter = 'Subject {}, %(name)s\n%(class)s, %(family)s, %(genus)s'.format(subj.name)
#     subjpath = basepath + 'subj{}/'.format(subj.name)
#     os.makedirs(subjpath, exist_ok=True)

#     f = open(subjpath + 'about.txt', 'w')
#     f.write('Subject {}\n'.format(subj.name))
#     f.write('Filtering parameters:\n')
#     f.write('\tHealthy: {}\n'.format(healthy))
#     f.write('\tNumber of consecutive time points: {}\n'.format(n_consecutive))
#     threshold_type = None
#     if dtype == 'rel':
#         threshold_type = 'Relative abundance'
#     elif dtype == 'raw':
#         threshold_type = 'Counts'
#     else:
#         threshold_type = 'Absolute abundance'
#     f.write('\tThreshold Type: {}\n'.format(threshold_type))
#     f.write('\tThreshold: {}\n'.format(threshold))
#     f.write('\tMinimum number of subjects: {}\n'.format(n_subjects))
#     f.write('\tColonization time: {} timepoints\n\n'.format(colonization_time))

#     f.write('Plotting params:\n')
#     f.write('\tc_hat: {} count\n'.format(min_counts))
#     f.write('\tc_m: {}\n'.format(c_m))
#     f.close()
    

#     # qpcr and read depth
#     min_traj = []
#     qpcr_means = []
#     read_depths = []
#     thresh_traj = []
#     for t in subj.times:
#         Q_mean = subj.qpcr[t].mean
#         read_depth = np.sum(subj.reads[t])

#         if dtype == 'rel':
#             thresh_traj.append(threshold * Q_mean)
#         elif dtype == 'raw':
#             thresh_traj.append(threshold * Q_mean / read_depth)
#         else:
#             thresh_traj.append(threshold)

#         qpcr_means.append(Q_mean)
#         read_depths.append(read_depth)
#         if min_counts is not None:
#             min_traj.append(Q_mean * min_counts / read_depth)
#     if min_counts is not None:
#         min_traj = np.asarray(min_traj)
#     else:
#         min_traj=None

#     qpcr_means = np.asarray(qpcr_means)

#     # Plot the read depth and the qpcr means
#     fig = plt.figure()
#     ax1 = fig.add_subplot(111)
#     ax2 = ax1.twinx()

#     ln1 = ax1.plot(subj.times, qpcr_means, marker='.', color='red', 
#         label=r'$ \overline{Q} $')
#     ln2 = ax2.plot(subj.times, read_depths, marker='.', color='green', 
#         label=r'$ \sum_i r_i $')
#     fig.suptitle('qPCR and read Depth for subject {}'.format(subj.name))
#     ax1.set_xlabel('Days')
#     ax1.set_ylabel('CFUs/g')
#     ax1.set_yscale('log')
#     ax2.set_ylabel('Counts')

#     pl.visualization.shade_in_perturbations(ax1, subjset.perturbations, 
#         textcolor='grey', textsize=11)

#     lns = [ln1[0], ln2[0]]
#     labs = [l.get_label() for l in lns]
#     ax1.legend(lns, labs, bbox_to_anchor=(1.1,1))
#     fig.subplots_adjust(right=0.80)

#     plt.savefig(subjpath + 'subj_data.pdf')
#     plt.close()

#     matrix = subj.matrix()['abs']

#     for oidx, asv in enumerate(subj.asvs):
#         fig = plt.figure(figsize=figsize)
#         title = pl.asvname_formatter(
#             format=plot_name_formatter, asv=asv, asvs=subj.asvs)
#         fig.suptitle(title)
#         ax = fig.add_subplot(111)

#         ax = main_base.plot_single_trajectory(given_times=subj.times, times=None, 
#             data=matrix[oidx, :], latent=None, aux=None, truth=None, min_traj=min_traj,
#             ax=ax, title=asv.name, yscale_log=True, subjset=subjset, c_m=c_m)

#         ax.plot(subj.times, thresh_traj, color='blue', label='threshold', marker='.', 
#             alpha=0.5)

#         pl.visualization.shade_in_perturbations(ax, subjset.perturbations, 
#             textcolor='grey', textsize=10)
        
#         ax.legend(bbox_to_anchor=(1,1))
#         fig.subplots_adjust(top=0.84, right=.86, left=0.09)
#         plt.savefig(subjpath + '{}.pdf'.format(asv.name))
#         plt.close()

# #####################################################
# # Make num asvs over time plots
# #####################################################

# healthy = True
# subjset = pl.SubjectSet.load('pickles/real_subjectset.pkl')
# min_rel_abundance = np.asarray([1, 2, 3, 4, 5, 8, 10, 15, 20])/50000
# min_num_subjects = np.arange(1,6)
# perturbations = [(21.5, 28.5), (35.5, 42.5), (50.5, 57.5)]

# if not healthy:
#     sidxs = ['2','3','4','5']
# else:
#     sidxs = ['6','7','8','9','10']
# for sidx in sidxs:
#     subjset.pop_subject(sidx)

# matrices = []
# for subj in subjset:
#     matrices.append(subj.matrix(min_rel_abund=None)['raw'])

# for matrix in matrices:
#     print(matrix.shape)

# timeses = []
# for subj in subjset:
#     timeses.append(subj.times)

# num_non_zeros = []
# num_singles = []
# num_doubles = []
# for matrix in matrices:
#     num_non_zeros.append([])
#     num_singles.append([])
#     num_doubles.append([])
#     for col in range(matrix.shape[1]):
#         num_non_zeros[-1].append(len(np.where(matrix[:, col] > 0)[0]))
#         num_singles[-1].append(len(np.where(matrix[:, col] == 3)[0]))
#         num_doubles[-1].append(len(np.where(matrix[:, col] == 4)[0]))
        

# fig = plt.figure()
# ax = fig.add_subplot(3, 1, 1)

# for i in range(len(num_non_zeros)):
#     ax.plot(timeses[i], num_non_zeros[i], marker='o', markersize=4, 
#         label=subjset.iloc(i).name)
# # xtick and perturbations
# loc = plticker.MultipleLocator(base=3)
# ax.xaxis.set_major_locator(loc)
# ax.legend(bbox_to_anchor=(1,1))
# for start,end in perturbations:
#     ax.axvspan(
#         xmin=start,
#         xmax=end, 
#         facecolor='orange', 
#         alpha=0.25)
# ax.set_title('Number of nonzero ASVs')
# ax.set_ylabel('Count (# ASVs)')
# ax.set_ylabel('Day')

# ax = fig.add_subplot(3, 1, 2)
# for i in range(len(num_non_zeros)):
#     ax.plot(timeses[i], num_singles[i], marker='o', markersize=4, 
#         label=subjset.iloc(i).name)

# # xtick and perturbations
# loc = plticker.MultipleLocator(base=3)
# ax.xaxis.set_major_locator(loc)
# ax.legend(bbox_to_anchor=(1,1))
# for start,end in perturbations:
#     ax.axvspan(
#         xmin=start,
#         xmax=end, 
#         facecolor='orange', 
#         alpha=0.25)
# ax.set_title('Number of Singletons')
# ax.set_ylabel('Count (# ASVs)')
# ax.set_ylabel('Day')

# ax = fig.add_subplot(3, 1, 3)
# for i in range(len(num_non_zeros)):
#     ax.plot(timeses[i], num_doubles[i], marker='o', markersize=4, 
#         label=subjset.iloc(i).name)

# # xtick and perturbations
# loc = plticker.MultipleLocator(base=3)
# ax.xaxis.set_major_locator(loc)
# ax.legend(bbox_to_anchor=(1,1))
# for start,end in perturbations:
#     ax.axvspan(
#         xmin=start,
#         xmax=end, 
#         facecolor='orange', 
#         alpha=0.25)
# ax.set_title('Number of Doubletons')
# ax.set_ylabel('Count (# ASVs)')
# ax.set_ylabel('Day')

# plt.show()

# #####################################################
# # Taxonomic distributions
# #####################################################
# reads = pd.read_csv('raw_data/replicate_data/counts.txt', sep="\t", header=0)

# mouse = 2
# day = 10
# taxlevel = 'family'
# suffixes = ['-1A', '-1B', '-2A', '-2B', '-3A', '-3B']
# labels = []
# for suffix in suffixes:
#     labels.append('M{}-D{}{}'.format(mouse,day,suffix))

# reads = reads.set_index('asvName')
# reads_replicates = reads[labels]
# reads_replicates = reads_replicates.to_numpy()

# asvs = subjset_real.asvs

# subj = pl.base.Subject(parent=subjset_real, name='mer')
# subj.reads['Original'] = subjset_real['2'].reads[day]
# for i in range(6):
#     subj.reads[labels[i]] = reads_replicates[:, i].ravel()

# subj.times = ['Original'] + labels

# ax = pl.visualization.taxonomic_distribution_over_time(
#     subj, taxlevel=taxlevel, drop_nan_taxa=False, legend=True, 
#     title='Mouse {}, Day {}, {} level'.format(mouse, day, taxlevel), #plot_abundant=9,
#     label_formatter='%(class)s', xlabel='Samples', ylabel='Relative abundance')
# ax.tick_params(axis = 'x', rotation=45)
# plt.show()

# sys.exit()

# #####################################################
# # Calling a function with a str
# #####################################################
# class Foo:
#     def __init__(self, a, b):
#         self.a = a
#         self.b = b

#     def add(self):
#         return self.a + self.b


# c = Foo(5,6)

# print(getattr(c,'add')())


# #####################################################
# # Testing hwo fast it is to retrieve from 1d or 2d array in numpy
# #####################################################
# arr2d = np.arange(100).reshape(10,10)
# arr1d = np.arange(100).ravel()
# d = {}
# for i,l in enumerate(np.arange(0,100,0.1)):
#     d[l] = i
# # print(d)
# start = time.time()
# for i in range(1000000):
#     a = arr2d[5,6]
# time2d = time.time() - start
# start = time.time()
# for i in range(1000000):
#     a = arr1d[88]
# time1d = time.time() - start
# start = time.time()
# for i in range(1000000):
#     a = d[0.2]
# timedict = time.time() - start
# print('2d time', time2d)
# print('1d time', time1d)
# print('dict time', timedict)
# print(time2d/time1d)
# print(timedict/time1d)


# #####################################################
# # Testing getting reads from negbin
# #####################################################
# r_k = 50000
# a0 = 3e-5
# a1 = 0.0386 * .1

# rel = np.array([0.1, 0.2, 0.3, 0.4])

# phi = r_k * rel
# eps = a0 / rel + a1

# print(a0/rel)
# print(a1)

# a = pl.random.negative_binomial.sample(phi, eps)
# print('reads', a)
# print(a / np.sum(a))
# print(np.sum(a))


# #####################################################
# # Replicate data
# #####################################################
# from sklearn.decomposition import PCA

# data = pd.read_csv('raw_data/replicate_data/counts.txt', sep="\t", header=0)

# data = data.to_numpy()
# n_samples = data.shape[1]
# n_asvs = data.shape[0]

# data1 = data/np.sum(data,axis=0)
# d = np.zeros(shape=(6,3))
# d[:,0] = data1[4,0:6]
# d[:,1] = data1[4,6:12]
# d[:,2] = data1[4,12:]

# d = pd.DataFrame(data=d, columns=['Day 10', 'Day 8', 'Day 9'])



# dm = np.zeros(shape=(n_samples,n_samples))
# for i in range(n_samples):
#     for j in range(n_samples):
#         dm[i,j] = diversity.beta.braycurtis(data[:,i], data[:,j])

# plt.figure()
# sns.heatmap(dm)

# pca = PCA(n_components=2)
# pca = pca.fit(dm)
# xlabel = 'PC1 - {:.4f} explained_variance'.format(pca.explained_variance_[0])
# ylabel = 'PC2 - {:.4f} explained_variance'.format(pca.explained_variance_[1])

# projected = pca.transform(dm)
# print('projected shape', projected.shape)
# plt.figure()
# plt.scatter(projected[:, 0], projected[:, 1],
#     alpha=0.5,
#     color=['red']*6 + ['blue']*6 + ['green']*6)
# plt.xlabel(xlabel)
# plt.ylabel(ylabel)
# plt.title('Bray-Curtis PCoA projection')
# plt.show()


# #####################################################
# # Check how many non-zeros in the real data
# #####################################################
# print('regression data total')
# subjset_real = pl.SubjectSet.load('pickles/real_subjectset.pkl')
# for subject in subjset_real:
#     reads = subject.matrix()['raw']
#     ii = 0
#     for i in range(reads.shape[0]):
#         if np.any(reads[i,:] > 0):
#             ii += 1
#     print('\tsubject ({}) Amplified sequence variants with at least one read:'.format(subject.name), ii)

# print('regression data at days 8,9,10')
# subjset_real = pl.SubjectSet.load('pickles/real_subjectset.pkl')
# for subject in subjset_real:
    
#     arrs = (
#         subject.reads[8].reshape(-1,1), 
#         subject.reads[9].reshape(-1,1), 
#         subject.reads[10].reshape(-1,1))
#     reads = np.hstack(arrs)

#     ii = 0
#     for i in range(reads.shape[0]):
#         if np.any(reads[i,:] > 0):
#             ii += 1
#     print('\tsubject ({}) Amplified sequence variants with at least one read:'.format(subject.name), ii)

# print('regression data at days 0AM, 0PM, 1AM, 1PM')
# subjset_real = pl.SubjectSet.load('pickles/real_subjectset.pkl')
# for subject in subjset_real:

#     arrs = (
#         subject.reads[0].reshape(-1,1), 
#         subject.reads[1.0].reshape(-1,1), 
#         subject.reads[1.5].reshape(-1,1),
#         subject.reads[2.0].reshape(-1,1))
#     reads = np.hstack(arrs)

#     ii = 0
#     for i in range(reads.shape[0]):
#         if np.any(reads[i,:] > 0):
#             ii += 1
#     print('\tsubject ({}) Amplified sequence variants with at least one read:'.format(subject.name), ii)


# data = pd.read_csv('raw_data/replicate_data/counts.txt', sep="\t", header=0)
# data = data.to_numpy()
# ii = 0
# for i in range(data.shape[0]):
#     if np.any(data[i,:] > 0):
#         ii += 1
# print('replicate data Amplified sequence variants with at least one read:', ii)


# #####################################################
# # Testing diversity
# #####################################################
# a = synthetic.SyntheticData()
# a.set_asvs(20)
# a.set_cluster_assignments(5,'sequence')
# print(str(a.dynamics.clustering))

# a.shuffle_cluster_assignments(.25)
# print(str(a.dynamics.clustering))


# #####################################################
# # Testing h5py
# #####################################################

# def get_dataset(f, name):
#     f = h5py.File(f, 'r+')
#     a = f[name]
#     f.close()
#     return a

# fname = 'test2.hdf5'
# f = h5py.File(fname, 'w')
# dset = f.create_dataset('lol', (10,10,10), chunks=True)
# dset[:,5:,:] = 33
# f.close()

# print('here')
# l = get_dataset(fname, 'lol')
# print(l[()])


# #####################################################
# # Testing clustering
# #####################################################

# def _inidicator_initializer():
#     return pl.random.bernoulli.sample(0.5) >= 0.5

# def _value_initializer():
#     return pl.random.uniform.sample()


# syndata = synthetic.SyntheticData()
# syndata.set_asvs(n_asvs=10)

# clusters = [[0,1],[2],[3,4], [5,6,7,8,9]]

# clustering = pl.cluster.Clustering(asvs=syndata.asvs, clusters=clusters)
# cids = clustering.order
# interactions = pl.cluster.Interactions(clustering=clustering, 
#     use_indicators=True, value_initializer=_value_initializer,
#     indicator_initializer=_inidicator_initializer)

# print('\n\n')
# print('OG')
# print(clustering)
# print(interactions)

# print('\n\n')
# print('move to valid')
# clustering.move_item(idx=2, cid=cids[0])
# # cids = clustering.order
# print(clustering)
# print(interactions)

# print('\n\n')
# print('move to make new')
# clustering.move_item(idx=3, cid=cids[1])
# print(clustering)
# print(interactions)


# #####################################################
# # Testing reading in data with pandas
# #####################################################
# QPCR_DADA_DATA_FOLDER = '../raw_data/'
# path = QPCR_DADA_DATA_FOLDER + 'qPCR/16s qPCR/1d0am-6d4pm/well2sampleid.xlsx'
# wellplatetosample = pd.read_excel(path, index_col=0)
# path = QPCR_DADA_DATA_FOLDER + 'qPCR/16s qPCR/1d0am-6d4pm/wellctscores.xls'
# ctscore = pd.read_excel(
#     path, 
#     sheet_name='Results', 
#     header=32, 
#     index_col='Well Position')
# ctscore = ctscore['CT']
# index = []
# for row in ctscore.index:
#     _row = row[0]
#     _col = int(row[1:])
#     print('\nrow', row)
#     idx = wellplatetosample.loc[_row, _col]
#     if idx == 'NTC':
#         continue
#     index.append(idx)
#     print(ctscore[row])
# print(df.loc['A', 24])
# # print(type(df['1d0am']))


# #####################################################
# # Testing the sampling of a synthetic system
# #####################################################
# filename = 'pickles/mouse_set_real.pkl'
#
# mouseset = data_wrappers.MouseSet.load(filename=filename)
# mouseset.filter(
#     min_rel_abundance=50/10000,
#     min_num_subjects=3,
#     min_num_counts=10,
#     min_num_consecutive=3)
#
# mouseset = mouseset.mouse_set1
#
# print('num ASVs', len(mouseset.data_tables.asvs))
#
# abs = mouseset.mice['2'].matrix(include_nan=False)['abs']
#
# # Get the last 5 days and take mean to get the steady states
# abs = abs[:,-5:]
# ss = np.mean(abs, axis=1)
# ss = ss[ss > 0]
#
# print('steady-states:\n', ss.reshape(-1,1))
#
# print('num ASVs', len(ss))
#
# np.save('temp_ss.npy', ss)
#
# ss = np.load('temp_ss.npy')
# n_replicates = 2
# n_clusters = 5
# n_asvs = 50
# ndays = 70
#
# timestamps = [(14., 21.), (28., 35.), (42., 49.), (56., 63.)]
# pert_vals = [-.99, -.99, .99, -.99]
# p_ind = np.zeros(n_clusters, dtype=bool)
#
# init_dist = pl.variables.TruncatedNormal(mean=1e7, var=1e16, low=0., high=float('inf'))
#
# load_syndata = False
# if load_syndata:
#     syndata = synthetic.SyntheticData.load('output/syndata_{}_{}_{}.pkl'.format(n_asvs,n_clusters,ndays))
#
# else:
#     # make the synthetic data object
#     syndata = synthetic.SyntheticData()
#
#     # Sample the interactions and growth rates
#     syndata.sample_system(
#         steady_state=ss, n_asvs=n_asvs, n_clusters=n_clusters, cluster_evenness='even',
#         topology='even', interaction_scaling_factor=10, replace=True,
#         threshold_pos_orthant_check=0.1)
#
#     # Set the perturbations
#     for i in range(len(timestamps)):
#         start = timestamps[i][0]
#         end = timestamps[i][1]
#         ind = np.asarray(p_ind)
#         ind[[i,i+1]] = True
#
#         syndata.set_single_perturbation(start=start,end=end,value=pert_vals[i],indicator=ind,
#             value_per_asv=False)
#
#     syndata.generate_trajectories(init_dist=init_dist,
#         n_replicates=n_replicates, dt=0.0005, n_days=ndays, process_var=True,
#         v1=0.3**2, v2 = 1., c_m=1e6, times=np.arange(0, ndays, .5))
#
#     syndata.save('output/syndata_{}_{}_{}.pkl'.format(n_asvs,n_clusters,ndays))
#
#     # pl.visualization.render_interaction_strength(
#     #     interaction_matrix=syndata.clustering.interactions.get_ele2ele_value_matrix(),
#     #     log_scale=True, asvs=syndata.asvs, clustering=syndata.clustering)
#     # plt.savefig('output/interaction_strength_{}_{}.pdf'.format(n_asvs, n_clusters))
#     # plt.close()
#     # pl.visualization.render_cocluster_proportions(
#     #     coclusters=syndata.clustering.clusters.gen_coclusters(), asvs=syndata.asvs,
#     #     clustering=syndata.clustering)
#     # plt.savefig('output/coclusters_{}_{}.pdf'.format(n_asvs, n_clusters))
#     # plt.close()
#
# order = syndata.clustering.order
#





# #####################################################
# #Looking at the trajectories to see if it is a good system
# #####################################################
# init_dist = pl.variables.TruncatedNormal(mean=8, var=5, low=5, high=15)
# n_replicates = 2
# syndata = synthetic.SyntheticData('ICML_rescale')
# syndata.set_single_perturbation(start=14, end=21, magnitude=-.25, indicator=[False, False, True])
# syndata.generate_trajectories(init_dist=init_dist,
#         n_replicates=n_replicates, dt=0.005, n_days=50, process_var=True,
#         v1=0.015**2, v2 = .1,
#         c_m=1,#1e6,
#         times=np.arange(0, 28))
# for i in range(n_replicates):
#     synthetic.plot_trajectories(
#         data=syndata.data[i],
#         times=syndata.times[i],
#         asvs=syndata.asvs,
#         clustering=syndata.clustering,
#         color_code_clusters=True,
#         legend=True,
#         use_markers=True,
#         perturbations=syndata.perturbations,
#         separate_panel_per_cluster=False,
#         yscale_log=False,
#         title='Replicate {}'.format(i))
# plt.show()



#####################################################
# Testing the time array making for the str format
#####################################################
# n_days = 65
# perturbations = [(21,28), (35,42), (50,57)]
# times = 'darpa-study-sampling'
# if type(times) == str:
#     # Do the addition in set format so that there are no duplicate days. Keep
#     # everything as floats. We also do the trimming of days at the end
#     if times == 'dense-around-perturbation':
#         times = set(np.arange(n_days, dtype=float).tolist())
#         if perturbations is not None:
#             # Double sample around the changes of the perturbation
#             for start,end in perturbations:
#                 for t in np.arange(start-1, start+2, 0.5):
#                     times.add(t)
#                 for t in np.arange(end-1, end+2, 0.5):
#                     times.add(t)
#     elif times == 'darpa-study-sampling':
#         # Dense sampling for the initial 4 days
#         times = np.asarray(np.arange(0,4,0.5), dtype=float)
#         # Next 7 days are only single samples
#         times = np.append(times, 
#             np.asarray(np.arange(4,11), dtype=float))
#         # All the rest of the days there is a sample every other day
#         times = np.append(times, 
#             np.asarray(np.arange(11, n_days, 2), dtype=float))
#         times = set(times.tolist())
#         if perturbations is not None:
#             # Perform dense sampling around the perturbation
#             for start,end in perturbations:
#                 # double sample around the changes
#                 for t in np.arange(start-1, start+2, 0.5):
#                         times.add(t)
#                 for t in np.arange(end-1, end+2, 0.5):
#                         times.add(t)
#                 # Single samples 2 days after the changes for 2 days
#                 for t in np.arange(start+2, start+4):
#                         times.add(float(t))
#                 for t in np.arange(end+2, end+4):
#                         times.add(float(t))
#     else:
#         raise ValueError('`times` ({}) not recognized'.format(times))
#     # Convert the times back into a np.ndarray of floats
#     # Because the times were a set we know there are no duplicate days
#     times = np.sort(list(times))
#     # check if we went over the number of days or if there are any negative days
#     # This should not happen but it might if the perturbation starts/ends close
#     # to the start or end of the sampling of the sampling 
#     times = times[times >=0]
#     times = times[times < n_days]
# print('n_days', n_days)
# print('perturbations', perturbations)
# print('times', times)



#####################################################
# Testing indexing out indexes
#####################################################
# idxs = np.arange(15,dtype=int)
# perts = [(3,5), (8,11)]
#
# # print('idxs',idxs)
#
# idxs_to_delete = []
# for start,end in perts:
#     # print('\n\nstart', start)
#     # print('end',end)
#     start_dtidx = start + 1
#     end_dtidx = end + 1
#
#     idxs_to_delete.append(
#         set(np.arange(start_dtidx, end_dtidx, dtype=int).tolist()))
#     # print('idxs_to_delete', idxs_to_delete)
# idxs_to_delete = np.sort(list(set.union(*idxs_to_delete)))
#
# # print('\n\nidxs_to_delete', idxs_to_delete)
#
# dt_idxs = np.delete(idxs, idxs_to_delete)
# print('dt_idxs now\n', dt_idxs)
#
# n_asvs = 13
# idxs = np.array([], dtype=int)
# for dtidx in dt_idxs:
#     print('\ndtidx', dtidx)
#     a = np.arange(n_asvs*dtidx, (dtidx+1)*n_asvs, dtype=int)
#     print('a', a)
#     idxs = np.append(idxs, a)
#     print('curr_idxs', idxs)
#
# print('\n\nend')
# print(idxs)








#####################################################
# Testing dataframes
#####################################################
# data = np.arange(100, dtype=float).reshape(10,10)
#
# cols = np.arange(10, dtype=float).tolist()
#
# cols[0] = 'mass'
#
#
# df = pd.DataFrame(
#     data=data,
#     index=['ASV{}'.format(i) for i in range(10)],
#     columns=cols)
#
# idx = list(df.index)
#
# print(np.asarray(df['mass']))














# #####################################################
# Testing the validity of the custom distributions
# #####################################################
# from math import sqrt as SQRT
# from math import pi as _PI
# from math import erf as ERF
# from numpy import exp as EXP
# from numpy import log as LOG
# from numpy import square as SQD
#
# _INV_SQRT_2PI = 1/SQRT(2*_PI)
# _LOG_INV_SQRT_2PI = LOG(1/SQRT(2*_PI))
# _LOG_2PI = LOG(2*_PI)
# _INV_SQRT_2 = 1/SQRT(2)
# _LOG_INV_SQRT_2 = LOG(1/SQRT(2))
# _LOG_ONE_HALF = LOG(0.5)
# _NEGINF = float('-inf')
#
# class _BaseSample:
#
#     @staticmethod
#     def rvs(*args, **kwargs):
#         '''Sample a random variable from the distribution
#         '''
#         raise UndefinedError('This function is undefined.')
#
#     @staticmethod
#     # @numba.jit(nopython=True, fastmath=True, parallel=True, cache=True)
#     def pdf(*args, **kwargs):
#         '''Calculate the pdf
#         '''
#         raise UndefinedError('This function is undefined.')
#
#     @staticmethod
#     # @numba.jit(nopython=True, fastmath=True, parallel=True, cache=True)
#     def logpdf(*args, **kwargs):
#         '''Calculate the logpdf
#         '''
#         raise UndefinedError('This function is undefined.')
#
#     @staticmethod
#     # @numba.jit(nopython=True, fastmath=True, parallel=True, cache=True)
#     def cdf(*args, **kwargs):
#         '''Calculate the cdf
#         '''
#         raise UndefinedError('This function is undefined.')
#
#     @staticmethod
#     # @numba.jit(nopython=True, fastmath=True, parallel=True, cache=True)
#     def logcdf(*args, **kwargs):
#         '''Calculate the logcdf
#         '''
#         raise UndefinedError('This function is undefined.')
#
#
# class normal(_BaseSample):
#     '''Scalar normal distribution
#     '''
#     @staticmethod
#     def rvs(mean, std, size=None):
#         '''Sample from a normal random distribution
#         '''
#         return npr.normal(mean, std, size=size)
#
#     @staticmethod
#     # @numba.jit(nopython=True, fastmath=True, parallel=True, cache=True)
#     def pdf(value, mean, std):
#         return _INV_SQRT_2PI * EXP(-0.5*SQD((value-mean)/std)) / std
#
#     @staticmethod
#     # @numba.jit(nopython=True, fastmath=True, parallel=True, cache=True)
#     def logpdf(value, mean, std):
#         return _LOG_INV_SQRT_2PI + (-0.5*SQD((value-mean)/std)) - LOG(std)
#
#     @staticmethod
#     # @numba.jit(nopython=True, fastmath=True, parallel=True, cache=True)
#     def cdf(value, mean, std):
#         return 0.5 * (1 + ERF(_INV_SQRT_2 * ((value-mean)/std)))
#
#     @staticmethod
#     # @numba.jit(nopython=True, fastmath=True, parallel=True, cache=True)
#     def logcdf(value, mean, std):
#         return _LOG_ONE_HALF + LOG(1 + ERF(_INV_SQRT_2 * ((value-mean)/std)))
#
#
# def pdf(value, mean, std, low=None, high=None):
#     # if low is None:
#     #     low = float('-inf')
#     # if high is None:
#     #     high = float('inf')
#     if low is not None and high is not None:
#         # both lower and high are defined
#         if value > high or value < low:
#             return 0.
#         return normal.pdf(value, mean, std)/(std*(normal.cdf(high,mean,std) - normal.cdf(low,mean,std)))
#
#     elif low is None and high is None:
#         # Just a regular normal distribution
#         return normal.pdf(value, mean, std)
#
#     elif low is None:
#         if value > high:
#             return 0.
#         return normal.pdf(value, mean, std)/normal.cdf(high,mean,std)
#
#     else:
#         # only high is None
#         if value < low:
#             return 0.
#         return normal.pdf(value, mean, std)/(1 - normal.cdf(low,mean,std))
#
# def logpdf(value, mean, std, low=None, high=None):
#     # if low is None:
#     #     low = float('-inf')
#     # if high is None:
#     #     high = float('inf')
#     if low is not None and high is not None:
#         # both lower and high are defined
#         if value > high or value < low:
#             return float('-inf')
#         return normal.logpdf(value, mean, std) - (LOG(std) + \
#             LOG(normal.cdf(high,mean,std) - normal.cdf(low,mean,std)))
#
#     elif low is None and high is None:
#         # Just a regular normal distribution
#         return normal.logpdf(value, mean, std)
#
#     elif low is None:
#         if value > high:
#             return float('-inf')
#         return normal.logpdf(value, mean, std) - normal.logcdf(high,mean,std)
#     else:
#         # only high is None
#         if value < low:
#             return float('-inf')
#         return normal.logpdf(value, mean, std) - LOG(1 - normal.cdf(low,mean,std))
#
# value = 12
# mean = 5
# std = 6
# low = 0
# high = 45
#
# print(pdf(value,mean,std,low,high))
#
#
# print(scipy.stats.truncnorm.pdf(value, (low-mean)/std, (high-mean)/std, mean, std))
