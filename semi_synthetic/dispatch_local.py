'''Run a small set of runs locally

Note that the config file we use is `MDSINE2/semi_synthetic/config`, not `MDSINE2/config`
'''

import os
import argparse
import sys

import config

n_samples = 10 # Total number of gibb steps
burnin = 5 # How many gibb steps for burnin
ckpt = 5 # How often to save the traces to disk
subjset_path = 'output/base_data/' # Where to save the ground truth data
output_path = 'output/runs/' # Where we want to save the runs
os.makedirs(subjset_path, exist_ok=True)
os.makedirs(output_path, exist_ok=True)

# Set up the logging
config.LoggingConfig(basepath='output/')

# Specify the Meshes that we want to do
# -------------------------------------
meshes = [
    (
        [5], # Number of replicates
        [55], # Number of timepoints
        2, # Total number of data seeds
        1, # Total number of initialization seeds
        [0.1, 0.2, 0.4], # Measurement Noises
        [0.1], # Process variances
        [1], # Clustering on/off
        0, # Uniform sampling of timepoints
    ),    
    (
        [5],  # Number of replicates
        [35, 50, 65],  # Number of timepoints
        2,  # Total number of data seeds
        1,  # Total number of initialization seeds
        [0.3],  # Measurement Noises
        [0.1],  # Process variances
        [1],  # Clustering on/off
        1,  # Uniform sampling of timepoints
    )]

# Make the subjectsets as a whole grid
# ------------------------------------
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
                                arr = [nr, nt, d, i, mn, pv, uniform_sampling]
                                arguments_global.append(arr)

lst_replicates = ' '.join(agg_repliates)
lst_measurement_noises = ' '.join(agg_measurement_noise)
lst_times = ' '.join(agg_times)
lst_process_variances = ' '.join(agg_process_variances)

command = 'python make_subjsets.py -b {basepath} -nr {nrs} -m {mns} -p {pvs} -d {nd} -dset semi-synthetic -nt {nts}'.format(
    basepath=subjset_path, nrs=lst_replicates, mns=lst_measurement_noises,
    pvs=lst_process_variances, nd=max_dataseeds, nts=lst_times)
os.system(command)

# Run each inference
# ------------------
for mesh in arguments_global:
    n_replicates = mesh[0]
    n_timepoints = mesh[1]
    data_seed = mesh[2]
    init_seed = mesh[3]
    measurement_noise = mesh[4]
    process_variance = mesh[5]
    uniform_sampling = mesh[6]

    command = 'python main_mcmc.py -d {d} -i {i} -m {m} -p {p} -b {b} -db {db} -ns {ns} -nb {nb} -nt {nt} -nr {nr} -us {us} -ckpt {ckpt}'.format(
        d=data_seed, i=init_seed, m=measurement_noise, p=process_variance, 
        b=output_path, db=subjset_path, ns=n_samples, nb=burnin, nt=n_timepoints, 
        nr=n_replicates, us=uniform_sampling, ckpt=ckpt)
    os.system(command)


