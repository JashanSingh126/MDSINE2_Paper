#!/bin/bash

NEGBIN = "../output/negbin/replicates/mcmc.pkl"
SEED = "0"
BURNIN = "5000"
N_SAMPLES = "15000"
CHECKPOINT = "100"
MULTIPROCESSING = "1"

# Healthy cohort
# --------------
python ../step_5_infer_mdsine2.py \
    --input ../output/processed_data/gibson_healthy_agg_taxa_filtered.pkl \
    --negbin $NEGBIN \
    --seed $SEED \
    --burnin $BURNIN \
    --n-samples $N_SAMPLES \
    --checkpoint $CHECKPOINT \
    --multiprocessing $MULTIPROCESSING \
    --basepath ../output/mdsine2/fixed_clustering \
    --fixed-clustering ../output/mdsine2/healthy-seed0/mcmc.pkl
python ../step_6_visualize_mdsine2.py \
    --chain  ../output/mdsine2/fixed_clustering/healthy/mcmc.pkl
    --output-basepath ../output/mdsine2/fixed_clustering/healthy/posterior
    --fixed-clustering 1

# UC cohort
# ---------
python ../step_5_infer_mdsine2.py \
    --input ../output/processed_data/gibson_uc_agg_taxa_filtered.pkl \
    --negbin $NEGBIN \
    --seed $SEED \
    --burnin $BURNIN \
    --n-samples $N_SAMPLES \
    --checkpoint $CHECKPOINT \
    --multiprocessing $MULTIPROCESSING \
    --basepath ../output/mdsine2/fixed_clustering \
    --fixed-clustering ../output/mdsine2/uc-seed0/mcmc.pkl
python ../step_6_visualize_mdsine2.py \
    --chain  ../output/mdsine2/fixed_clustering/uc/mcmc.pkl
    --output-basepath ../output/mdsine2/fixed_clustering/uc/posterior
    --fixed-clustering 1