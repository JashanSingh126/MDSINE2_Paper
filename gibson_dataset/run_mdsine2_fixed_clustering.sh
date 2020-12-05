#!/bin/bash

# Healthy cohort
# --------------
python ../step_5_infer_mdsine2.py \
    --input ../output/processed_data/gibson_healthy_agg_taxa_filtered.pkl \
    --negbin ../output/negbin/replicates/mcmc.pkl \
    --seed 0 \
    --burnin 5000 \
    --n-samples 15000 \
    --checkpoint 100 \
    --multiprocessing 1 \
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
    --negbin ../output/negbin/replicates/mcmc.pkl \
    --seed 0 \
    --burnin 5000 \
    --n-samples 15000 \
    --checkpoint 100 \
    --multiprocessing 1 \
    --basepath ../output/mdsine2/fixed_clustering \
    --fixed-clustering ../output/mdsine2/uc-seed0/mcmc.pkl
python ../step_6_visualize_mdsine2.py \
    --chain  ../output/mdsine2/fixed_clustering/uc/mcmc.pkl
    --output-basepath ../output/mdsine2/fixed_clustering/uc/posterior
    --fixed-clustering 1