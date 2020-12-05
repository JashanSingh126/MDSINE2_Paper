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
    --rename-study healthy-seed0 \
    --basepath ../output/mdsine2
python ../step_6_visualize_mdsine2.py \
    --chain  ../output/mdsine2/healthy-seed0/mcmc.pkl
    --output-basepath ../output/mdsine2/healthy-seed0/posterior

python ../step_5_infer_mdsine2.py \
    --input ../output/processed_data/gibson_healthy_agg_taxa_filtered.pkl \
    --negbin ../output/negbin/replicates/mcmc.pkl \
    --seed 1 \
    --burnin 5000 \
    --n-samples 15000 \
    --checkpoint 100 \
    --multiprocessing 1 \
    --rename-study healthy-seed1 \
    --basepath ../output/mdsine2
python ../step_6_visualize_mdsine2.py \
    --chain  ../output/mdsine2/healthy-seed1/mcmc.pkl
    --output-basepath ../output/mdsine2/healthy-seed1/posterior

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
    --rename-study uc-seed0 \
    --basepath ../output/mdsine2
python ../step_6_visualize_mdsine2.py \
    --chain  ../output/mdsine2/uc-seed0/mcmc.pkl
    --output-basepath ../output/mdsine2/uc-seed0/posterior

python ../step_5_infer_mdsine2.py \
    --input ../output/processed_data/gibson_uc_agg_taxa_filtered.pkl \
    --negbin ../output/negbin/replicates/mcmc.pkl \
    --seed 1 \
    --burnin 5000 \
    --n-samples 15000 \
    --checkpoint 100 \
    --multiprocessing 1 \
    --rename-study uc-seed0 \
    --basepath ../output/mdsine2
python ../step_6_visualize_mdsine2.py \
    --chain  ../output/mdsine2/uc-seed1/mcmc.pkl
    --output-basepath ../output/mdsine2/uc-seed1/posterior
