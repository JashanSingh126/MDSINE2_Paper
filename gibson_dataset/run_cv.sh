#!/bin/bash

# Run healthy for each subject
# ----------------------------
python ../run_cross_validation.py \
    --dataset ../output/processed_data/gibson_healthy_agg_taxa_filtered.pkl \
    --cv-basepath ../output/mdsine2/cv \
    --dset-basepath ../output/processed_data/cv \
    --negbin-run ../output/negbin/replicates/mcmc.pkl \
    --seed 0 \
    --burnin 20 \
    --n-samples 40 \
    --checkpoint 20 \
    --multiprocessing 1 \
    --leave-out-subject 2

python ../run_cross_validation.py \
    --dataset ../output/processed_data/gibson_healthy_agg_taxa_filtered.pkl \
    --cv-basepath ../output/mdsine2/cv \
    --dset-basepath ../output/processed_data/cv \
    --negbin-run ../output/negbin/replicates/mcmc.pkl \
    --seed 0 \
    --burnin 5000 \
    --n-samples 15000 \
    --checkpoint 100 \
    --multiprocessing 1 \
    --leave-out-subject 3

python ../run_cross_validation.py \
    --dataset ../output/processed_data/gibson_healthy_agg_taxa_filtered.pkl \
    --cv-basepath ../output/mdsine2/cv \
    --dset-basepath ../output/processed_data/cv \
    --negbin-run ../output/negbin/replicates/mcmc.pkl \
    --seed 0 \
    --burnin 5000 \
    --n-samples 15000 \
    --checkpoint 100 \
    --multiprocessing 1 \
    --leave-out-subject 4

python ../run_cross_validation.py \
    --dataset ../output/processed_data/gibson_healthy_agg_taxa_filtered.pkl \
    --cv-basepath ../output/mdsine2/cv \
    --dset-basepath ../output/processed_data/cv \
    --negbin-run ../output/negbin/replicates/mcmc.pkl \
    --seed 0 \
    --burnin 5000 \
    --n-samples 15000 \
    --checkpoint 100 \
    --multiprocessing 1 \
    --leave-out-subject 5

# Run uc for each subject
# -----------------------
python ../run_cross_validation.py \
    --dataset ../output/processed_data/gibson_uc_agg_taxa_filtered.pkl \
    --cv-basepath ../output/mdsine2/cv \
    --dset-basepath ../output/processed_data/cv \
    --negbin-run ../output/negbin/replicates/mcmc.pkl \
    --seed 0 \
    --burnin 5000 \
    --n-samples 15000 \
    --checkpoint 100 \
    --multiprocessing 1 \
    --leave-out-subject 6

python ../run_cross_validation.py \
    --dataset ../output/processed_data/gibson_uc_agg_taxa_filtered.pkl \
    --cv-basepath ../output/mdsine2/cv \
    --dset-basepath ../output/processed_data/cv \
    --negbin-run ../output/negbin/replicates/mcmc.pkl \
    --seed 0 \
    --burnin 5000 \
    --n-samples 15000 \
    --checkpoint 100 \
    --multiprocessing 1 \
    --leave-out-subject 7

python ../run_cross_validation.py \
    --dataset ../output/processed_data/gibson_uc_agg_taxa_filtered.pkl \
    --cv-basepath ../output/mdsine2/cv \
    --dset-basepath ../output/processed_data/cv \
    --negbin-run ../output/negbin/replicates/mcmc.pkl \
    --seed 0 \
    --burnin 5000 \
    --n-samples 15000 \
    --checkpoint 100 \
    --multiprocessing 1 \
    --leave-out-subject 8

python ../run_cross_validation.py \
    --dataset ../output/processed_data/gibson_uc_agg_taxa_filtered.pkl \
    --cv-basepath ../output/mdsine2/cv \
    --dset-basepath ../output/processed_data/cv \
    --negbin-run ../output/negbin/replicates/mcmc.pkl \
    --seed 0 \
    --burnin 5000 \
    --n-samples 15000 \
    --checkpoint 100 \
    --multiprocessing 1 \
    --leave-out-subject 9

python ../run_cross_validation.py \
    --dataset ../output/processed_data/gibson_uc_agg_taxa_filtered.pkl \
    --cv-basepath ../output/mdsine2/cv \
    --dset-basepath ../output/processed_data/cv \
    --negbin-run ../output/negbin/replicates/mcmc.pkl \
    --seed 0 \
    --burnin 5000 \
    --n-samples 15000 \
    --checkpoint 100 \
    --multiprocessing 1 \
    --leave-out-subject 10

