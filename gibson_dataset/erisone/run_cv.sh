#!/bin/bash

# Run healthy for each subject
# ----------------------------
python cv.py \
    --dataset ../../output/processed_data/gibson_healthy_agg_taxa_filtered.pkl \
    --cv-basepath ../../output/mdsine2/cv \
    --dset-basepath ../../output/processed_data/cv \
    --negbin-run ../../output/negbin/replicates/mcmc.pkl \
    --seed 0 \
    --burnin 5000 \
    --n-samples 15000 \
    --checkpoint 100 \
    --multiprocessing 1 \
    --environment-name mdsine2_403 \
    --code-basepath /data/cctm/darpa_perturbation_mouse_study/MDSINE2 \
    --leave-out-subject 2

python cv.py \
    --dataset ../../output/processed_data/gibson_healthy_agg_taxa_filtered.pkl \
    --cv-basepath ../../output/mdsine2/cv \
    --dset-basepath ../../output/processed_data/cv \
    --negbin-run ../../output/negbin/replicates/mcmc.pkl \
    --seed 0 \
    --burnin 5000 \
    --n-samples 15000 \
    --checkpoint 100 \
    --multiprocessing 1 \
    --environment-name mdsine2_403 \
    --code-basepath /data/cctm/darpa_perturbation_mouse_study/MDSINE2 \
    --leave-out-subject 3

python cv.py \
    --dataset ../../output/processed_data/gibson_healthy_agg_taxa_filtered.pkl \
    --cv-basepath ../../output/mdsine2/cv \
    --dset-basepath ../../output/processed_data/cv \
    --negbin-run ../../output/negbin/replicates/mcmc.pkl \
    --seed 0 \
    --burnin 5000 \
    --n-samples 15000 \
    --checkpoint 100 \
    --multiprocessing 1 \
    --environment-name mdsine2_403 \
    --code-basepath /data/cctm/darpa_perturbation_mouse_study/MDSINE2 \
    --leave-out-subject 4

python cv.py \
    --dataset ../../output/processed_data/gibson_healthy_agg_taxa_filtered.pkl \
    --cv-basepath ../../output/mdsine2/cv \
    --dset-basepath ../../output/processed_data/cv \
    --negbin-run ../../output/negbin/replicates/mcmc.pkl \
    --seed 0 \
    --burnin 5000 \
    --n-samples 15000 \
    --checkpoint 100 \
    --multiprocessing 1 \
    --environment-name mdsine2_403 \
    --code-basepath /data/cctm/darpa_perturbation_mouse_study/MDSINE2 \
    --leave-out-subject 5

# Run uc for each subject
# -----------------------
python cv.py \
    --dataset ../../output/processed_data/gibson_uc_agg_taxa_filtered.pkl \
    --cv-basepath ../../output/mdsine2/cv \
    --dset-basepath ../../output/processed_data/cv \
    --negbin-run ../../output/negbin/replicates/mcmc.pkl \
    --seed 0 \
    --burnin 5000 \
    --n-samples 15000 \
    --checkpoint 100 \
    --multiprocessing 1 \
    --environment-name mdsine2_403 \
    --code-basepath /data/cctm/darpa_perturbation_mouse_study/MDSINE2 \
    --leave-out-subject 6

python cv.py \
    --dataset ../../output/processed_data/gibson_uc_agg_taxa_filtered.pkl \
    --cv-basepath ../../output/mdsine2/cv \
    --dset-basepath ../../output/processed_data/cv \
    --negbin-run ../../output/negbin/replicates/mcmc.pkl \
    --seed 0 \
    --burnin 5000 \
    --n-samples 15000 \
    --checkpoint 100 \
    --multiprocessing 1 \
    --environment-name mdsine2_403 \
    --code-basepath /data/cctm/darpa_perturbation_mouse_study/MDSINE2 \
    --leave-out-subject 7

python cv.py \
    --dataset ../../output/processed_data/gibson_uc_agg_taxa_filtered.pkl \
    --cv-basepath ../../output/mdsine2/cv \
    --dset-basepath ../../output/processed_data/cv \
    --negbin-run ../../output/negbin/replicates/mcmc.pkl \
    --seed 0 \
    --burnin 5000 \
    --n-samples 15000 \
    --checkpoint 100 \
    --multiprocessing 1 \
    --environment-name mdsine2_403 \
    --code-basepath /data/cctm/darpa_perturbation_mouse_study/MDSINE2 \
    --leave-out-subject 8

python cv.py \
    --dataset ../../output/processed_data/gibson_uc_agg_taxa_filtered.pkl \
    --cv-basepath ../../output/mdsine2/cv \
    --dset-basepath ../../output/processed_data/cv \
    --negbin-run ../../output/negbin/replicates/mcmc.pkl \
    --seed 0 \
    --burnin 5000 \
    --n-samples 15000 \
    --checkpoint 100 \
    --multiprocessing 1 \
    --environment-name mdsine2_403 \
    --code-basepath /data/cctm/darpa_perturbation_mouse_study/MDSINE2 \
    --leave-out-subject 9

python cv.py \
    --dataset ../../output/processed_data/gibson_uc_agg_taxa_filtered.pkl \
    --cv-basepath ../../output/mdsine2/cv \
    --dset-basepath ../../output/processed_data/cv \
    --negbin-run ../../output/negbin/replicates/mcmc.pkl \
    --seed 0 \
    --burnin 5000 \
    --n-samples 15000 \
    --checkpoint 100 \
    --multiprocessing 1 \
    --environment-name mdsine2_403 \
    --code-basepath /data/cctm/darpa_perturbation_mouse_study/MDSINE2 \
    --leave-out-subject 10