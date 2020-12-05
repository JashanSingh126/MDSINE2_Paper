#!/bin/bash

# Run healthy for each subject
# ----------------------------
python ../run_cross_validation.py \
    --dataset ../output/processed_data/gibson_healthy_agg_taxa_filtered.pkl \
    --cv-basepath ../output/mdsine2/cv \
    --dset-basepath ../output/processed_data/cv \
    --negbin ../output/negbin/replicates/mcmc.pkl \
    --seed 0 \
    --burnin 5000 \
    --n-samples 15000 \
    --checkpoint 100 \
    --multiprocessing 1 \
    --leave-out-subject 2
python ../step_6_visualize_mdsine2.py \
    --chain ../output/mdsine2/cv/healthy-cv2/mcmc.pkl
    --output-basepath ../output/mdsine2/cv/healthy-cv2/posterior

python ../run_cross_validation.py \
    --dataset ../output/processed_data/gibson_healthy_agg_taxa_filtered.pkl \
    --cv-basepath ../output/mdsine2/cv \
    --dset-basepath ../output/processed_data/cv \
    --negbin ../output/negbin/replicates/mcmc.pkl \
    --seed 0 \
    --burnin 5000 \
    --n-samples 15000 \
    --checkpoint 100 \
    --multiprocessing 1 \
    --leave-out-subject 3
python ../step_6_visualize_mdsine2.py \
    --chain ../output/mdsine2/cv/healthy-cv3/mcmc.pkl
    --output-basepath ../output/mdsine2/cv/healthy-cv3/posterior

python ../run_cross_validation.py \
    --dataset ../output/processed_data/gibson_healthy_agg_taxa_filtered.pkl \
    --cv-basepath ../output/mdsine2/cv \
    --dset-basepath ../output/processed_data/cv \
    --negbin ../output/negbin/replicates/mcmc.pkl \
    --seed 0 \
    --burnin 5000 \
    --n-samples 15000 \
    --checkpoint 100 \
    --multiprocessing 1 \
    --leave-out-subject 4
python ../step_6_visualize_mdsine2.py \
    --chain ../output/mdsine2/cv/healthy-cv4/mcmc.pkl
    --output-basepath ../output/mdsine2/cv/healthy-cv4/posterior

python ../run_cross_validation.py \
    --dataset ../output/processed_data/gibson_healthy_agg_taxa_filtered.pkl \
    --cv-basepath ../output/mdsine2/cv \
    --dset-basepath ../output/processed_data/cv \
    --negbin ../output/negbin/replicates/mcmc.pkl \
    --seed 0 \
    --burnin 5000 \
    --n-samples 15000 \
    --checkpoint 100 \
    --multiprocessing 1 \
    --leave-out-subject 5
python ../step_6_visualize_mdsine2.py \
    --chain ../output/mdsine2/cv/healthy-cv5/mcmc.pkl
    --output-basepath ../output/mdsine2/cv/healthy-cv5/posterior

# Run uc for each subject
# -----------------------
python ../run_cross_validation.py \
    --dataset ../output/processed_data/gibson_uc_agg_taxa_filtered.pkl \
    --cv-basepath ../output/mdsine2/cv \
    --dset-basepath ../output/processed_data/cv \
    --negbin ../output/negbin/replicates/mcmc.pkl \
    --seed 0 \
    --burnin 5000 \
    --n-samples 15000 \
    --checkpoint 100 \
    --multiprocessing 1 \
    --leave-out-subject 6
python ../step_6_visualize_mdsine2.py \
    --chain ../output/mdsine2/cv/uc-cv6/mcmc.pkl
    --output-basepath ../output/mdsine2/cv/uc-cv6/posterior

python ../run_cross_validation.py \
    --dataset ../output/processed_data/gibson_uc_agg_taxa_filtered.pkl \
    --cv-basepath ../output/mdsine2/cv \
    --dset-basepath ../output/processed_data/cv \
    --negbin ../output/negbin/replicates/mcmc.pkl \
    --seed 0 \
    --burnin 5000 \
    --n-samples 15000 \
    --checkpoint 100 \
    --multiprocessing 1 \
    --leave-out-subject 7
python ../step_6_visualize_mdsine2.py \
    --chain ../output/mdsine2/cv/uc-cv7/mcmc.pkl
    --output-basepath ../output/mdsine2/cv/uc-cv7/posterior

python ../run_cross_validation.py \
    --dataset ../output/processed_data/gibson_uc_agg_taxa_filtered.pkl \
    --cv-basepath ../output/mdsine2/cv \
    --dset-basepath ../output/processed_data/cv \
    --negbin ../output/negbin/replicates/mcmc.pkl \
    --seed 0 \
    --burnin 5000 \
    --n-samples 15000 \
    --checkpoint 100 \
    --multiprocessing 1 \
    --leave-out-subject 8
python ../step_6_visualize_mdsine2.py \
    --chain ../output/mdsine2/cv/uc-cv8/mcmc.pkl
    --output-basepath ../output/mdsine2/cv/uc-cv8/posterior

python ../run_cross_validation.py \
    --dataset ../output/processed_data/gibson_uc_agg_taxa_filtered.pkl \
    --cv-basepath ../output/mdsine2/cv \
    --dset-basepath ../output/processed_data/cv \
    --negbin ../output/negbin/replicates/mcmc.pkl \
    --seed 0 \
    --burnin 5000 \
    --n-samples 15000 \
    --checkpoint 100 \
    --multiprocessing 1 \
    --leave-out-subject 9
python ../step_6_visualize_mdsine2.py \
    --chain ../output/mdsine2/cv/uc-cv9/mcmc.pkl
    --output-basepath ../output/mdsine2/cv/uc-cv9/posterior

python ../run_cross_validation.py \
    --dataset ../output/processed_data/gibson_uc_agg_taxa_filtered.pkl \
    --cv-basepath ../output/mdsine2/cv \
    --dset-basepath ../output/processed_data/cv \
    --negbin ../output/negbin/replicates/mcmc.pkl \
    --seed 0 \
    --burnin 5000 \
    --n-samples 15000 \
    --checkpoint 100 \
    --multiprocessing 1 \
    --leave-out-subject 10
python ../step_6_visualize_mdsine2.py \
    --chain ../output/mdsine2/cv/uc-cv10/mcmc.pkl
    --output-basepath ../output/mdsine2/cv/uc-cv10/posterior
