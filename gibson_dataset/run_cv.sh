#!/bin/bash

NEGBIN = "../output/negbin/replicates/mcmc.pkl"
SEED = "0"
BURNIN = "5000"
N_SAMPLES = "15000"
CHECKPOINT = "100"
MULTIPROCESSING = "1"
DSET_BASEPATH = "../output/processed_data/cv"
CV_BASEPATH = "../output/mdsine2/cv"
HEALTHY_DATASET = "../output/processed_data/gibson_healthy_agg_taxa_filtered.pkl"
UC_DATASET = "../output/processed_data/gibson_uc_agg_taxa_filtered.pkl"

# Run healthy for each subject
# ----------------------------
# Regular inference
python ../run_cross_validation.py \
    --dataset $HEALTHY_DATASET \
    --cv-basepath $CV_BASEPATH \
    --dset-basepath $DSET_BASEPATH \
    --negbin $NEGBIN \
    --seed $SEED \
    --burnin $BURNIN \
    --n-samples $N_SAMPLES \
    --checkpoint $CHECKPOINT \
    --multiprocessing $MULTIPROCESSING \
    --leave-out-subject 2
python ../step_6_visualize_mdsine2.py \
    --chain $CV_BASEPATH/healthy-cv2/mcmc.pkl
    --output-basepath $CV_BASEPATH/healthy-cv2/posterior

python ../run_cross_validation.py \
    --dataset $HEALTHY_DATASET \
    --cv-basepath $CV_BASEPATH \
    --dset-basepath $DSET_BASEPATH \
    --negbin $NEGBIN \
    --seed $SEED \
    --burnin $BURNIN \
    --n-samples $N_SAMPLES \
    --checkpoint $CHECKPOINT \
    --multiprocessing $MULTIPROCESSING \
    --leave-out-subject 3
python ../step_6_visualize_mdsine2.py \
    --chain $CV_BASEPATH/healthy-cv3/mcmc.pkl
    --output-basepath $CV_BASEPATH/healthy-cv3/posterior

python ../run_cross_validation.py \
    --dataset $HEALTHY_DATASET \
    --cv-basepath $CV_BASEPATH \
    --dset-basepath $DSET_BASEPATH \
    --negbin $NEGBIN \
    --seed $SEED \
    --burnin $BURNIN \
    --n-samples $N_SAMPLES \
    --checkpoint $CHECKPOINT \
    --multiprocessing $MULTIPROCESSING \
    --leave-out-subject 4
python ../step_6_visualize_mdsine2.py \
    --chain $CV_BASEPATH/healthy-cv4/mcmc.pkl
    --output-basepath $CV_BASEPATH/healthy-cv4/posterior

python ../run_cross_validation.py \
    --dataset $HEALTHY_DATASET \
    --cv-basepath $CV_BASEPATH \
    --dset-basepath $DSET_BASEPATH \
    --negbin $NEGBIN \
    --seed $SEED \
    --burnin $BURNIN \
    --n-samples $N_SAMPLES \
    --checkpoint $CHECKPOINT \
    --multiprocessing $MULTIPROCESSING \
    --leave-out-subject 5
python ../step_6_visualize_mdsine2.py \
    --chain $CV_BASEPATH/healthy-cv5/mcmc.pkl
    --output-basepath $CV_BASEPATH/healthy-cv5/posterior

# Run uc for each subject
# -----------------------
python ../run_cross_validation.py \
    --dataset $uc_DATASET \
    --cv-basepath $CV_BASEPATH \
    --dset-basepath $DSET_BASEPATH \
    --negbin $NEGBIN \
    --seed $SEED \
    --burnin $BURNIN \
    --n-samples $N_SAMPLES \
    --checkpoint $CHECKPOINT \
    --multiprocessing $MULTIPROCESSING \
    --leave-out-subject 6
python ../step_6_visualize_mdsine2.py \
    --chain $CV_BASEPATH/uc-cv6/mcmc.pkl
    --output-basepath $CV_BASEPATH/uc-cv6/posterior

python ../run_cross_validation.py \
    --dataset $uc_DATASET \
    --cv-basepath $CV_BASEPATH \
    --dset-basepath $DSET_BASEPATH \
    --negbin $NEGBIN \
    --seed $SEED \
    --burnin $BURNIN \
    --n-samples $N_SAMPLES \
    --checkpoint $CHECKPOINT \
    --multiprocessing $MULTIPROCESSING \
    --leave-out-subject 7
python ../step_6_visualize_mdsine2.py \
    --chain $CV_BASEPATH/uc-cv7/mcmc.pkl
    --output-basepath $CV_BASEPATH/uc-cv7/posterior

python ../run_cross_validation.py \
    --dataset $uc_DATASET \
    --cv-basepath $CV_BASEPATH \
    --dset-basepath $DSET_BASEPATH \
    --negbin $NEGBIN \
    --seed $SEED \
    --burnin $BURNIN \
    --n-samples $N_SAMPLES \
    --checkpoint $CHECKPOINT \
    --multiprocessing $MULTIPROCESSING \
    --leave-out-subject 8
python ../step_6_visualize_mdsine2.py \
    --chain $CV_BASEPATH/uc-cv8/mcmc.pkl
    --output-basepath $CV_BASEPATH/uc-cv8/posterior

python ../run_cross_validation.py \
    --dataset $uc_DATASET \
    --cv-basepath $CV_BASEPATH \
    --dset-basepath $DSET_BASEPATH \
    --negbin $NEGBIN \
    --seed $SEED \
    --burnin $BURNIN \
    --n-samples $N_SAMPLES \
    --checkpoint $CHECKPOINT \
    --multiprocessing $MULTIPROCESSING \
    --leave-out-subject 9
python ../step_6_visualize_mdsine2.py \
    --chain $CV_BASEPATH/uc-cv9/mcmc.pkl
    --output-basepath $CV_BASEPATH/uc-cv9/posterior

python ../run_cross_validation.py \
    --dataset $uc_DATASET \
    --cv-basepath $CV_BASEPATH \
    --dset-basepath $DSET_BASEPATH \
    --negbin $NEGBIN \
    --seed $SEED \
    --burnin $BURNIN \
    --n-samples $N_SAMPLES \
    --checkpoint $CHECKPOINT \
    --multiprocessing $MULTIPROCESSING \
    --leave-out-subject 10
python ../step_6_visualize_mdsine2.py \
    --chain $CV_BASEPATH/uc-cv10/mcmc.pkl
    --output-basepath $CV_BASEPATH/uc-cv10/posterior
