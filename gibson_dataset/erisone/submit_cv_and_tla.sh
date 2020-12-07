#!/bin/bash

# ErisOne parameters
# ------------------
# Path to MDSINE2_Paper code
MDSINE2_PAPER_CODE_PATH="/data/cctm/darpa_perturbation_mouse_study/MDSINE2_paper"
# Conda environment
ENVIRONMENT_NAME="mdsine2_403"
# Queues, memory (_MEM), and nodes (_N) for forward simulation (TLA) and cross-validation (CV)
TLA_QUEUE="medium"
TLA_MEM="8000"
TLA_N="1"

CV_QUEUE="big-multi"
CV_MEM="10000"
CV_N="4"


# Running parameters
# NOTE: THESE PATHS MUST BE RELATIVE TO `MDSINE2_PAPER_CODE_PATH`
NEGBIN="output/negbin/replicates/mcmc.pkl"
SEED="0"
BURNIN="5000"
N_SAMPLES="15000"
CHECKPOINT="100"
MULTIPROCESSING="1"
DSET_BASEPATH="output/processed_data/cv"
CV_BASEPATH="output/mdsine2/cv"
HEALTHY_DATASET="output/processed_data/gibson_healthy_agg_taxa_filtered.pkl"
HEALTHY_DATASET_CURR_PATH="../../output/processed_data/gibson_healthy_agg_taxa_filtered.pkl"
UC_DATASET="output/processed_data/gibson_uc_agg_taxa_filtered.pkl"
UC_DATASET_CURR_PATH="../../output/processed_data/gibson_uc_agg_taxa_filtered.pkl"
MAX_TLA="8"

# Run healthy for each subject
# ----------------------------
python scripts/run_cv.py \
    --dataset $HEALTHY_DATASET \
    --dataset-curr-path $HEALTHY_DATASET_CURR_PATH \
    --cv-basepath $CV_BASEPATH \
    --dset-basepath $DSET_BASEPATH \
    --negbin $NEGBIN \
    --seed $SEED \
    --burnin $BURNIN \
    --n-samples $N_SAMPLES \
    --checkpoint $CHECKPOINT \
    --multiprocessing $MULTIPROCESSING \
    --environment-name $ENVIRONMENT_NAME \
    --code-basepath $MDSINE2_PAPER_CODE_PATH \
    --leave-out-subject 2 \
    --max-tla $MAX_TLA \
    --cv-queue $CV_QUEUE \
    --cv-memory $CV_MEM \
    --cv-n-cpus $CV_N \
    --tla-queue $TLA_QUEUE \
    --tla-n-cpus $TLA_N \
    --tla-memory $TLA_MEM

sleep 100 

# python scripts/run_cv.py \
#     --dataset $HEALTHY_DATASET \
#     --cv-basepath $CV_BASEPATH \
#     --dset-basepath $DSET_BASEPATH \
#     --negbin $NEGBIN \
#     --seed $SEED \
#     --burnin $BURNIN \
#     --n-samples $N_SAMPLES \
#     --checkpoint $CHECKPOINT \
#     --multiprocessing $MULTIPROCESSING \
#     --environment-name $ENVIRONMENT_NAME \
#     --code-basepath $MDSINE2_PAPER_CODE_PATH \
#     --leave-out-subject 3 \
#     --max-tla $MAX_TLA \
#     --cv-queue $CV_QUEUE \
#     --cv-memory $CV_MEM \
#     --cv-n-cpus $CV_N \
#     --tla-queue $TLA_QUEUE \
#     --tla-n-cpus $TLA_N \
#     --tla-memory $TLA_MEM

# python scripts/run_cv.py \
#     --dataset $HEALTHY_DATASET \
#     --cv-basepath $CV_BASEPATH \
#     --dset-basepath $DSET_BASEPATH \
#     --negbin $NEGBIN \
#     --seed $SEED \
#     --burnin $BURNIN \
#     --n-samples $N_SAMPLES \
#     --checkpoint $CHECKPOINT \
#     --multiprocessing $MULTIPROCESSING \
#     --environment-name $ENVIRONMENT_NAME \
#     --code-basepath $MDSINE2_PAPER_CODE_PATH \
#     --leave-out-subject 4 \
#     --max-tla $MAX_TLA \
#     --cv-queue $CV_QUEUE \
#     --cv-memory $CV_MEM \
#     --cv-n-cpus $CV_N \
#     --tla-queue $TLA_QUEUE \
#     --tla-n-cpus $TLA_N \
#     --tla-memory $TLA_MEM

# python scripts/run_cv.py \
#     --dataset $HEALTHY_DATASET \
#     --cv-basepath $CV_BASEPATH \
#     --dset-basepath $DSET_BASEPATH \
#     --negbin $NEGBIN \
#     --seed $SEED \
#     --burnin $BURNIN \
#     --n-samples $N_SAMPLES \
#     --checkpoint $CHECKPOINT \
#     --multiprocessing $MULTIPROCESSING \
#     --environment-name $ENVIRONMENT_NAME \
#     --code-basepath $MDSINE2_PAPER_CODE_PATH \
#     --leave-out-subject 5 \
#     --max-tla $MAX_TLA \
#     --cv-queue $CV_QUEUE \
#     --cv-memory $CV_MEM \
#     --cv-n-cpus $CV_N \
#     --tla-queue $TLA_QUEUE \
#     --tla-n-cpus $TLA_N \
#     --tla-memory $TLA_MEM

# # Run uc for each subject
# # -----------------------
# python scripts/run_cv.py \
#     --dataset $UC_DATASET \
#     --cv-basepath $CV_BASEPATH \
#     --dset-basepath $DSET_BASEPATH \
#     --negbin $NEGBIN \
#     --seed $SEED \
#     --burnin $BURNIN \
#     --n-samples $N_SAMPLES \
#     --checkpoint $CHECKPOINT \
#     --multiprocessing $MULTIPROCESSING \
#     --environment-name $ENVIRONMENT_NAME \
#     --code-basepath $MDSINE2_PAPER_CODE_PATH \
#     --leave-out-subject 6 \
#     --max-tla $MAX_TLA \
#     --cv-queue $CV_QUEUE \
#     --cv-memory $CV_MEM \
#     --cv-n-cpus $CV_N \
#     --tla-queue $TLA_QUEUE \
#     --tla-n-cpus $TLA_N \
#     --tla-memory $TLA_MEM

# python scripts/run_cv.py \
#     --dataset $UC_DATASET \
#     --cv-basepath $CV_BASEPATH \
#     --dset-basepath $DSET_BASEPATH \
#     --negbin $NEGBIN \
#     --seed $SEED \
#     --burnin $BURNIN \
#     --n-samples $N_SAMPLES \
#     --checkpoint $CHECKPOINT \
#     --multiprocessing $MULTIPROCESSING \
#     --environment-name $ENVIRONMENT_NAME \
#     --code-basepath $MDSINE2_PAPER_CODE_PATH \
#     --leave-out-subject 7 \
#     --max-tla $MAX_TLA \
#     --cv-queue $CV_QUEUE \
#     --cv-memory $CV_MEM \
#     --cv-n-cpus $CV_N \
#     --tla-queue $TLA_QUEUE \
#     --tla-n-cpus $TLA_N \
#     --tla-memory $TLA_MEM

# python scripts/run_cv.py \
#     --dataset $UC_DATASET \
#     --cv-basepath $CV_BASEPATH \
#     --dset-basepath $DSET_BASEPATH \
#     --negbin $NEGBIN \
#     --seed $SEED \
#     --burnin $BURNIN \
#     --n-samples $N_SAMPLES \
#     --checkpoint $CHECKPOINT \
#     --multiprocessing $MULTIPROCESSING \
#     --environment-name $ENVIRONMENT_NAME \
#     --code-basepath $MDSINE2_PAPER_CODE_PATH \
#     --leave-out-subject 8 \
#     --max-tla $MAX_TLA \
#     --cv-queue $CV_QUEUE \
#     --cv-memory $CV_MEM \
#     --cv-n-cpus $CV_N \
#     --tla-queue $TLA_QUEUE \
#     --tla-n-cpus $TLA_N \
#     --tla-memory $TLA_MEM

# python scripts/run_cv.py \
#     --dataset $UC_DATASET \
#     --cv-basepath $CV_BASEPATH \
#     --dset-basepath $DSET_BASEPATH \
#     --negbin $NEGBIN \
#     --seed $SEED \
#     --burnin $BURNIN \
#     --n-samples $N_SAMPLES \
#     --checkpoint $CHECKPOINT \
#     --multiprocessing $MULTIPROCESSING \
#     --environment-name $ENVIRONMENT_NAME \
#     --code-basepath $MDSINE2_PAPER_CODE_PATH \
#     --leave-out-subject 9 \
#     --max-tla $MAX_TLA \
#     --cv-queue $CV_QUEUE \
#     --cv-memory $CV_MEM \
#     --cv-n-cpus $CV_N \
#     --tla-queue $TLA_QUEUE \
#     --tla-n-cpus $TLA_N \
#     --tla-memory $TLA_MEM

# python scripts/run_cv.py \
#     --dataset $UC_DATASET \
#     --cv-basepath $CV_BASEPATH \
#     --dset-basepath $DSET_BASEPATH \
#     --negbin $NEGBIN \
#     --seed $SEED \
#     --burnin $BURNIN \
#     --n-samples $N_SAMPLES \
#     --checkpoint $CHECKPOINT \
#     --multiprocessing $MULTIPROCESSING \
#     --environment-name $ENVIRONMENT_NAME \
#     --code-basepath $MDSINE2_PAPER_CODE_PATH \
#     --leave-out-subject 10 \
#     --max-tla $MAX_TLA \
#     --cv-queue $CV_QUEUE \
#     --cv-memory $CV_MEM \
#     --cv-n-cpus $CV_N \
#     --tla-queue $TLA_QUEUE \
#     --tla-n-cpus $TLA_N \
#     --tla-memory $TLA_MEM