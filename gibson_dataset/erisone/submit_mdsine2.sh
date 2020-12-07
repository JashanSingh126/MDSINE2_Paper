#!/bin/bash

# ErisOne parameters
# ------------------
# Path to MDSINE2_Paper code
MDSINE2_PAPER_CODE_PATH="/data/cctm/darpa_perturbation_mouse_study/MDSINE2_Paper"
# Conda environment
ENVIRONMENT_NAME="mdsine2"
# Queues, memory, and numpy of cpus
QUEUE="vlong"
MEM="8000"
N_CPUS="1"

# NOTE: THESE PATHS MUST BE RELATIVE TO `MDSINE2_PAPER_CODE_PATH`
NEGBIN="output/negbin/replicates/mcmc.pkl"
BURNIN="5000"
N_SAMPLES="15000"
CHECKPOINT="100"
MULTIPROCESSING="0"

HEALTHY_DATASET="output/processed_data/gibson_healthy_agg_taxa_filtered.pkl"
UC_DATASET="output/processed_data/gibson_uc_agg_taxa_filtered.pkl"
BASEPATH="output/mdsine2"
FIXED_BASEPATH="output/mdsine2/fixed_clustering"
INTERACTION_IND_PRIOR="strong-sparse"
PERTURBATION_IND_PRIOR="strong-sparse"


# Healthy
# -------
python scripts/run_model.py \
    --dataset $HEALTHY_DATASET \
    --negbin $NEGBIN \
    --seed 0 \
    --burnin $BURNIN \
    --n-samples $N_SAMPLES \
    --checkpoint $CHECKPOINT \
    --multiprocessing $MULTIPROCESSING \
    --rename-study "healthy-seed0" \
    --output-basepath $BASEPATH \
    --fixed-output-basepath $FIXED_BASEPATH \
    --environment-name $ENVIRONMENT_NAME \
    --code-basepath $MDSINE2_PAPER_CODE_PATH \
    --queue $QUEUE \
    --memory $MEM \
    --n-cpus $N_CPUS \
    --interaction-ind-prior $INTERACTION_IND_PRIOR \
    --perturbation-ind-prior $PERTURBATION_IND_PRIOR

python scripts/run_model.py \
    --dataset $HEALTHY_DATASET \
    --negbin $NEGBIN \
    --seed 1 \
    --burnin $BURNIN \
    --n-samples $N_SAMPLES \
    --checkpoint $CHECKPOINT \
    --multiprocessing $MULTIPROCESSING \
    --rename-study "healthy-seed1" \
    --output-basepath $BASEPATH \
    --fixed-output-basepath $FIXED_BASEPATH \
    --environment-name $ENVIRONMENT_NAME \
    --code-basepath $MDSINE2_PAPER_CODE_PATH \
    --queue $QUEUE \
    --memory $MEM \
    --n-cpus $N_CPUS \
    --interaction-ind-prior $INTERACTION_IND_PRIOR \
    --perturbation-ind-prior $PERTURBATION_IND_PRIOR

# UC Cohort
# ---------
python scripts/run_model.py \
    --dataset $UC_DATASET \
    --negbin $NEGBIN \
    --seed 0 \
    --burnin $BURNIN \
    --n-samples $N_SAMPLES \
    --checkpoint $CHECKPOINT \
    --multiprocessing $MULTIPROCESSING \
    --rename-study "uc-seed0" \
    --output-basepath $BASEPATH \
    --fixed-output-basepath $FIXED_BASEPATH \
    --environment-name $ENVIRONMENT_NAME \
    --code-basepath $MDSINE2_PAPER_CODE_PATH \
    --queue $QUEUE \
    --memory $MEM \
    --n-cpus $N_CPUS \
    --interaction-ind-prior $INTERACTION_IND_PRIOR \
    --perturbation-ind-prior $PERTURBATION_IND_PRIOR

python scripts/run_model.py \
    --dataset $UC_DATASET \
    --negbin $NEGBIN \
    --seed 1 \
    --burnin $BURNIN \
    --n-samples $N_SAMPLES \
    --checkpoint $CHECKPOINT \
    --multiprocessing $MULTIPROCESSING \
    --rename-study "uc-seed1" \
    --output-basepath $BASEPATH \
    --fixed-output-basepath $FIXED_BASEPATH \
    --environment-name $ENVIRONMENT_NAME \
    --code-basepath $MDSINE2_PAPER_CODE_PATH \
    --queue $QUEUE \
    --memory $MEM \
    --n-cpus $N_CPUS \
    --interaction-ind-prior $INTERACTION_IND_PRIOR \
    --perturbation-ind-prior $PERTURBATION_IND_PRIOR