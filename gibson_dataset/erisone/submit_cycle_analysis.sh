#!/bin/bash

# ErisOne parameters
# ------------------
# Path to MDSINE2_Paper code
MDSINE2_PAPER_CODE_PATH="/data/cctm/darpa_perturbation_mouse_study/MDSINE2_Paper"
# Conda environment
ENVIRONMENT_NAME="mdsine2"
# Queues, memory, and numpy of cpus
QUEUE="big-multi"
MEM="8000"
N_CPUS="4"
LSF_BASEPATH="lsf_files/cycles"

# Forward simulation parameters
# -----------------------------
SEED_SETTING="seed0-strong-sparse"
HEALTHY_CHAIN="output/mdsine2/healthy-${SEED_SETTING}/mcmc.pkl"
UC_CHAIN="output/mdsine2/uc-${SEED_SETTING}/mcmc.pkl"
HEALTHY_FIXED_CHAIN="output/mdsine2/fixed_clustering/healthy-${SEED_SETTING}/mcmc.pkl"
UC_FIXED_CHAIN="output/mdsine2/fixed_clustering/uc-${SEED_SETTING}/mcmc.pkl"
EIGEN_OUTDIR="output/postprocessing/eigenvalues/${SEED_SETTING}"
HEALTHY_UNFIXED_DIR="output/postprocessing/cycles/unfixed_clustering/healthy-${SEED_SETTING}"
HEALTHY_FIXED_DIR="output/postprocessing/cycles/fixed_clustering/healthy-${SEED_SETTING}"
UC_UNFIXED_DIR="output/postprocessing/cycles/unfixed_clustering/uc-${SEED_SETTING}"
UC_FIXED_DIR="output/postprocessing/cycles/fixed_clustering/uc-${SEED_SETTING}"


# Compute eigenvalues
# --------------------
python scripts/run_eigenvalue_analysis.py \
    --healthy_chain $HEALTHY_CHAIN \
    --uc_chain $UC_CHAIN \
    --outdir $EIGEN_OUTDIR \
    --environment-name $ENVIRONMENT_NAME \
    --code-basepath $MDSINE2_PAPER_CODE_PATH \
    --queue $QUEUE \
    --memory $MEM \
    --n-cpus $N_CPUS \
    --lsf-basepath $LSF_BASEPATH


# Compute cycles
# --------------------
python scripts/run_cycle_analysis.py \
    --chain $HEALTHY_CHAIN \
    --outdir $HEALTHY_UNFIXED_DIR \
    --environment-name $ENVIRONMENT_NAME \
    --code-basepath $MDSINE2_PAPER_CODE_PATH \
    --queue $QUEUE \
    --memory $MEM \
    --n-cpus $N_CPUS \
    --lsf-basepath $LSF_BASEPATH \
    --jobname "cycle-healthy-unfixed"

python scripts/run_cycle_analysis.py \
    --chain $HEALTHY_FIXED_CHAIN \
    --outdir $HEALTHY_FIXED_DIR \
    --environment-name $ENVIRONMENT_NAME \
    --code-basepath $MDSINE2_PAPER_CODE_PATH \
    --queue $QUEUE \
    --memory $MEM \
    --n-cpus $N_CPUS \
    --lsf-basepath $LSF_BASEPATH \
    --jobname "cycle-healthy-fixed"

python scripts/run_cycle_analysis.py \
    --chain $UC_CHAIN \
    --outdir $UC_UNFIXED_DIR \
    --environment-name $ENVIRONMENT_NAME \
    --code-basepath $MDSINE2_PAPER_CODE_PATH \
    --queue $QUEUE \
    --memory $MEM \
    --n-cpus $N_CPUS \
    --lsf-basepath $LSF_BASEPATH \
    --jobname "cycle-uc-unfixed"

python scripts/run_cycle_analysis.py \
    --chain $UC_FIXED_CHAIN \
    --outdir $UC_FIXED_DIR \
    --environment-name $ENVIRONMENT_NAME \
    --code-basepath $MDSINE2_PAPER_CODE_PATH \
    --queue $QUEUE \
    --memory $MEM \
    --n-cpus $N_CPUS \
    --lsf-basepath $LSF_BASEPATH \
    --jobname "cycle-uc-fixed"
