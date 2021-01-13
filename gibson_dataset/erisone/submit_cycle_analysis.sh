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
N_CPUS="1"
LSF_BASEPATH="lsf_files/cycles"

# Forward simulation parameters
# -----------------------------
HEALTHY_CHAIN="output/mdsine2/healthy-seed0/mcmc.pkl"
UC_CHAIN="output/mdsine2/uc-seed0/mcmc.pkl"
HEALTHY_FIXED_CHAIN="output/mdsine2/fixed_clustering/healthy/mcmc.pkl"
UC_FIXED_CHAIN="output/mdsine2/fixed_clustering/uc/mcmc.pkl"

EIGEN_OUTDIR="../output/postprocessing/eigenvalues"
HEALTHY_UNFIXED_DIR="output/postprocessing/cycles/unfixed_clustering/healthy"
HEALTHY_FIXED_DIR="output/postprocessing/cycles/fixed_clustering/healthy"
UC_UNFIXED_DIR="output/postprocessing/cycles/unfixed_clustering/uc"
UC_FIXED_DIR="output/postprocessing/cycles/fixed_clustering/uc"


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
