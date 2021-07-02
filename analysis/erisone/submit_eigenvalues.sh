#!/bin/bash

# ErisOne parameters
# ------------------
# Path to MDSINE2_Paper code
MDSINE2_PAPER_CODE_PATH=${MDSINE2_PAPER_CODE_PATH:-"/data/cctm/darpa_perturbation_mouse_study/MDSINE2_Paper"}
# Conda environment
ENVIRONMENT_NAME="mdsine2"
# Queues, memory, and numpy of cpus
QUEUE="big-multi"
MEM="8000"
N_CPUS="4"
LSF_BASEPATH="lsf_files/eigenvalues"

# Forward simulation parameters
# -----------------------------
SEED_SETTING="seed0-strong-sparse"
HEALTHY_MCMC="output/mdsine2/healthy-${SEED_SETTING}/mcmc.pkl"
UC_MCMC="output/mdsine2/uc-${SEED_SETTING}/mcmc.pkl"
EIGEN_OUTDIR="output/postprocessing/eigenvalues/${SEED_SETTING}"


# Compute eigenvalues
# --------------------
python scripts/run_eigenvalue_analysis.py \
    --healthy_chain $HEALTHY_MCMC \
    --uc_chain $UC_MCMC \
    --outdir $EIGEN_OUTDIR \
    --environment-name $ENVIRONMENT_NAME \
    --code-basepath $MDSINE2_PAPER_CODE_PATH \
    --queue $QUEUE \
    --memory $MEM \
    --n-cpus $N_CPUS \
    --lsf-basepath $LSF_BASEPATH
