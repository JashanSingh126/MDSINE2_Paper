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
LSF_BASEPATH="lsf_files/cycles"

# Forward simulation parameters
# -----------------------------
SEED_SETTING="seed0-strong-sparse"
HEALTHY_MCMC="output/mdsine2/healthy-${SEED_SETTING}/mcmc.pkl"
UC_MCMC="output/mdsine2/uc-${SEED_SETTING}/mcmc.pkl"
HEALTHY_FIXED_MCMC="output/mdsine2/fixed_clustering/healthy-${SEED_SETTING}/mcmc.pkl"
UC_FIXED_MCMC="output/mdsine2/fixed_clustering/uc-${SEED_SETTING}/mcmc.pkl"

CYCLE_HEALTHY_UNFIXED_DIR="output/postprocessing/cycles/unfixed_clustering/healthy-${SEED_SETTING}"
CYCLE_HEALTHY_FIXED_DIR="output/postprocessing/cycles/fixed_clustering/healthy-${SEED_SETTING}"
CYCLE_UC_UNFIXED_DIR="output/postprocessing/cycles/unfixed_clustering/uc-${SEED_SETTING}"
CYCLE_UC_FIXED_DIR="output/postprocessing/cycles/fixed_clustering/uc-${SEED_SETTING}"

CHAIN_HEALTHY_UNFIXED_DIR="output/postprocessing/chains/unfixed_clustering/healthy-${SEED_SETTING}"
CHAIN_HEALTHY_FIXED_DIR="output/postprocessing/chains/fixed_clustering/healthy-${SEED_SETTING}"
CHAIN_UC_UNFIXED_DIR="output/postprocessing/chains/unfixed_clustering/uc-${SEED_SETTING}"
CHAIN_UC_FIXED_DIR="output/postprocessing/chains/fixed_clustering/uc-${SEED_SETTING}"

CYCLE_PATH_LEN=3
CHAIN_PATH_LEN=1


# Compute cycles
# --------------------
python scripts/run_cycle_analysis.py \
    --chain $HEALTHY_MCMC \
    --path_len $CYCLE_PATH_LEN \
    --outdir $CYCLE_HEALTHY_UNFIXED_DIR \
    --environment-name $ENVIRONMENT_NAME \
    --code-basepath $MDSINE2_PAPER_CODE_PATH \
    --queue $QUEUE \
    --memory $MEM \
    --n-cpus $N_CPUS \
    --lsf-basepath $LSF_BASEPATH \
    --jobname "cycle-healthy-unfixed"

python scripts/run_cycle_analysis.py \
    --chain $HEALTHY_FIXED_MCMC \
    --path_len $CYCLE_PATH_LEN \
    --outdir $CYCLE_HEALTHY_FIXED_DIR \
    --environment-name $ENVIRONMENT_NAME \
    --code-basepath $MDSINE2_PAPER_CODE_PATH \
    --queue $QUEUE \
    --memory $MEM \
    --n-cpus $N_CPUS \
    --lsf-basepath $LSF_BASEPATH \
    --jobname "cycle-healthy-fixed"

python scripts/run_cycle_analysis.py \
    --chain $UC_MCMC \
    --path_len $CYCLE_PATH_LEN \
    --outdir $CYCLE_UC_UNFIXED_DIR \
    --environment-name $ENVIRONMENT_NAME \
    --code-basepath $MDSINE2_PAPER_CODE_PATH \
    --queue $QUEUE \
    --memory $MEM \
    --n-cpus $N_CPUS \
    --lsf-basepath $LSF_BASEPATH \
    --jobname "cycle-uc-unfixed"

python scripts/run_cycle_analysis.py \
    --chain $UC_FIXED_MCMC \
    --path_len $CYCLE_PATH_LEN \
    --outdir $CYCLE_UC_FIXED_DIR \
    --environment-name $ENVIRONMENT_NAME \
    --code-basepath $MDSINE2_PAPER_CODE_PATH \
    --queue $QUEUE \
    --memory $MEM \
    --n-cpus $N_CPUS \
    --lsf-basepath $LSF_BASEPATH \
    --jobname "cycle-uc-fixed"

# --------------------
# Compute chains
# --------------------
python scripts/run_cycle_analysis.py \
    --chain $HEALTHY_MCMC \
    --path_len $CHAIN_PATH_LEN \
    --outdir $CHAIN_HEALTHY_UNFIXED_DIR \
    --environment-name $ENVIRONMENT_NAME \
    --code-basepath $MDSINE2_PAPER_CODE_PATH \
    --queue $QUEUE \
    --memory $MEM \
    --n-cpus $N_CPUS \
    --lsf-basepath $LSF_BASEPATH \
    --jobname "chain-healthy-unfixed" \
    --do_chains

python scripts/run_cycle_analysis.py \
    --chain $HEALTHY_FIXED_MCMC \
    --path_len $CHAIN_PATH_LEN \
    --outdir $CHAIN_HEALTHY_FIXED_DIR \
    --environment-name $ENVIRONMENT_NAME \
    --code-basepath $MDSINE2_PAPER_CODE_PATH \
    --queue $QUEUE \
    --memory $MEM \
    --n-cpus $N_CPUS \
    --lsf-basepath $LSF_BASEPATH \
    --jobname "chain-healthy-fixed" \
    --do_chains

python scripts/run_cycle_analysis.py \
    --chain $UC_MCMC \
    --path_len $CHAIN_PATH_LEN \
    --outdir $CHAIN_UC_UNFIXED_DIR \
    --environment-name $ENVIRONMENT_NAME \
    --code-basepath $MDSINE2_PAPER_CODE_PATH \
    --queue $QUEUE \
    --memory $MEM \
    --n-cpus $N_CPUS \
    --lsf-basepath $LSF_BASEPATH \
    --jobname "chain-uc-unfixed" \
    --do_chains

python scripts/run_cycle_analysis.py \
    --chain $UC_FIXED_MCMC \
    --path_len $CHAIN_PATH_LEN \
    --outdir $CHAIN_UC_FIXED_DIR \
    --environment-name $ENVIRONMENT_NAME \
    --code-basepath $MDSINE2_PAPER_CODE_PATH \
    --queue $QUEUE \
    --memory $MEM \
    --n-cpus $N_CPUS \
    --lsf-basepath $LSF_BASEPATH \
    --jobname "chain-uc-fixed" \
    --do_chains