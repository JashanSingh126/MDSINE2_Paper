#!/bin/bash

# Input arguments
# ---------------
# This allows a single argument. The argument is a string indicating the 
# prior to use for the perturbation indicators and interaction indicators

# ErisOne parameters
# ------------------
# Path to MDSINE2_Paper code
MDSINE2_PAPER_CODE_PATH=${MDSINE2_PAPER_CODE_PATH:-"/data/cctm/darpa_perturbation_mouse_study/MDSINE2_Paper"}
MDSINE2_OUTPUT_PATH="/data/cctm/darpa_perturbation_mouse_study/MDSINE_output"
PREPROCESSED_PATH="${MDSINE2_PAPER_CODE_PATH}/analysis/output/gibson/preprocessed"

# Conda environment
ENVIRONMENT_NAME="mdsine2"
# Queues, memory, and numpy of cpus
QUEUE="vlong"
MEM="8000"
N_CPUS="1"

# Have the first argument be the sparsity we are running with. Default to strong sparse
DEFAULT_IND_PRIOR="strong-sparse"
IND_PRIOR=${1:-$DEFAULT_IND_PRIOR}

# NOTE: THESE PATHS MUST BE RELATIVE TO `MDSINE2_PAPER_CODE_PATH`
NEGBIN="${MDSINE2_OUTPUT_PATH}/negbin/replicates/mcmc.pkl"
BURNIN="5000"
N_SAMPLES="15000"
CHECKPOINT="100"
MULTIPROCESSING="0"

HEALTHY_DATASET="${PREPROCESSED_PATH}/gibson_healthy_agg_taxa_filtered.pkl"
UC_DATASET="${PREPROCESSED_PATH}/gibson_uc_agg_taxa_filtered.pkl"
INTERACTION_IND_PRIOR=${IND_PRIOR}
PERTURBATION_IND_PRIOR=${IND_PRIOR}

if [ "$IND_PRIOR" == "$DEFAULT_IND_PRIOR" ]; then
    echo "Default parameters"
    BASEPATH="${MDSINE2_OUTPUT_PATH}/mdsine2"
    FIXED_BASEPATH="${MDSINE2_OUTPUT_PATH}/mdsine2/fixed_clustering"
else
    echo "From sensitivity"
    BASEPATH="${MDSINE2_OUTPUT_PATH}/mdsine2/sensitivity"
    FIXED_BASEPATH="${MDSINE2_OUTPUT_PATH}/mdsine2/sensitivity/fixed_clustering"
fi

# Set the name of the studies if an argument is passed
HEALTHY_SEED0="healthy-seed0-${IND_PRIOR}"
HEALTHY_SEED1="healthy-seed1-${IND_PRIOR}"
UC_SEED0="uc-seed0-${IND_PRIOR}"
UC_SEED1="uc-seed1-${IND_PRIOR}"

echo $HEALTHY_SEED0
echo $BASEPATH
echo $FIXED_BASEPATH


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
    --rename-study $HEALTHY_SEED0 \
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
    --rename-study $HEALTHY_SEED1 \
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
    --rename-study $UC_SEED0 \
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
    --rename-study $UC_SEED1 \
    --output-basepath $BASEPATH \
    --fixed-output-basepath $FIXED_BASEPATH \
    --environment-name $ENVIRONMENT_NAME \
    --code-basepath $MDSINE2_PAPER_CODE_PATH \
    --queue $QUEUE \
    --memory $MEM \
    --n-cpus $N_CPUS \
    --interaction-ind-prior $INTERACTION_IND_PRIOR \
    --perturbation-ind-prior $PERTURBATION_IND_PRIOR