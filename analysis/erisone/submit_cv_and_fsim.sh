#!/bin/bash

# ErisOne parameters
# ------------------
# Path to MDSINE2_Paper code
MDSINE2_PAPER_CODE_PATH=${MDSINE2_PAPER_CODE_PATH:-"/data/cctm/darpa_perturbation_mouse_study/sawal_test/mdsine2_final/MDSINE2_Paper"}
MDSINE2_OUTPUT_PATH="${MDSINE2_PAPER_CODE_PATH}/analysis/output/gibson"
PREPROCESSED_PATH="${MDSINE2_PAPER_CODE_PATH}/analysis/output/gibson/preprocessed"

# Conda environment
ENVIRONMENT_NAME="mdsine2_"
# Queues, memory (_MEM), and nodes (_N) for forward simulation  and cross-validation (CV)
FSIM_QUEUE="medium"
FSIM_MEM="4000"
FSIM_N="1"

CV_QUEUE="vlong"
CV_MEM="8000"
CV_N="1"


# Running parameters
# NOTE: THESE PATHS MUST BE RELATIVE TO `MDSINE2_PAPER_CODE_PATH`
NEGBIN="${MDSINE2_OUTPUT_PATH}/negbin/replicates/mcmc.pkl"
SEED="0"
BURNIN="400"
N_SAMPLES="1200"
CHECKPOINT="100"
MULTIPROCESSING="0"
MAX_TLA="8"
INTERACTION_IND_PRIOR="weak-agnostic"
PERTURBATION_IND_PRIOR="weak-agnostic"

DSET_BASEPATH="${MDSINE2_OUTPUT_PATH}/cv/datasets"
CV_BASEPATH="${MDSINE2_OUTPUT_PATH}/cv/mdsine2"

HEALTHY_DATASET="${PREPROCESSED_PATH}/gibson_healthy_agg_taxa_filtered.pkl"
HEALTHY_DATASET_CURR_PATH="${PREPROCESSED_PATH}/gibson_healthy_agg_taxa_filtered.pkl"
UC_DATASET="${PREPROCESSED_PATH}/gibson_uc_agg_taxa_filtered.pkl"
UC_DATASET_CURR_PATH="${PREPROCESSED_PATH}/gibson_uc_agg_taxa_filtered.pkl"

# Run healthy for each subject
# ----------------------------

python scripts/run_cv_and_forward_sim.py \
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
    --fsim-queue $FSIM_QUEUE \
    --fsim-n-cpus $FSIM_N \
    --fsim-memory $FSIM_MEM \
    --interaction-ind-prior $INTERACTION_IND_PRIOR \
    --perturbation-ind-prior $PERTURBATION_IND_PRIOR

python scripts/run_cv_and_forward_sim.py \
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
    --leave-out-subject 3 \
    --max-tla $MAX_TLA \
    --cv-queue $CV_QUEUE \
    --cv-memory $CV_MEM \
    --cv-n-cpus $CV_N \
    --fsim-queue $FSIM_QUEUE \
    --fsim-n-cpus $FSIM_N \
    --fsim-memory $FSIM_MEM \
    --interaction-ind-prior $INTERACTION_IND_PRIOR \
    --perturbation-ind-prior $PERTURBATION_IND_PRIOR

python scripts/run_cv_and_forward_sim.py \
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
    --leave-out-subject 4 \
    --max-tla $MAX_TLA \
    --cv-queue $CV_QUEUE \
    --cv-memory $CV_MEM \
    --cv-n-cpus $CV_N \
    --fsim-queue $FSIM_QUEUE \
    --fsim-n-cpus $FSIM_N \
    --fsim-memory $FSIM_MEM \
    --interaction-ind-prior $INTERACTION_IND_PRIOR \
    --perturbation-ind-prior $PERTURBATION_IND_PRIOR

python scripts/run_cv_and_forward_sim.py \
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
    --leave-out-subject 5 \
    --max-tla $MAX_TLA \
    --cv-queue $CV_QUEUE \
    --cv-memory $CV_MEM \
    --cv-n-cpus $CV_N \
    --fsim-queue $FSIM_QUEUE \
    --fsim-n-cpus $FSIM_N \
    --fsim-memory $FSIM_MEM \
    --interaction-ind-prior $INTERACTION_IND_PRIOR \
    --perturbation-ind-prior $PERTURBATION_IND_PRIOR

# Run uc for each subject
# -----------------------
python scripts/run_cv_and_forward_sim.py \
    --dataset $UC_DATASET \
    --dataset-curr-path $UC_DATASET_CURR_PATH \
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
    --leave-out-subject 6 \
    --max-tla $MAX_TLA \
    --cv-queue $CV_QUEUE \
    --cv-memory $CV_MEM \
    --cv-n-cpus $CV_N \
    --fsim-queue $FSIM_QUEUE \
    --fsim-n-cpus $FSIM_N \
    --fsim-memory $FSIM_MEM \
    --interaction-ind-prior $INTERACTION_IND_PRIOR \
    --perturbation-ind-prior $PERTURBATION_IND_PRIOR

python scripts/run_cv_and_forward_sim.py \
    --dataset $UC_DATASET \
    --dataset-curr-path $UC_DATASET_CURR_PATH \
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
    --leave-out-subject 7 \
    --max-tla $MAX_TLA \
    --cv-queue $CV_QUEUE \
    --cv-memory $CV_MEM \
    --cv-n-cpus $CV_N \
    --fsim-queue $FSIM_QUEUE \
    --fsim-n-cpus $FSIM_N \
    --fsim-memory $FSIM_MEM \
    --interaction-ind-prior $INTERACTION_IND_PRIOR \
    --perturbation-ind-prior $PERTURBATION_IND_PRIOR

python scripts/run_cv_and_forward_sim.py \
    --dataset $UC_DATASET \
    --dataset-curr-path $UC_DATASET_CURR_PATH \
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
    --leave-out-subject 8 \
    --max-tla $MAX_TLA \
    --cv-queue $CV_QUEUE \
    --cv-memory $CV_MEM \
    --cv-n-cpus $CV_N \
    --fsim-queue $FSIM_QUEUE \
    --fsim-n-cpus $FSIM_N \
    --fsim-memory $FSIM_MEM \
    --interaction-ind-prior $INTERACTION_IND_PRIOR \
    --perturbation-ind-prior $PERTURBATION_IND_PRIOR

python scripts/run_cv_and_forward_sim.py \
    --dataset $UC_DATASET \
    --dataset-curr-path $UC_DATASET_CURR_PATH \
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
    --leave-out-subject 9 \
    --max-tla $MAX_TLA \
    --cv-queue $CV_QUEUE \
    --cv-memory $CV_MEM \
    --cv-n-cpus $CV_N \
    --fsim-queue $FSIM_QUEUE \
    --fsim-n-cpus $FSIM_N \
    --fsim-memory $FSIM_MEM \
    --interaction-ind-prior $INTERACTION_IND_PRIOR \
    --perturbation-ind-prior $PERTURBATION_IND_PRIOR

python scripts/run_cv_and_forward_sim.py \
    --dataset $UC_DATASET \
    --dataset-curr-path $UC_DATASET_CURR_PATH \
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
    --leave-out-subject 10 \
    --max-tla $MAX_TLA \
    --cv-queue $CV_QUEUE \
    --cv-memory $CV_MEM \
    --cv-n-cpus $CV_N \
    --fsim-queue $FSIM_QUEUE \
    --fsim-n-cpus $FSIM_N \
    --fsim-memory $FSIM_MEM \
    --interaction-ind-prior $INTERACTION_IND_PRIOR \
    --perturbation-ind-prior $PERTURBATION_IND_PRIOR

