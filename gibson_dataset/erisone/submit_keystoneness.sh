#!/bin/bash

# ErisOne parameters
# ------------------
# Path to MDSINE2_Paper code
MDSINE2_PAPER_CODE_PATH="/data/cctm/darpa_perturbation_mouse_study/MDSINE2_Paper"
SCRIPT_BASEPATH="${MDSINE2_PAPER_CODE_PATH}/gibson_dataset/scripts"
ERISONE_SCRIPT_BASEPATH="${MDSINE2_PAPER_CODE_PATH}/gibson_dataset/erisone/scripts"
OUTPUT_BASEPATH="${MDSINE2_PAPER_CODE_PATH}/output"
# Conda environment
ENVIRONMENT_NAME="mdsine2"
# Queues, memory, and numpy of cpus
QUEUE="short"
MEM="4000"
N_CPUS="1"
LSF_BASEPATH="lsf_files/keystoneness"

# Forward simulation parameters
# -----------------------------
SIM_DT="0.01"
N_DAYS="60"
KY_OUTPUT_BASEPATH="${OUTPUT_BASEPATH}/keystoneness"
SEP=","

TABLE_BASEPATH="${KY_OUTPUT_BASEPATH}/tables"
CLUSTER_BASEPATH="${KY_OUTPUT_BASEPATH}/clusters"
TAXA_BASEPATH="${KY_OUTPUT_BASEPATH}/taxa"

HEALTHY_CHAIN="${OUTPUT_BASEPATH}/mdsine2/healthy-seed0/mcmc.pkl"
UC_CHAIN="${OUTPUT_BASEPATH}/mdsine2/uc-seed0/mcmc.pkl"

HEALTHY_STUDY="${OUTPUT_BASEPATH}/mdsine2/healthy-seed0/subjset.pkl"
UC_STUDY="${OUTPUT_BASEPATH}/mdsine2/uc-seed0/subjset.pkl"


# Make the tables
# ---------------
echo "Make the tables"

python ${SCRIPT_BASEPATH}/make_leave_out_tables.py \
    --chain ${OUTPUT_BASEPATH}/mdsine2/healthy-seed0/mcmc.pkl \
    --output-basepath "${TABLE_BASEPATH}" \
    --sep $SEP
python ${SCRIPT_BASEPATH}/make_leave_out_tables.py \
    --chain ../../output/mdsine2/healthy-seed1/mcmc.pkl \
    --output-basepath "${TABLE_BASEPATH}" \
    --sep $SEP
python ${SCRIPT_BASEPATH}/make_leave_out_tables.py \
    --chain ../../output/mdsine2/uc-seed0/mcmc.pkl \
    --output-basepath "${TABLE_BASEPATH}" \
    --sep $SEP
python ${SCRIPT_BASEPATH}/make_leave_out_tables.py \
    --chain ../../output/mdsine2/uc-seed1/mcmc.pkl \
    --output-basepath "${TABLE_BASEPATH}" \
    --sep $SEP

# Compute keystoneness
# --------------------
python ${ERISONE_SCRIPT_BASEPATH}/run_keystoneness.py \
    --chain $HEALTHY_CHAIN \
    --study $HEALTHY_STUDY \
    --simulation-dt $SIM_DT \
    --n-days$N_DAYS  \
    --output-basepath $TAXA_BASEPATH \
    --leave-out-table "${TABLE_BASEPATH}/healthy-seed0-taxa.csv" \
    --sep $SEP \
    --environment-name $ENVIRONMENT_NAME \
    --code-basepath $MDSINE2_PAPER_CODE_PATH \
    --queue $QUEUE \
    --memory $MEM \
    --n-cpus $N_CPUS \
    --lsf-basepath $LSF_BASEPATH

python ${ERISONE_SCRIPT_BASEPATH}/run_keystoneness.py \
    --chain $HEALTHY_CHAIN \
    --study $HEALTHY_STUDY \
    --simulation-dt $SIM_DT \
    --n-days$N_DAYS  \
    --output-basepath $CLUSTER_BASEPATH \
    --leave-out-table "${TABLE_BASEPATH}/healthy-seed0-clusters.csv" \
    --sep $SEP \
    --environment-name $ENVIRONMENT_NAME \
    --code-basepath $MDSINE2_PAPER_CODE_PATH \
    --queue $QUEUE \
    --memory $MEM \
    --n-cpus $N_CPUS \
    --lsf-basepath $LSF_BASEPATH

python ${ERISONE_SCRIPT_BASEPATH}/run_keystoneness.py \
    --chain $UC_CHAIN \
    --study $UC_STUDY \
    --simulation-dt $SIM_DT \
    --n-days$N_DAYS  \
    --output-basepath $TAXA_BASEPATH \
    --leave-out-table "${TABLE_BASEPATH}/uc-seed0-taxa.csv" \
    --sep $SEP \
    --environment-name $ENVIRONMENT_NAME \
    --code-basepath $MDSINE2_PAPER_CODE_PATH \
    --queue $QUEUE \
    --memory $MEM \
    --n-cpus $N_CPUS \
    --lsf-basepath $LSF_BASEPATH

python ${ERISONE_SCRIPT_BASEPATH}/run_keystoneness.py \
    --chain $UC_CHAIN \
    --study $UC_STUDY \
    --simulation-dt $SIM_DT \
    --n-days$N_DAYS  \
    --output-basepath $CLUSTER_BASEPATH \
    --leave-out-table "${TABLE_BASEPATH}/uc-seed0-clusters.csv" \
    --sep $SEP \
    --environment-name $ENVIRONMENT_NAME \
    --code-basepath $MDSINE2_PAPER_CODE_PATH \
    --queue $QUEUE \
    --memory $MEM \
    --n-cpus $N_CPUS \
    --lsf-basepath $LSF_BASEPATH