#!/bin/bash

echo "DEPRECATED."
exit 1

# ErisOne parameters
# ------------------
# Path to MDSINE2_Paper code
MDSINE2_PAPER_CODE_PATH=${MDSINE2_PAPER_CODE_PATH:-"/data/cctm/darpa_perturbation_mouse_study/MDSINE2_Paper"}
SCRIPT_BASEPATH="${MDSINE2_PAPER_CODE_PATH}/figures_analysis/scripts"
ERISONE_SCRIPT_BASEPATH="${MDSINE2_PAPER_CODE_PATH}/figures_analysis/erisone/scripts"
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

HEALTHY_RUN="healthy-seed0-strong-sparse"
UC_RUN="uc-seed0-strong-sparse"

HEALTHY_CHAIN="${OUTPUT_BASEPATH}/mdsine2/${HEALTHY_RUN}/mcmc.pkl"
UC_CHAIN="${OUTPUT_BASEPATH}/mdsine2/${UC_RUN}/mcmc.pkl"
HEALTHY_STUDY="${OUTPUT_BASEPATH}/mdsine2/${HEALTHY_RUN}/subjset.pkl"
UC_STUDY="${OUTPUT_BASEPATH}/mdsine2/${UC_RUN}/subjset.pkl"


# Make the tables
# ---------------
echo "Make the tables"

python ${SCRIPT_BASEPATH}/make_leave_out_tables.py \
    --chain ${HEALTHY_CHAIN} \
    --output-basepath "${TABLE_BASEPATH}" \
    --sep $SEP
python ${SCRIPT_BASEPATH}/make_leave_out_tables.py \
    --chain ${UC_CHAIN} \
    --output-basepath "${TABLE_BASEPATH}" \
    --sep $SEP

# Compute keystoneness
# --------------------
python ${ERISONE_SCRIPT_BASEPATH}/run_keystoneness.py \
    --chain $HEALTHY_CHAIN \
    --study $HEALTHY_STUDY \
    --simulation-dt $SIM_DT \
    --n-days $N_DAYS  \
    --output-basepath $TAXA_BASEPATH \
    --leave-out-table "${TABLE_BASEPATH}/${HEALTHY_RUN}-taxa.csv" \
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
    --n-days $N_DAYS  \
    --output-basepath $CLUSTER_BASEPATH \
    --leave-out-table "${TABLE_BASEPATH}/${HEALTHY_RUN}-clusters.csv" \
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
    --n-days $N_DAYS  \
    --output-basepath $TAXA_BASEPATH \
    --leave-out-table "${TABLE_BASEPATH}/${UC_RUN}-taxa.csv" \
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
    --n-days $N_DAYS  \
    --output-basepath $CLUSTER_BASEPATH \
    --leave-out-table "${TABLE_BASEPATH}/${UC_RUN}-clusters.csv" \
    --sep $SEP \
    --environment-name $ENVIRONMENT_NAME \
    --code-basepath $MDSINE2_PAPER_CODE_PATH \
    --queue $QUEUE \
    --memory $MEM \
    --n-cpus $N_CPUS \
    --lsf-basepath $LSF_BASEPATH