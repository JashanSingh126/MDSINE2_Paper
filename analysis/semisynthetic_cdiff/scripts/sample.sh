#!/bin/bash
set -e
source semisynthetic_cdiff/settings.sh

# Generates three cohorts (low/medium/high) noise, one per noise level.


seed=$1
require_arg "seed", $seed


python semisynthetic_cdiff/scripts/sample.py \
--mdsine_result_path ${MDSINE_BVS_PATH} \
--num_subjects ${COHORT_SIZE} \
--out_dir ${OUTPUT_DIR}/seed_${seed} \
--seed $seed \
--process_var ${PROCESS_VAR} \
-dt ${SIMULATION_DT} \
-a0 ${NEGBIN_A0} \
-a1 ${NEGBIN_A1} \
--read_depth ${READ_DEPTH} \
--low_noise ${LOW_NOISE_SCALE} \
--medium_noise ${MEDIUM_NOISE_SCALE} \
--high_noise ${HIGH_NOISE_SCALE} \
