#!/bin/bash

set -e
source gibson_inference/settings.sh

echo "Running Keystoneness forward simulations for Healthy dataset."

python gibson_inference/downstream_analysis/keystoneness/evaluate_keystoneness.py \
--input-mcmc-fixed-cluster "${MDSINE_FIXED_CLUSTER_OUT_DIR}/healthy/mcmc.pkl" \
--study "${MDSINE_FIXED_CLUSTER_OUT_DIR}/healthy/subjset.pkl" \
--out_path $DOWNSTREAM_ANALYSIS_OUT_DIR/keystoneness/healthy_fwsims.h5 \
--n-days 64 \
--simulation-dt 0.01 \
--limit-of-detection 10000 \
--sim-max 1e20


echo "Running Keystoneness forward simulations for Dysbiotic dataset."

python gibson_inference/downstream_analysis/keystoneness/evaluate_keystoneness.py \
--input-mcmc-fixed-cluster "${MDSINE_FIXED_CLUSTER_OUT_DIR}/uc/mcmc.pkl" \
--study "${MDSINE_FIXED_CLUSTER_OUT_DIR}/uc/subjset.pkl" \
--out-dir $DOWNSTREAM_ANALYSIS_OUT_DIR/keystoneness/uc_fwsims.h5 \
--n-days 64 \
--simulation-dt 0.01 \
--limit-of-detection 10000 \
--sim-max 1e20
