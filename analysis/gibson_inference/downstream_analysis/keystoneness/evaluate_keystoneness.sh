#!/bin/bash

set -e
source gibson_inference/settings.sh

echo "Running Keystoneness forward simulations for Healthy dataset."

mdsine2 extract-abundances \
--study "${MDSINE_FIXED_CLUSTER_OUT_DIR}/healthy/subjset.pkl" \
-t 19 \
-o $DOWNSTREAM_ANALYSIS_OUT_DIR/keystoneness/healthy-initial_condition.tsv \

mdsine2 evaluate-keystoneness \
--fixed-cluster-mcmc-path "${MDSINE_FIXED_CLUSTER_OUT_DIR}/healthy/mcmc.pkl" \
--study "${MDSINE_FIXED_CLUSTER_OUT_DIR}/healthy/subjset.pkl" \
--initial-conditions $DOWNSTREAM_ANALYSIS_OUT_DIR/keystoneness/healthy-initial_condition.tsv \
-o $DOWNSTREAM_ANALYSIS_OUT_DIR/keystoneness \
--n-days 64 \
--simulation-dt 0.01 \
--limit-of-detection 10000 \
--sim-max 1e20


echo "Running Keystoneness forward simulations for Dysbiotic dataset."

mdsine2 extract-abundances \
--study "${MDSINE_FIXED_CLUSTER_OUT_DIR}/uc/subjset.pkl" \
-t 19 \
-o $DOWNSTREAM_ANALYSIS_OUT_DIR/keystoneness/uc-initial_condition.tsv \

mdsine2 evaluate-keystoneness \
--fixed-cluster-mcmc-path "${MDSINE_FIXED_CLUSTER_OUT_DIR}/uc/mcmc.pkl" \
--study "${MDSINE_FIXED_CLUSTER_OUT_DIR}/uc/subjset.pkl" \
--initial-conditions $DOWNSTREAM_ANALYSIS_OUT_DIR/keystoneness/uc-initial_condition.tsv \
-o $DOWNSTREAM_ANALYSIS_OUT_DIR/keystoneness \
--n-days 64 \
--simulation-dt 0.01 \
--limit-of-detection 10000 \
--sim-max 1e20
