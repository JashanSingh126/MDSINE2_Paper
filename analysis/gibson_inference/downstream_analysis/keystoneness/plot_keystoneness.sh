#!/bin/bash

set -e
source gibson_inference/settings.sh

echo "Drawing keystoneness plots for Healthy cohort."

python gibson_inference/downstream_analysis/keystoneness/plot_keystoneness.py \
--fixed-cluster-mcmc-path "${MDSINE_FIXED_CLUSTER_OUT_DIR}/healthy/mcmc.pkl" \
--subjset-path "${MDSINE_FIXED_CLUSTER_OUT_DIR}/healthy/subjset.pkl" \
--dataset-name 'H' \
--fwsim-df-path "$DOWNSTREAM_ANALYSIS_OUT_DIR/keystoneness/healthy_fwsims.h5" \
--plot-path "$PLOTS_OUT_DIR/healthy_keystoneness.pdf" \
--format PDF

echo "Plot saved to $PLOTS_OUT_DIR/healthy_keystoneness.pdf"

echo "Drawing keystoneness plots for Dysbiotic cohort."

python gibson_inference/downstream_analysis/keystoneness/plot_keystoneness.py \
--fixed-cluster-mcmc-path "${MDSINE_FIXED_CLUSTER_OUT_DIR}/uc/mcmc.pkl" \
--subjset-path "${MDSINE_FIXED_CLUSTER_OUT_DIR}/uc/subjset.pkl" \
--dataset-name 'H' \
--fwsim-df-path "$DOWNSTREAM_ANALYSIS_OUT_DIR/keystoneness/uc_fwsims.h5" \
--plot-path "$PLOTS_OUT_DIR/uc_keystoneness.pdf" \
--format PDF

echo "Plot saved to $PLOTS_OUT_DIR/uc_keystoneness.pdf"
