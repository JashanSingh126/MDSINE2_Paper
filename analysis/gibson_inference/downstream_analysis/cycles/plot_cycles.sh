#!/bin/bash

set -e
source gibson_inference/settings.sh

echo "Plotting cycle counts."

python gibson_inference/downstream_analysis/cycles/plot_cycles.py \
--healthy_mcmc_path "${MDSINE_FIXED_CLUSTER_OUT_DIR}/healthy/mcmc.pkl" \
--uc_mcmc_path "${MDSINE_FIXED_CLUSTER_OUT_DIR}/uc/mcmc.pkl" \
--plot-path $PLOTS_OUT_DIR/cycle_counts.pdf \
--format PDF

echo "Plot saved to $PLOTS_OUT_DIR/cycle_counts.pdf"
