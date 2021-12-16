#!/bin/bash

set -e
source gibson_inference/settings.sh

echo "Plotting results of forward simulations."

python gibson_inference/downstream_analysis/stability/plot_stability.py \
--healthy-dir $DOWNSTREAM_ANALYSIS_OUT_DIR/stability/healthy \
--uc-dir $DOWNSTREAM_ANALYSIS_OUT_DIR/stability/uc \
--plot-path $PLOTS_OUT_DIR/stability.pdf \
--format PDF

echo "Plot saved to $PLOTS_OUT_DIR/stability.pdf"
