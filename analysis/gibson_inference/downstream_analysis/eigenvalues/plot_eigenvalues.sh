#!/bin/bash

set -e
source gibson_inference/settings.sh

echo "Plotting eigenvalues."

python gibson_inference/downstream_analysis/eigenvalues/plot_eigenvalues.py \
--mcmc_path $MDSINE_OUT_DIR/healthy-seed0/mcmc.pkl \
--mcmc_names 'Healthy' \
--mcmc_path $MDSINE_OUT_DIR/uc-seed0/mcmc.pkl \
--mcmc_names 'Dysbiotic' \
--plot-path $PLOTS_OUT_DIR/eigenvalues.pdf \
--format PDF

echo "Plot saved to $PLOTS_OUT_DIR/eigenvalues.pdf"
