#!/bin/bash

set -e
source gibson_inference/settings.sh

# Plot the alpha and beta diversity

python gibson_inference/figures/supplemental_figure3.py \
    -file1 "${PREPROCESS_TIME0_DIR}/gibson_healthy_agg_taxa.pkl" \
    -file2 "${PREPROCESS_TIME0_DIR}/gibson_uc_agg_taxa.pkl" \


