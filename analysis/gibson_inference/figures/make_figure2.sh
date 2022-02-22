#!/bin/bash

set -e
source gibson_inference/settings.sh

#Plot the experimental design, qpcr and relative abundance plot 

python gibson_inference/figures/figure2.py \
       -file1 "${PREPROCESS_TIME0_DIR}/gibson_healthy_agg_taxa.pkl" \
       -file2 "${PREPROCESS_TIME0_DIR}/gibson_uc_agg_taxa.pkl" \
       -file3 "${PREPROCESS_TIME0_DIR}/gibson_inoculum_agg_taxa.pkl" \
       -o_loc "${PLOTS_OUT_DIR}"
