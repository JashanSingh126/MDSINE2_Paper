#!/bin/bash

set -e
source gibson_inference/settings.sh

#Plot figure3
python gibson_inference/figures/figure5.py \
    -file1 "${OUT_DIR}/coarsening/agg_distance_threshold.csv" \
    -file2 "${OUT_DIR}/coarsening/mdsine2_sp.csv" \
    -file3 "${OUT_DIR}/coarsening/null_distribution_sp.csv" \
    -o_loc "${PLOTS_OUT_DIR}"
