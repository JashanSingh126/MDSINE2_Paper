#!/bin/bash

#makes a figure like figure 3 (only for MDSINE2)
set -e
source gibson_inference/settings.sh

python helpers/plot_error.py \
    --mdsine_path "output/gibson/forward_sim/"\
    --output_path "${PLOTS_OUT_DIR}/"

