#!/bin/bash
#makes a figure like supplemental figure 4 (only for MDSINE2)
set -e
source gibson_inference/settings.sh

python helpers/plot_error_abundance.py \
    --mdsine_path "output/gibson/forward_sim/"\
    --output_path "${PLOTS_OUT_DIR}/"
