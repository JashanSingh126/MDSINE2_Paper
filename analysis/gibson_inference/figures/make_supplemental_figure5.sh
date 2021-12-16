#!/bin/bash

set -e
source gibson_inference/settings.sh


python gibson_inference/figures/supplemental_figure5.py \
-loc1  "${MDSINE_FIXED_CLUSTER_OUT_DIR}/healthy" \
-loc2 "${MDSINE_FIXED_CLUSTER_OUT_DIR}/uc" \

