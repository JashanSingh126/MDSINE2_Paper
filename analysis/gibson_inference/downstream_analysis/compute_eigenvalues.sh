#!/bin/bash

set -e
source gibson_inference/settings.sh

HEALTHY_MCMC="${MDSINE_OUT_DIR}/healthy-seed0/mcmc.pkl"
UC_MCMC="${MDSINE_OUT_DIR}/uc-seed0/mcmc.pkl"

echo "Computing eigenvalues."

python helpers/compute_eigenvalues.py \
    --healthy $HEALTHY_MCMC \
    --uc $UC_MCMC \
    --out_dir $DOWNSTREAM_ANALYSIS_OUT_DIR/eigenvalues

echo "Done."
