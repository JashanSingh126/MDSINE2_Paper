#!/bin/bash

set -e
source gibson_inference/settings.sh

echo "Learning negative binomial dispersion parameters..."
echo "Output Directory: ${NEGBIN_OUT_DIR}"

python helpers/step_3_infer_negbin.py \
    --input "${PREPROCESS_DIR}/gibson_replicates_agg_taxa_filtered.pkl" \
    --seed 0 \
    --burnin 2000 \
    --n-samples 6000 \
    --checkpoint 200 \
    --multiprocessing 0 \
    --basepath ${NEGBIN_OUT_DIR}

python helpers/step_4_visualize_negbin.py \
    --chain "${NEGBIN_OUT_DIR}/replicates/mcmc.pkl" \
    --output-basepath "${NEGBIN_OUT_DIR}/replicates/posterior"

echo "Done."
