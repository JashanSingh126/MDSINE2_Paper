#!/bin/bash

BASEPATH="../../output/debug/negbin/"

echo "Learning negative binomial dispersion parameters"
echo "Writing the output to ${BASEPATH}"

python ../../step_3_infer_negbin.py \
    --input ../../output/debug/processed_data/gibson_replicates_agg_taxa_filtered.pkl \
    --seed 0 \
    --burnin 100 \
    --n-samples 200 \
    --checkpoint 100 \
    --multiprocessing 0 \
    --basepath $BASEPATH

python ../../step_4_visualize_negbin.py \
    --chain "${BASEPATH}replicates/mcmc.pkl" \
    --output-basepath "${BASEPATH}replicates/posterior"

