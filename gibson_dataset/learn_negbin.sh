#!/bin/bash

BASEPATH="../output/negbin/"

echo "Learning negative binomial dispersion parameters"
echo "Writing the output to ${BASEPATH}"

python ../step_3_infer_negbin.py \
    --input ../processed_data/gibson_replicates_agg_taxa_filtered.pkl \
    --seed 0 \
    --burnin 2000 \
    --n-samples 6000 \
    --checkpoint 200 \
    --multiprocessing 0 \
    --basepath $BASEPATH

python ../step_4_visualize_negbin.py \
    --chain "${BASEPATH}replicates/mcmc.pkl" \
    --output-basepath "${BASEPATH}replicates/posterior"
