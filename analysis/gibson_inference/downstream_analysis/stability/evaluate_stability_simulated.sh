#!/bin/bash

set -e
source gibson_inference/settings.sh

echo "Running forward simulations for Healthy dataset."

python gibson_inference/downstream_analysis/stability/evaluate_stability_simulated.py \
--input-mcmc $MDSINE_OUT_DIR/healthy-seed0/mcmc.pkl \
--study $MDSINE_OUT_DIR/healthy-seed0/subjset.pkl \
--out-dir $DOWNSTREAM_ANALYSIS_OUT_DIR/stability/healthy \
--perturbation '-2.0' \
--pert-start-day 20 \
--pert-end-day 34 \
--num-trials 30 \
--seed 31415 \
--gibbs-subsample 100 \
--n-days 64 \
--simulation-dt 0.01 \
--limit-of-detection 10000 \
--sim-max 1e20


echo "Running forward simulations for Dysbiotic dataset."

python gibson_inference/downstream_analysis/stability/evaluate_stability_simulated.py \
--input-mcmc $MDSINE_OUT_DIR/uc-seed0/mcmc.pkl \
--study $MDSINE_OUT_DIR/uc-seed0/subjset.pkl \
--out-dir $DOWNSTREAM_ANALYSIS_OUT_DIR/stability/uc \
--perturbation '-2.0' \
--pert-start-day 20 \
--pert-end-day 34 \
--num-trials 30 \
--seed 31415 \
--gibbs-subsample 100 \
--n-days 64 \
--simulation-dt 0.01 \
--limit-of-detection 10000 \
--sim-max 1e20
