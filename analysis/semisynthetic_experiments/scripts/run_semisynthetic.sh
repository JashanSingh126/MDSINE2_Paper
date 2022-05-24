#!/bin/bash

set -e
source semisynthetic_experiments/settings.sh

# Generates trajectories for three noise levels: high, medium, and low 
echo "output: ${OUTPUT_DIR}"
echo "mcmc directory ${MCMC_DIR}"

seed=$1
require_arg "seed", $seed

python semisynthetic_experiments/scripts/sample.py\
    -s $seed \
    -f1 "${MCMC_DIR}"\
    -v 0.01 \
    -t 0.01 \
    -p "${OUTPUT_DIR}"