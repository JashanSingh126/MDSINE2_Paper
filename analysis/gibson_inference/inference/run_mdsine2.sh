#!/bin/bash

set -e
source gibson_inference/settings.sh

NEGBIN="${NEGBIN_OUT_DIR}/replicates/mcmc.pkl"
BURNIN="5000"
N_SAMPLES="15000"
CHECKPOINT="100"
MULTIPROCESSING="0"
HEALTHY_DSET="${PREPROCESS_DIR}/gibson_healthy_agg_taxa_filtered.pkl"
UC_DSET="${PREPROCESS_DIR}/gibson_uc_agg_taxa_filtered.pkl"
INTERACTION_IND_PRIOR="strong-sparse"
PERTURBATION_IND_PRIOR="weak-agnostic"

echo "Running MDSINE2 model"
echo "Writing files to ${MDSINE_OUT_DIR}"

# Healthy cohort
# --------------

# Seed 0
mdsine2 infer \
    --input $HEALTHY_DSET \
    --negbin $NEGBIN \
    --seed 0 \
    --burnin $BURNIN \
    --n-samples $N_SAMPLES \
    --checkpoint $CHECKPOINT \
    --multiprocessing $MULTIPROCESSING \
    --rename-study healthy-seed0 \
    --basepath $MDSINE_OUT_DIR \
    --interaction-ind-prior $INTERACTION_IND_PRIOR \
    --perturbation-ind-prior $PERTURBATION_IND_PRIOR

mdsine2 visualize-posterior \
    --chain $MDSINE_OUT_DIR/healthy-seed0/mcmc.pkl \
    --output-basepath $MDSINE_OUT_DIR/healthy-seed0/posterior

echo "Finished Healthy (seed 0)."

# Seed 1
mdsine2 infer \
    --input $HEALTHY_DSET \
    --negbin $NEGBIN \
    --seed 1 \
    --burnin $BURNIN \
    --n-samples $N_SAMPLES \
    --checkpoint $CHECKPOINT \
    --multiprocessing $MULTIPROCESSING \
    --rename-study healthy-seed1 \
    --basepath $MDSINE_OUT_DIR \
    --interaction-ind-prior $INTERACTION_IND_PRIOR \
    --perturbation-ind-prior $PERTURBATION_IND_PRIOR

mdsine2 visualize-posterior \
    --chain $MDSINE_OUT_DIR/healthy-seed1/mcmc.pkl \
    --output-basepath $MDSINE_OUT_DIR/healthy-seed1/posterior

echo "Finished Healthy (seed 1)."

# ---------
# UC cohort
# ---------
# Seed 0
mdsine2 infer \
    --input $UC_DSET \
    --negbin $NEGBIN \
    --seed 0 \
    --burnin $BURNIN \
    --n-samples $N_SAMPLES \
    --checkpoint $CHECKPOINT \
    --multiprocessing $MULTIPROCESSING \
    --rename-study uc-seed0 \
    --basepath $MDSINE_OUT_DIR \
    --interaction-ind-prior $INTERACTION_IND_PRIOR \
    --perturbation-ind-prior $PERTURBATION_IND_PRIOR

mdsine2 visualize-posterior \
    --chain $MDSINE_OUT_DIR/uc-seed0/mcmc.pkl \
    --output-basepath $MDSINE_OUT_DIR/uc-seed0/posterior

echo "Finished UC (seed 0)."

# Seed 1
mdsine2 infer \
    --input $UC_DSET \
    --negbin $NEGBIN \
    --seed 1 \
    --burnin $BURNIN \
    --n-samples $N_SAMPLES \
    --checkpoint $CHECKPOINT \
    --multiprocessing $MULTIPROCESSING \
    --rename-study uc-seed1 \
    --basepath $MDSINE_OUT_DIR \
    --interaction-ind-prior $INTERACTION_IND_PRIOR \
    --perturbation-ind-prior $PERTURBATION_IND_PRIOR

mdsine2 visualize-posterior \
    --chain $MDSINE_OUT_DIR/uc-seed1/mcmc.pkl \
    --output-basepath $MDSINE_OUT_DIR/uc-seed1/posterior

echo "Finished UC (seed 1)."
