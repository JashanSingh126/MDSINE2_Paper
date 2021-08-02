#!/bin/bash

set -e
source gibson_inference/settings.sh

NEGBIN="${NEGBIN_OUT_DIR}/replicates/mcmc.pkl"
SEED="0"
BURNIN="5000"
N_SAMPLES="15000"
CHECKPOINT="100"
MULTIPROCESSING="0"
INTERACTION_IND_PRIOR="strong-sparse"
PERTURBATION_IND_PRIOR="weak-agnostic"

echo "Running fixed clustering inference of MDSINE2"
echo "Writing files to ${MDSINE_FIXED_CLUSTER_OUT_DIR}"

# Healthy cohort
# --------------
: '
mdsine2 infer \
    --input ${PREPROCESS_DIR}/gibson_healthy_agg_taxa_filtered.pkl \
    --negbin $NEGBIN \
    --seed $SEED \
    --burnin $BURNIN \
    --n-samples $N_SAMPLES \
    --checkpoint $CHECKPOINT \
    --multiprocessing $MULTIPROCESSING \
    --basepath ${MDSINE_FIXED_CLUSTER_OUT_DIR} \
    --fixed-clustering ${MDSINE_OUT_DIR}/healthy-seed0/mcmc.pkl \
    --interaction-ind-prior $INTERACTION_IND_PRIOR \
    --perturbation-ind-prior $PERTURBATION_IND_PRIOR
mdsine2 visualize-posterior \
    --chain  "${MDSINE_FIXED_CLUSTER_OUT_DIR}/healthy/mcmc.pkl" \
    --output-basepath "${MDSINE_FIXED_CLUSTER_OUT_DIR}/healthy/posterior" \
    --is-fixed-clustering

echo "Finished Healthy."
' 
# UC cohort
# ---------
mdsine2 infer \
    --input ${PREPROCESS_DIR}/gibson_uc_agg_taxa_filtered.pkl \
    --negbin $NEGBIN \
    --seed $SEED \
    --burnin $BURNIN \
    --n-samples $N_SAMPLES \
    --checkpoint $CHECKPOINT \
    --multiprocessing $MULTIPROCESSING \
    --basepath ${MDSINE_FIXED_CLUSTER_OUT_DIR} \
    --fixed-clustering ${MDSINE_OUT_DIR}/uc-seed0/mcmc.pkl \
    --interaction-ind-prior $INTERACTION_IND_PRIOR \
    --perturbation-ind-prior $PERTURBATION_IND_PRIOR
mdsine2 visualize-posterior \
    --chain "${MDSINE_FIXED_CLUSTER_OUT_DIR}/uc/mcmc.pkl" \
    --output-basepath "${MDSINE_FIXED_CLUSTER_OUT_DIR}/uc/posterior" \
    --is-fixed-clustering

echo "Finished UC."
