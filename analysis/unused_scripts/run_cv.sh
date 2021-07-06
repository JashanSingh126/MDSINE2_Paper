#!/bin/bash

set -e
source gibson_inference/settings.sh

NEGBIN="${NEGBIN_OUT_DIR}/replicates/mcmc.pkl"
SEED="0"
BURNIN="5000"
N_SAMPLES="15000"
CHECKPOINT="100"
MULTIPROCESSING="0"

HEALTHY_DATASET="${TAX_OUT_DIR}/gibson_healthy_agg_taxa_filtered.pkl"
UC_DATASET="${TAX_OUT_DIR}/gibson_uc_agg_taxa_filtered.pkl"

INTERACTION_IND_PRIOR="weak-agnostic"
PERTURBATION_IND_PRIOR="weak-agnostic"

echo "Computing Cross validation"
echo "CV Datasets located in: ${CV_DATASET_PATH}"
echo "CV inference outputs in: ${CV_OUT_DIR}"

# Run healthy for each subject
# ----------------------------
# Regular inference

for heldout in 2 3 4 5
do
	echo "Healthy run: heldout subject ${heldout}"
	python ../run_cross_validation.py \
			--dataset $HEALTHY_DATASET \
			--cv-basepath $CV_OUT_DIR \
			--dset-basepath $CV_DATASET_PATH \
			--negbin $NEGBIN \
			--seed $SEED \
			--burnin $BURNIN \
			--n-samples $N_SAMPLES \
			--checkpoint $CHECKPOINT \
			--multiprocessing $MULTIPROCESSING \
			--leave-out-subject $heldout \
			--interaction-ind-prior $INTERACTION_IND_PRIOR \
			--perturbation-ind-prior $PERTURBATION_IND_PRIOR

	python ../step_6_visualize_mdsine2.py \
			--chain ${CV_OUT_DIR}/healthy-cv${heldout}/mcmc.pkl \
			--output-basepath ${CV_OUT_DIR}/healthy-cv${heldout}/posterior
done


# Run uc for each subject
# -----------------------
for heldout in 6 7 8 9 10
do
	echo "UC run: heldout subject ${uc}"
	python ../run_cross_validation.py \
    --dataset $uc_DATASET \
    --cv-basepath $CV_OUT_DIR \
    --dset-basepath $CV_DATASET_PATH \
    --negbin $NEGBIN \
    --seed $SEED \
    --burnin $BURNIN \
    --n-samples $N_SAMPLES \
    --checkpoint $CHECKPOINT \
    --multiprocessing $MULTIPROCESSING \
    --leave-out-subject $heldout \
    --interaction-ind-prior $INTERACTION_IND_PRIOR \
    --perturbation-ind-prior $PERTURBATION_IND_PRIOR

	python ../step_6_visualize_mdsine2.py \
			--chain ${CV_OUT_DIR}/uc-cv${heldout}/mcmc.pkl \
			--output-basepath ${CV_OUT_DIR}/uc-cv${heldout}/posterior
done
