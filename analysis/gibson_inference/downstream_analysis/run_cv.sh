#!/bin/bash

set -e
source gibson_inference/settings.sh

BURNIN="20"
N_SAMPLES="50"
SEED="0"
CHECKPOINT="10"
MULTIPROCESSING="0"

HEALTHY_DSET="${PREPROCESS_DIR}/gibson_healthy_agg_taxa_filtered.pkl"
UC_DSET="${PREPROCESS_DIR}/gibson_uc_agg_taxa_filtered.pkl"
NEGBIN="${NEGBIN_OUT_DIR}/replicates/mcmc.pkl"

INTERACTION_IND_PRIOR="strong-sparse"
PERTURBATION_IND_PRIOR="weak-agnostic"

CV_OUT_DIR="${OUT_DIR}/cv_inference"
CV_DATA_DIR="${OUT_DIR}/cv_dataset"

SIMULATION_DT=0.01
START=NONE
N_DAYS=NONE
FWSIM_BASEPATH="${OUT_DIR}/forward_sim"

echo "Running MDSINE2 CV"
echo "Writing cv dataset to ${CV_DATA_DIR} and mcmc output to ${CV_OUT_DIR}"

echo "Healthy"
for SUBJECT_NAME in 2 3 4 5
do 
	CV_NAME="healthy-cv${SUBJECT_NAME}"
	#run the inference with the hold-out data removed
    python helpers/run_cv_inference.py \
        --dataset $HEALTHY_DSET \
        --cv-basepath $CV_OUT_DIR \
        --dset-basepath $CV_DATA_DIR \
        --negbin $NEGBIN \
        --seed $SEED \
        --burnin $BURNIN \
        --n-samples $N_SAMPLES \
        --checkpoint $CHECKPOINT \
        --multiprocessing $MULTIPROCESSING \
        --leave-out-subject $SUBJECT_NAME \
        --interaction-ind-prior $INTERACTION_IND_PRIOR \
        --perturbation-ind-prior $PERTURBATION_IND_PRIOR

    # Make the posterior as numpy arrays
    python helpers/convert_trace_to_numpy.py \
        --chain "${CV_OUT_DIR}/${CV_NAME}/mcmc.pkl" \
        --output-basepath "${CV_OUT_DIR}/${CV_NAME}/numpy_trace"  \
        --section posterior

    # Visualize the posterior
    #mdsine2 visualize-posterior \
     #   --chain "${CV_OUT_DIR}/${CV_NAME}/mcmc.pkl" \
      #  --section posterior \
       # --output-basepath "${CV_OUT_DIR}/${CV_NAME}/posterior" 

    # Compute forward simulations for this fold
    python helpers/forward_sim_validation.py \
        --input "${CV_OUT_DIR}/${CV_NAME}/mcmc.pkl" \
        --validation "${CV_DATA_DIR}/${CV_NAME}-validate.pkl" \
        --simulation-dt $SIMULATION_DT \
        --start $START \
        --n-days $N_DAYS \
        --output-basepath $FWSIM_BASEPATH \
        --save-intermediate-times 0
done

echo "CV Inference and Forward Sim for Healthy Complete " 

echo "UC"
for SUBJECT_NAME in 6 7 8 9 10
do 
	CV_NAME="uc-cv${SUBJECT_NAME}"
	#run the inference with the hold-out data removed
    python helpers/run_cv_inference.py \
        --dataset $UC_DSET \
        --cv-basepath $CV_OUT_DIR \
        --dset-basepath $CV_DATA_DIR \
        --negbin $NEGBIN \
        --seed $SEED \
        --burnin $BURNIN \
        --n-samples $N_SAMPLES \
        --checkpoint $CHECKPOINT \
        --multiprocessing $MULTIPROCESSING \
        --leave-out-subject $SUBJECT_NAME \
        --interaction-ind-prior $INTERACTION_IND_PRIOR \
        --perturbation-ind-prior $PERTURBATION_IND_PRIOR

    # Make the posterior as numpy arrays
    python helpers/convert_trace_to_numpy.py \
        --chain "${CV_OUT_DIR}/${CV_NAME}/mcmc.pkl" \
        --output-basepath "${CV_OUT_DIR}/${CV_NAME}/numpy_trace"  \
        --section posterior

    # Visualize the posterior
    #mdsine2 visualize-posterior \
     #   --chain "${CV_OUT_DIR}/${CV_NAME}/mcmc.pkl" \
      #  --section posterior \
       # --output-basepath "${CV_OUT_DIR}/${CV_NAME}/posterior" 
    
    # Compute forward simulations for this fold
    python helpers/forward_sim_validation.py \
        --input "${CV_OUT_DIR}/${CV_NAME}/mcmc.pkl" \
        --validation "${CV_DATA_DIR}/${CV_NAME}-validate.pkl" \
        --simulation-dt $SIMULATION_DT \
        --start $START \
        --n-days $N_DAYS \
        --output-basepath $FWSIM_BASEPATH \
        --save-intermediate-times 0
done

echo "CV Inference and Forward Sim for UC Complete " 


    