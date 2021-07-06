#!/bin/bash

set -e
source gibson_inference/settings.sh

# Run forward simulation for each fold
SIMULATION_DT="0.01"
N_DAYS="8"

echo "Running forward simulation"
echo "Writing files to ${CV_FWSIM_OUT_DIR}"

# Healthy cohort
# --------------
for heldout in 2 3 4 5
do
	python helpers/time_lookahead.py \
	  --chain ${CV_OUT_DIR}/healthy-cv${heldout}/mcmc.pkl \
	  --validation ../processed_data/cv/healthy-cv${heldout}-validate.pkl \
	  --simulation-dt $SIMULATION_DT \
	  --n-days $N_DAYS \
	  --output-basepath $CV_FWSIM_OUT_DIR
done

# UC Cohort
# ---------
for heldout in 6 7 8 9 10
do
	python helpers/time_lookahead.py \
    --chain ${CV_OUT_DIR}/uc-cv${heldout}/mcmc.pkl \
    --validation ../processed_data/cv/uc-cv${heldout}-validate.pkl \
    --simulation-dt $SIMULATION_DT \
    --n-days $N_DAYS \
    --output-basepath $CV_FWSIM_OUT_DIR
done
