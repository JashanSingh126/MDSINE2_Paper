# Run forward simulation for each fold

SIMULATION_DT="0.01"
N_DAYS="8"
OUTPUT_BASEPATH="../output/mdsine2/cv/forward_sims"
ERROR_TABLE_PATH="../output/mdsine2/cv/forward_sims/errors.tsv"

# Healthy cohort
# --------------
python scripts/time_lookahead.py \
    --chain ../output/mdsine2/cv/healthy-cv2/mcmc.pkl \
    --validation ../output/processed_data/cv/healthy-cv2-validate.pkl \
    --simulation-dt $SIMULATION_DT \
    --n-days $N_DAYS \
    --output-basepath $OUTPUT_BASEPATH

python scripts/time_lookahead.py \
    --chain ../output/mdsine2/cv/healthy-cv3/mcmc.pkl \
    --validation ../output/processed_data/cv/healthy-cv3-validate.pkl \
    --simulation-dt $SIMULATION_DT \
    --n-days $N_DAYS \
    --output-basepath $OUTPUT_BASEPATH

python scripts/time_lookahead.py \
    --chain ../output/mdsine2/cv/healthy-cv4/mcmc.pkl \
    --validation ../output/processed_data/cv/healthy-cv4-validate.pkl \
    --simulation-dt $SIMULATION_DT \
    --n-days $N_DAYS \
    --output-basepath $OUTPUT_BASEPATH

python scripts/time_lookahead.py \
    --chain ../output/mdsine2/cv/healthy-cv5/mcmc.pkl \
    --validation ../output/processed_data/cv/healthy-cv5-validate.pkl \
    --simulation-dt $SIMULATION_DT \
    --n-days $N_DAYS \
    --output-basepath $OUTPUT_BASEPATH

# UC Cohort
# ---------
python scripts/time_lookahead.py \
    --chain ../output/mdsine2/cv/uc-cv6/mcmc.pkl \
    --validation ../output/processed_data/cv/uc-cv6-validate.pkl \
    --simulation-dt $SIMULATION_DT \
    --n-days $N_DAYS \
    --output-basepath $OUTPUT_BASEPATH

python scripts/time_lookahead.py \
    --chain ../output/mdsine2/cv/uc-cv7/mcmc.pkl \
    --validation ../output/processed_data/cv/uc-cv7-validate.pkl \
    --simulation-dt $SIMULATION_DT \
    --n-days $N_DAYS \
    --output-basepath $OUTPUT_BASEPATH

python scripts/time_lookahead.py \
    --chain ../output/mdsine2/cv/uc-cv8/mcmc.pkl \
    --validation ../output/processed_data/cv/uc-cv8-validate.pkl \
    --simulation-dt $SIMULATION_DT \
    --n-days $N_DAYS \
    --output-basepath $OUTPUT_BASEPATH

python scripts/time_lookahead.py \
    --chain ../output/mdsine2/cv/uc-cv9/mcmc.pkl \
    --validation ../output/processed_data/cv/uc-cv9-validate.pkl \
    --simulation-dt $SIMULATION_DT \
    --n-days $N_DAYS \
    --output-basepath $OUTPUT_BASEPATH

python scripts/time_lookahead.py \
    --chain ../output/mdsine2/cv/uc-cv10/mcmc.pkl \
    --validation ../output/processed_data/cv/uc-cv10-validate.pkl \
    --simulation-dt $SIMULATION_DT \
    --n-days $N_DAYS \
    --output-basepath $OUTPUT_BASEPATH

# Compute errors
python ../compute_forward_sim_error.py \
    --input ${OUTPUT_BASEPATH} \
    --output ${ERROR_TABLE_PATH} \
    --error RMSE relRMSE spearman