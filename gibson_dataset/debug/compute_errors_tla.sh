
OUTPUT_BASEPATH="../../output/mdsine2/cv/forward_sims"
ERROR_TABLE_PATH="../../output/mdsine2/cv/forward_sims/errors.tsv"

# Compute errors
echo "Computing errors from forward simulation"
echo "Saving forward sims in ${OUTPUT_BASEPATH}"
echo "Saving Error table of MDSINE2 in ${ERROR_TABLE_PATH}"
python ../../compute_forward_sim_error.py \
    --input ${OUTPUT_BASEPATH} \
    --output ${ERROR_TABLE_PATH} \
    --error RMSE relRMSE spearman