#!/bin/bash

set -e
source gibson_inference/settings.sh
echo "Performing consistency filtering over Study objects."

# Filter the OTUs using consistency filtering

# ==== Healthy
python helpers/step_2_filtering.py \
    --dataset ${PREPROCESS_DIR}/gibson_healthy_agg_taxa.pkl \
    --outfile ${PREPROCESS_DIR}/gibson_healthy_agg_taxa_filtered.pkl \
    --dtype rel \
    --threshold 0.0001 \
    --min-num-consecutive 7 \
    --min-num-subjects 2 \
    --colonization-time 5

# ==== UC
python helpers/step_2_filtering.py \
    --dataset ${PREPROCESS_DIR}/gibson_uc_agg_taxa.pkl \
    --outfile ${PREPROCESS_DIR}/gibson_uc_agg_taxa_filtered.pkl \
    --dtype rel \
    --threshold 0.0001 \
    --min-num-consecutive 7 \
    --min-num-subjects 2 \
    --colonization-time 5

#=========================
# TODO: find out what this does.

python helpers/filter_replicates_like_other_dataset.py \
    --replicate-dataset ${PREPROCESS_DIR}/gibson_replicates_agg_taxa.pkl \
    --like-other ${PREPROCESS_DIR}/gibson_healthy_agg_taxa_filtered.pkl \
    --output-basepath ${PREPROCESS_DIR}/gibson_replicates_agg_taxa_filtered.pkl

echo "Done."
