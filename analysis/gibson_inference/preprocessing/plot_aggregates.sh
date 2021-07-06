#!/bin/bash

set -e
source gibson_inference/settings.sh

TOP="200"

echo "Plotting the Aggregation of the OTUs."
echo "Output dir: ${PLOTS_OUT_DIR}."
echo "Only writing the top ${TOP} OTUs."

# Plot the OTU aggregates
python helpers/plot_otus.py \
    --study ${PREPROCESS_DIR}/gibson_healthy_agg_taxa.pkl \
    --output-basepath ${PLOTS_OUT_DIR}/aggregate_plots \
    --top $TOP

python helpers/plot_otus.py \
    --study ${PREPROCESS_DIR}/gibson_uc_agg_taxa.pkl \
    --output-basepath ${PLOTS_OUT_DIR}/aggregate_plots \
    --top $TOP

echo "Done."
