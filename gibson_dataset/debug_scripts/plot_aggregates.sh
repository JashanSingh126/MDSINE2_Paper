#!/bin/bash

BASEPATH="../../output/aggregate_plots/"
TOP="10"

echo "Plotting the Aggregation of the OTUs"
echo "Files written to ${BASEPATH}"
echo "Only writing the top ${TOP} OTUs"

# Plot the OTU aggregates
python ../scripts/plot_otus.py \
    --study ../../processed_data/gibson_healthy_agg_taxa.pkl \
    --output-basepath $BASEPATH \
    --top $TOP
python ../scripts/plot_otus.py \
    --study ../../processed_data/gibson_uc_agg_taxa.pkl \
    --output-basepath $BASEPATH \
    --top $TOP