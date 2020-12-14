#!/bin/bash

# # Plot the OTU aggregates
# python scripts/plot_otus.py \
#     --study ../processed_data/gibson_healthy_agg_taxa.pkl \
#     --output-basepath ../output/aggregate_plots/ \
#     --top 200
# python scripts/plot_otus.py \
#     --study ../processed_data/gibson_uc_agg_taxa.pkl \
#     --output-basepath ../output/aggregate_plots/ \
#     --top 200

# Plot the OTU aggregates
python scripts/plot_otus.py \
    --study ../processed_data/gibson_healthy_agg_taxa.pkl \
    --output-basepath ../output/aggregate_plots/ \
    --top 10
python scripts/plot_otus.py \
    --study ../processed_data/gibson_uc_agg_taxa.pkl \
    --output-basepath ../output/aggregate_plots/ \
    --top 10