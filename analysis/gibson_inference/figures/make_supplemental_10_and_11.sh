#!/bin/bash

# Plot the Interaction Coefficients
python gibson_inference/figures/supplemental_figure_10_and_11.py -file1 "files/figures/healthy_coclusters.tsv"\
      -file2 "files/figures/uc_coclusters.tsv"\
      -file3 "files/figures/healthy_clusters.tsv"\
      -file4 "files/figures/uc_clusters.tsv"\
      -file5 "../processed_data/gibson_healthy_agg_taxa.pkl"\
      -file6 "../processed_data/gibson_uc_agg_taxa.pkl"\
      -file7 "files/figures/healthy_interactions.tsv"\
      -file8 "files/figures/uc_interactions.tsv"\
      -opt True
