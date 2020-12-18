#!/bin/bash

# Plot the co-clustering probabilities 
python scripts/supplemental_figure_6_and_7.py -file1 "files/figures/healthy_coclusters.tsv"\
      -file2 "files/figures/uc_coclusters.tsv"\
      -file3 "files/figures/healthy_clusters.tsv"\
      -file4 "files/figures/uc_clusters.tsv"\
      -file5 "../processed_data/gibson_healthy_agg_taxa.pkl"\
      -file6 "../processed_data/gibson_uc_agg_taxa.pkl"\
      -opt True

