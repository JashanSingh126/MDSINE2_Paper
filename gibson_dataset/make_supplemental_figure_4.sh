#!/bin/bash

#Plot the alpha and beta diversity 

python scripts/supplemental_figure4.py -file1 "../processed_data/gibson_healthy_agg_taxa.pkl" \
       -file2 "../processed_data/gibson_uc_agg_taxa.pkl" \
       -file3 "../processed_data/gibson_inoculum_agg_taxa.pkl"