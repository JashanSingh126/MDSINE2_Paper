#!/bin/bash

#Plot the experimental design, qpcr and relative abundance plot 

python scripts/main_figure2.py -file1 "../processed_data/gibson_healthy_agg_taxa.pkl" \
       -file2 "../processed_data/gibson_uc_agg_taxa.pkl" \
       -file3 "../processed_data/gibson_inoculum_agg_taxa.pkl"