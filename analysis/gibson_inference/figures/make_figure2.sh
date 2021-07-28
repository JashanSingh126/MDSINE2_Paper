#!/bin/bash

#Plot the experimental design, qpcr and relative abundance plot 

python gibson_inference/figures/figure2.py \
       -file1 "output/gibson/preprocessed/gibson_healthy_agg_taxa.pkl" \
       -file2 "output/gibson/preprocessed/gibson_uc_agg_taxa.pkl" \
       -file3 "output/gibson/preprocessed/gibson_inoculum_agg_taxa.pkl"
