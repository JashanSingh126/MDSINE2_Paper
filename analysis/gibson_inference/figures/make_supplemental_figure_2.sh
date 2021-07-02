#!/bin/bash

#Plot the raw data visualization 

python gibson_inference/figures/supplemental_figure2.py -filter1 "files/figures/healthy_7.txt"\
        -filter2 "files/figures/uc_7.txt"\
        -file1 "../processed_data/gibson_healthy_agg_taxa.pkl"\
        -file2 "../processed_data/gibson_uc_agg_taxa.pkl"