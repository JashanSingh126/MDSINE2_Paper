#!/bin/bash

# Plot the alpha and beta diversity

python gibson_inference/figures/deseq_heatmap.py \
    -loc "gibson_inference/figures/figure2_files" \
    -abund "high" \
    -txt "abundant_species" \
    -taxo "order" \
    -o "mat_order_high" \
    -o_loc "gibson_inference/figures/output_figures"


python gibson_inference/figures/deseq_heatmap.py \
    -loc "gibson_inference/figures/figure2_files" \
    -abund "low" \
    -txt "abundant_species" \
    -taxo "order" \
    -o "mat_order_low" \
    -o_loc "gibson_inference/figures/output_figures"
