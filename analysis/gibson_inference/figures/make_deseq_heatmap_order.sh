#!/bin/bash


echo "HeatMap figure 2 deseq"
python gibson_inference/figures/deseq_heatmap_order.py \
    -loc "gibson_inference/figures/supplemental_figure2_files" \
    -abund "high" \
    -txt "abundant_species_order" \
    -taxo "order" \
    -o "mat_order_high" \
    -o_loc "gibson_inference/figures/output_figures"


python gibson_inference/figures/deseq_heatmap_order.py \
    -loc "gibson_inference/figures/supplemental_figure2_files" \
    -abund "low" \
    -txt "abundant_species_order" \
    -taxo "order" \
    -o "mat_order_low" \
    -o_loc "gibson_inference/figures/output_figures"
