#!/bin/bash

# ====== figure 6 (eigenvalues and cycle histograms)
# Relies on output of compute_eigenvalues.sh and cycle_counting.sh
python gibson_inference/figures/figure_6.py \
    --healthy_cycles_taxa "../output/postprocessing/cycles/unfixed_clustering/healthy/paths.csv" \
    --uc_cycles_taxa "../output/postprocessing/cycles/unfixed_clustering/uc/paths.csv" \
    --healthy_cycles_clusters "../output/postprocessing/cycles/fixed_clustering/healthy/paths.csv" \
    --uc_cycles_clusters "../output/postprocessing/cycles/fixed_clustering/uc/paths.csv" \
    --eig_path "../output/postprocessing/eigenvalues/eigenvalues.npz" \
    --out_dir "../output" \
    --format "pdf" \
    --dpi 500
