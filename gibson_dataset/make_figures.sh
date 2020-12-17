#!/bin/bash

# ====== figure 6 (eigenvalues and cycle histograms)
# Relies on output of compute_eigenvalues.sh and cycle_counting.sh
python scripts/figure_6.py \
    --healthy_cycles_taxa TODO \
    --uc_cycles_taxa TODO \
    --healthy_cycles_clusters TODO \
    --uc_cycles_clusters TODO \
    --eig_path TODO \
    --out_dir TODO \
    --format "pdf" \
    --dpi 500
