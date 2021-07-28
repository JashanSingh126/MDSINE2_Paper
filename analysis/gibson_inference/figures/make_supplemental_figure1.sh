#!/bin/bash

# Plot the alpha and beta diversity

python gibson_inference/figures/supplemental_figure1.py \
-file1 "/data/cctm/darpa_perturbation_mouse_study/Sawal_MDSINE2_Paper/processed_data_all/gibson_healthy_agg_taxa.pkl" \
-file2 "/data/cctm/darpa_perturbation_mouse_study/Sawal_MDSINE2_Paper/processed_data_all/gibson_uc_agg_taxa.pkl" \
-file3 "/data/cctm/darpa_perturbation_mouse_study/Sawal_MDSINE2_Paper/processed_data_all/gibson_inoculum_agg_taxa.pkl"
