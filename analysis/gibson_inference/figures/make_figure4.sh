#!/bin/bash

#Plot figure4
python gibson_inference/figures/figure4.py \
    --chain_healthy "/data/cctm/darpa_perturbation_mouse_study/MDSINE2_Paper/output/mdsine2/fixed_clustering_mixed_prior/healthy-seed0-mixed/mcmc.pkl" \
    --chain_uc "/data/cctm/darpa_perturbation_mouse_study/MDSINE2_Paper/output/mdsine2/fixed_clustering_mixed_prior/uc-seed0-mixed/mcmc.pkl" \
    --tree_fname '/data/cctm/darpa_perturbation_mouse_study/MDSINE2_Paper/gibson_dataset/files/phylogenetic_placement_OTUs/phylogenetic_tree_only_query.nhx' \
    --study_healthy "output/gibson/preprocessed/gibson_healthy_agg_taxa.pkl" \
    --study_uc "output/gibson/preprocessed/gibson_uc_agg_taxa.pkl" \
    --study_inoc "output/gibson/preprocessed/gibson_inoculum_agg_taxa.pkl" \
    --detected_study_healthy "/data/cctm/darpa_perturbation_mouse_study/Sawal_MDSINE2_Paper/processed_data_all/gibson_healthy_agg_taxa_filtered3.pkl" \
    --detected_study_uc "/data/cctm/darpa_perturbation_mouse_study/Sawal_MDSINE2_Paper/processed_data_all/gibson_uc_agg_taxa_filtered3.pkl" \
    --output_loc "gibson_inference/figures/output_figures"
