#!/bin/bash

set -e
source gibson_inference/settings.sh

#Plot figure4
python gibson_inference/figures/figure4.py \
    --chain_healthy "$MDSINE_OUT_DIR/healthy-seed0/mcmc.pkl" \
    --chain_uc "$MDSINE_OUT_DIR/uc-seed0/mcmc.pkl \
    --tree_fname './files/phylogenetic_placement_OTUs/phylogenetic_tree_only_query.nhx' \
    --study_healthy "${PREPROCESS_DIR}/gibson_healthy_agg_taxa.pkl" \
    --study_uc "${PREPROCESS_DIR}/gibson_uc_agg_taxa.pkl" \
    --study_inoc "${PREPROCESS_DIR}/gibson_inoculum_agg_taxa.pkl" \
    --detected_study_healthy "${PREPROCESS_TIME0_DIR}/gibson_healthy_agg_taxa_filtered3.pkl" \
    --detected_study_uc "${PREPROCESS_TIME0_DIR}/gibson_uc_agg_taxa_filtered3.pkl" \
    --output_loc "${PLOTS_OUT_DIR}"
