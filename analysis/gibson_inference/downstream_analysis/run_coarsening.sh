#!/bin/bash

set -e
source gibson_inference/settings.sh

python helpers/run_PNA.py -mcmc1 "$MDSINE_OUT_DIR/healthy-seed0/mcmc.pkl"\
    -mcmc2 "$MDSINE_OUT_DIR/uc-seed0/mcmc.pkl"\
    -pkl1 "${PREPROCESS_DIR}/gibson_healthy_agg_taxa.pkl"\
    -pkl2 "${PREPROCESS_DIR}/gibson_healthy_agg_taxa.pkl"\
    -sf 'files/phylogenetic_placement_OTUs/placed_sequences_on_v4_region.sto'\
    -m_type 'arithmetic'\
    -k 10 \
    -o "${OUT_DIR}/coarsening"
