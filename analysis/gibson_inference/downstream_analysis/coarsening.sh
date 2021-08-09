#!/bin/bash

set -e
source gibson_inference/settings.sh

python helpers/run_coarsening.py -mcmc1 '/data/cctm/darpa_perturbation_mouse_study/MDSINE2_Paper/output/mdsine2/mixed_prior/healthy-seed0-mixed/'\
    -mcmc2 '/data/cctm/darpa_perturbation_mouse_study/MDSINE2_Paper/output/mdsine2/mixed_prior/uc-seed0-mixed/'\
    -pkl1 '/data/cctm/darpa_perturbation_mouse_study/MDSINE2_Paper/processed_data/gibson_healthy_agg_taxa.pkl'\
    -pkl2 '/data/cctm/darpa_perturbation_mouse_study/MDSINE2_Paper/processed_data/gibson_uc_agg_taxa.pkl'\
    -sf '/data/cctm/darpa_perturbation_mouse_study/MDSINE2_Paper/gibson_dataset/files/phylogenetic_placement_OTUs/placed_sequences_on_v4_region.sto'\
    -m_type 'arithmetic'\
    -k 100 \
    -o "output/gibson/coarsening"
