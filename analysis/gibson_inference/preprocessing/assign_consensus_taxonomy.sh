#!/bin/bash

set -e
source gibson_inference/settings.sh

echo "Assigning Taxonomy for OTUs..."

# Assign taxonomy for OTUs
python helpers/assign_taxonomy_for_consensus_seqs.py \
    --rdp-table files/assign_taxonomy_OTUs/taxonomy_RDP.txt \
    --confidence-threshold 50 \
    --output-basepath ${PREPROCESS_DIR}

echo "Done."
