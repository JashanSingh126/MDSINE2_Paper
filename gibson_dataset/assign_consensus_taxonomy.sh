#!/bin/bash

# Assign taxonomy for OTUs
python scripts/assign_taxonomy_for_consensus_seqs.py \
    --rdp-table files/assign_taxonomy_OTUs/taxonomy_RDP.txt \
    --confidence-threshold 50 \
    --output-basepath ../processed_data
