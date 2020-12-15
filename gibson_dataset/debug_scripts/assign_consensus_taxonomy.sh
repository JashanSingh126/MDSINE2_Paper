#!/bin/bash

# Assign taxonomy for OTUs
PATH="../../output/debug/processed_data"

echo "Assigning taxonomy to consensus sequences"
echo "Saving OTUs with assigned taxonomy in ${PATH}"

python ../scripts/assign_taxonomy_for_consensus_seqs.py \
--rdp-table ../files/assign_taxonomy_OTUs/taxonomy_RDP.txt \
--confidence-threshold 50 \
--output-basepath $PATH
