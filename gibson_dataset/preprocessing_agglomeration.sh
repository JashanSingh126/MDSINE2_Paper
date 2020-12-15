#!/bin/bash

BASEPATH="../processed_data"

echo "Agglomerating ASVs into OTUs"
echo "Writing files in the folder ${BASEPATH}"

# Agglomerate ASVs into OTUs
python scripts/preprocess.py \
    --hamming-distance 2 \
    --rename-prefix OTU \
    --sequences files/preprocessing/gibson_16S_rRNA_v4_ASV_seqs_aligned_filtered.fa \
    --output-basepath $BASEPATH \
    --remove-timepoints 0 0.5 \
    --max-n-species 2