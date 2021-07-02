#!/bin/bash

set -e
source gibson_inference/settings.sh

echo "Agglomerating ASVs into OTUs"
echo "Using dataset found in ${DATASET_DIR}."
echo "Writing files into ${TAX_OUT_DIR}."

# Agglomerate ASVs into OTUs
python helpers/preprocess.py \
    --hamming-distance 2 \
    --rename-prefix OTU \
    --sequences files/preprocessing/gibson_16S_rRNA_v4_ASV_seqs_aligned_filtered.fa \
    --output-basepath ${TAX_OUT_DIR} \
    --remove-timepoints 0 0.5 \
    --max-n-species 2 \
    --dataset_dir ${DATASET_DIR}

echo "Done."
