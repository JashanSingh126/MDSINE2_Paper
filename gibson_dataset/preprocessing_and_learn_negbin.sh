#!/bin/bash

python preprocess.py --hamming-distance 2 \
    --rename-prefix OTU \
    --sequences files/preprocessing/gibson_16S_rRNA_v4_seqs_aligned_filtered.fa \
    --output-basepath ../processed_data/