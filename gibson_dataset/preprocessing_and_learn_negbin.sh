#!/bin/bash

# Agglomerate ASVs into OTUs
python preprocess.py \
    --hamming-distance 2 \
    --rename-prefix OTU \
    --sequences files/preprocessing/gibson_16S_rRNA_v4_ASV_seqs_aligned_filtered.fa \
    --output-basepath ../processed_data

# Assign taxonomy for OTUs
python assign_taxonomy_for_consensus_seqs.py \
    --rdp-table files/assign_taxonomy_OTUs/taxonomy_RDP.txt \
    --confidence-threshold 50 \
    --output-basepath ../processed_data

# Filter the OTUs using consistency filtering
python ../step_2_filtering.py \
    --dataset ../processed_data/gibson_healthy_agg_taxa.pkl \
    --outfile ../processed_data/gibson_healthy_agg_taxa_filtered.pkl \
    --dtype rel \
    --threshold 0.0001 \
    --min-num-consecutive 7 \
    --min-num-subjects 2 \
    --colonization-time 5
python ../step_2_filtering.py \
    --dataset ../processed_data/gibson_uc_agg_taxa.pkl \
    --outfile ../processed_data/gibson_uc_agg_taxa_filtered.pkl \
    --dtype rel \
    --threshold 0.0001 \
    --min-num-consecutive 7 \
    --min-num-subjects 2 \
    --colonization-time 5

# Learn negative binomial dispersion parameters
python filter_replicates_like_other_dataset.py \
    --replicate-dataset ../processed_data/gibson_replicates_agg_taxa.pkl \
    --like-other ../processed_data/gibson_healthy_agg_taxa.pkl \
    --output-basepath ../processed_data/gibson_replicates_agg_taxa_filtered.pkl

python ../step_3_infer_negbin.py \
    --input ../processed_data/gibson_replicates_agg_taxa_filtered.pkl \
    --seed 0 \
    --burnin 2000 \
    --n-samples 6000 \
    --checkpoint 200 \
    --basepath ../output/negbin/

python ../step_4_visualize_negbin.py \
    --chain ../output/negbin/replicates/mcmc.pkl \
    --output-basepath ../output/negbin/replicates/posterior

