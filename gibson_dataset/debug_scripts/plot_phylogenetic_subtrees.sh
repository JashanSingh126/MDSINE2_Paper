#!/bin/bash

BASEPATH="../../output/subtrees/"

echo "Making the phylogenetic subtrees"
echo "Writing the files to ${BASEPATH}"

# Plot the phylogenetic subtrees for each OTU
python ../scripts/plot_phylogenetic_subtrees.py \
    --study ../../processed_data/gibson_healthy_agg_taxa.pkl \
    --output-basepath $BASEPATH \
    --tree ../files/phylogenetic_placement_OTUs/phylogenetic_tree_full_taxid.nhx \
    --seq-info ../files/subtrees/RDP-11-5_BA_TS_info.tsv \
    --family-radius-factor 1.5 \
    --top 10