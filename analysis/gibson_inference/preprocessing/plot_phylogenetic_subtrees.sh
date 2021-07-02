#!/bin/bash

set -e
source gibson_inference/settings.sh

SUBTREE_OUT_DIR="${TAX_OUT_DIR}/subtrees"

echo "Making the phylogenetic subtrees."
echo "Writing the files to ${SUBTREE_OUT_DIR}"

# Plot the phylogenetic subtrees for each OTU
python helpers/plot_phylogenetic_subtrees.py \
    --study ${TAX_OUT_DIR}/gibson_healthy_agg_taxa.pkl \
    --output-basepath ${SUBTREE_OUT_DIR} \
    --tree files/phylogenetic_placement_OTUs/phylogenetic_tree_full_taxid.nhx \
    --seq-info files/subtrees/RDP-11-5_BA_TS_info.tsv \
    --family-radius-factor 1.5 \
    --top 200

echo "Done."
