import numpy as np
import pandas as pd
import ete3
import copy
from Bio import SeqIO, AlignIO, Phylo
import os
import sys

import pylab as pl
import util

subjset_real = pl.base.SubjectSet.load('pickles/real_subjectset.pkl')
TAX_IDXS = {'kingdom': 0, 'phylum': 1, 'class': 2,  'order': 3, 'family': 4, 
    'genus': 5, 'species': 6, 'asv': 7}

####################################################
# Make family level plots of the ASVs in the phylogenetic trees
####################################################
# import treeswift
# # Make the distance matrix
# tree = treeswift.read_tree_newick('raw_data/newick_tree_full.nhx')
# print('here')
# M = tree.distance_matrix(leaf_labels=True)
# print('done')
# df = pd.DataFrame(M)
# print('saving')
# df.to_csv('tmp/dist_matrix_tree_temp.tsv', sep='\t', index=True, header=True)
# print(df)

# Get the families of the reference trees
fname = 'tmp/RDP-11-5_TS_Processed_seq_info.csv'
df = pd.read_csv(fname, index_col=0)

fname = 'tmp/rdp_download_12588seqs.gen'
seqs = SeqIO.to_dict(SeqIO.parse(fname, 'genbank'), key_function=lambda rec:rec.name)
ref_families = {}
for seq in df.index:
    try:
        record = seqs[seq]
    except:
        # print(record)
        continue
    l = len(record.annotations['taxonomy'])
    if l != 7:
        continue
    family = record.annotations['taxonomy'][-2]
    if family not in ref_families:
        ref_families[family] = []
    ref_families[family].append(df['species_name'][seq])


chain_locs = [
    'output_real/pylab24/real_runs/strong_priors/healthy1_5_0.0001_rel_2_5/ds0_is0_b5000_ns15000_mo-1_logTrue_pertsmult/graph_leave_out-1/mcmc.pkl',
    'output_real/pylab24/real_runs/strong_priors/healthy0_5_0.0001_rel_2_5/ds0_is1_b5000_ns15000_mo-1_logTrue_pertsmult/graph_leave_out-1/mcmc.pkl']

tree_loc = 'raw_data/newick_tree_full.nhx'
os.makedirs('tmp', exist_ok=True)
os.makedirs('tmp/subtrees_125percentmedian_new_ecoli/', exist_ok=True)
asvnames = set([])
for chainloc in chain_locs:
    chain = pl.inference.BaseMCMC.load(chainloc)
    asvs = chain.graph.data.asvs

    print(asvs.names.keys())

    for asv in asvs:
        asvnames.add(str(asv.name))

# print(subjset_real.asvs.names.keys())

asvnames = list(asvnames)
set_asvnames = set(asvnames)

# totalnames = copy.deepcopy(asvnames)
# tree = ete3.Tree(tree_loc)
# for name in tree.get_leaf_names():
#     if 'ASV_' not in name:
#         totalnames.append(name)
# tree.prune(totalnames, preserve_branch_length=True)
# print('here')
# tree.write(outfile='tmp/tree_temp.nhx')
# print('here2')
tree_name = 'tmp/tree_temp.nhx'
tree = ete3.Tree(tree_name)

# print(tree.get_leaf_names())
# for family in ref_families:
#     print(ref_families[family])
#     sys.exit()

# print(tree.get_leaf_names())
# sys.exit()

d = {}
for i, aname in enumerate(asvnames):
    print('{}/{}'.format(i,len(asvnames)))
    asv = subjset_real.asvs[aname]
    if asv.tax_is_defined('family'):
        family = asv.taxonomy['family']
        if family not in d:
            d[family] = []

        if family not in ref_families:
            print('{} NOT IN REFERENCE TREE'.format(family))
            continue
        refs = ref_families[family]

        for ref in refs:
            ref = ref.replace(' ', '_')
            try:
                d[family].append(tree.get_distance(aname, ref))
                
            except:
                # print('no worked', aname, ref)

                continue

f = open('tmp/subtrees_125percentmedian_new_ecoli/family_dists_asv.txt', 'w')
family_dists = {}
for family in d:
    f.write(family + '\n')
    arr = np.asarray(d[family])
    summ = pl.variables.summary(arr)
    family_dists[family] = summ['median']
    for k,v in summ.items():
        f.write('\t{}: {}\n'.format(k,v))
f.write('total\n')
arr = []
for ele in d.values():
    arr += ele
arr = np.asarray(arr)
summ = pl.variables.summary(arr)
for k,v in summ.items():
    f.write('\t{}:{}\n'.format(k,v))


# Set the radius to 150% the median
default_radius = 1.5*summ['median']
f.write('Default radius set to 150% of median ({}): {}'.format(summ['median'], default_radius))
f.close()

# Make the distance matrix
print('reading')
df = pd.read_csv('tmp/dist_matrix_tree_temp.tsv', sep='\t', index_col=0)
print('read')
names = df.index.to_numpy()

from ete3 import TreeStyle
def my_layout_fn(node):
    if node.is_leaf() and 'ASV' in node.name:
        node.img_style["bgcolor"] = "#9db0cf"
        node.name = util.asvname_for_paper(asv=subjset_real.asvs[node.name], asvs=subjset_real.asvs)

i = 0
f = open('tmp/subtrees_125percentmedian_new_ecoli/table.tsv', 'w')
for asvname in asvnames:
    asv = subjset_real.asvs[asvname]
    print('\n\nLooking at {}, {}'.format(i,asv))
    print('-------------------------')
    
    f.write('{}\n'.format(asv.name))

    tree = ete3.Tree(tree_name)
    # Get the all elements within `radius`
    
    row = df[asv.name].to_numpy()
    idxs = np.argsort(row)

    names_found = False
    mult = 1.
    title = asv.name
    
    while not names_found:
        if mult == 3:
            title += '\nTOO MANY, BREAKING'
            break
        if asv.tax_is_defined('family'):
            if family_dists[asv.taxonomy['family']] < 1e-2:
                radius = default_radius * mult
                title += '\nfamily defined but not ref: {} Median family distance: {:.4f}'.format(asv.taxonomy['family'], radius)
            else:
                radius = family_dists[asv.taxonomy['family']] * mult
                title += '\nfamily defined: Median {} distance: {:.4f}'.format(asv.taxonomy['family'], radius)
        else:
            radius = default_radius * mult
            mmm = mult*1.5*100
            title += '\nfamily not defined: {}% Median family distance: {:.4f}'.format(mmm, radius)

        names_to_keep = []
        for idx in idxs:
            if row[idx] > radius:
                break
            if 'ASV_' in names[idx]:
                continue
            names_to_keep.append(names[idx])

        if len(names_to_keep) > 5:
            print(len(names_to_keep), ' found for ', asv.name)
            names_found = True
        else:
            print('expand radius')
            title += '\n`{}` reference seqs within radius, expanding radius by 25%'.format(len(names_to_keep))
            mult += .25

    suffix_taxa = {
        'genus': '*',
        'family': '**', 
        'order': '***', 
        'class': '****', 
        'phylum': '*****', 
        'kingdom': '******'}
    title += '\nTaxonomic Key for ASV\n'
    i = 0
    for k,v in suffix_taxa.items():
        if i == 2:
            title += '\n'
            i = 0
        title += '{}: {}'.format(v,k)
        if i == 0:
            title += ', '
        i += 1

    # print(names_to_keep)

    # Make subtree of just these names
    names_to_keep.append(asv.name)
    tree.prune(names_to_keep, preserve_branch_length=False)

    ts = TreeStyle()
    ts.layout_fn = my_layout_fn
    ts.title.add_face(ete3.TextFace(title, fsize=15), column=1)
    tree.render('tmp/subtrees_125percentmedian_new_ecoli/{}.pdf'.format(asv.name), tree_style=ts)
f.close()
sys.exit()