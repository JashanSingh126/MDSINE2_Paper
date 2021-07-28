'''Figure 4 of the gibson paper
'''

import matplotlib.pyplot as plt
import seaborn as sns
import mdsine2 as md2
from mdsine2.names import STRNAMES
import numpy as np
import sys
import os
import pandas as pd
import argparse
from mdsine2.pylab.inference import BaseMCMC

import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import matplotlib.colors as colors
from matplotlib.collections import PatchCollection

import ete3
from Bio import Phylo

def phylogenetic_heatmap_gram_split(inoc_pkl, healthy_pkl, uc_pkl, chain_healthy,
    chain_uc, outfile, tree_fname, taxa, healthy_filter, uc_filter):
    '''Split the phylogenetic tree into gram negative and gram positive side by side.
    Gram positive is on left, gram negative is on right.

    Parameters
    ----------
    chain_healthy, chain_uc : str
        Paths to the mcmc.pkl objects for the healthy and uc cohort for a fixed
        clustering inference, respectively
    outfile : str
        Location to save the figure
    tree_fname : str
        Location to load the newick tree that shows the placement of the OTUs
    taxa : md2.TaxaSet
        This is the MDSINE2.TaxaSet object that contains all of the OTUs. This
        is used for naming
    '''
    # Load data and get taxa names
    # ---------------------------
    os.makedirs('tmp', exist_ok=True)
    names = set([])
    for chainname in [chain_healthy, chain_uc]:
        chain = md2.BaseMCMC.load(chainname)
        for otu in chain.graph.data.taxa:
            names.add(str(otu.name))
    names = list(names)

    # Split up into gram negative and gram positive
    # ---------------------------------------------
    gramneg_taxanames = []
    grampos_taxanames = []
    for name in names:
        taxon = taxa[name]
        if md2.is_gram_negative(taxon=taxon):
            gramneg_taxanames.append(name)
        else:
            grampos_taxanames.append(name)

    fig = plt.figure(figsize=(42,54))
    gs = fig.add_gridspec(112,112)

    # Make the number of rows of the heatmap proportional
    # ---------------------------------------------------
    pert_nrows = 7
    grampos_nrows = 111 - pert_nrows
    gramneg_nrows = int(grampos_nrows * len(gramneg_taxanames)/len(grampos_taxanames))
    ntwk_buffer = 0
    ntwk_row_start = 10 + gramneg_nrows + ntwk_buffer
    ntwk_nrows = int((111 - ntwk_row_start)/2)

    # Make the number of columns of the heatmap proportional to healthy and uc
    tree_ncols = 22
    gap = 2
    gap_inoc = 1
    tree_branch_ncols = 5
    heatmap_width = 55 - tree_ncols
    pert_start = 3

    mcmc = md2.BaseMCMC.load(chain_healthy)
    healthy_nclusters = len(mcmc.graph[STRNAMES.CLUSTERING_OBJ])
    mcmc = md2.BaseMCMC.load(chain_uc)
    uc_nclusters = len(mcmc.graph[STRNAMES.CLUSTERING_OBJ])

    uc_ncols = 11# int(heatmap_width*uc_nclusters/(uc_nclusters + healthy_nclusters))
    healthy_ncols = 16 #heatmap_width - uc_ncols
    print(tree_ncols, uc_ncols, healthy_ncols)
    inoc_cols = 1

    max_uc, min_uc = get_scale(chain_uc)
    max_healthy, min_healthy = get_scale(chain_healthy)
    max_= max(max_uc, max_healthy)
    min_= max(min_uc, min_healthy)
    if min_ < 1e-6:
        min_=1e-6
    if max_ > 1e-1:
        max_=1e-1

    print('Number of columns for UC heatmaps', uc_ncols)
    print('Number of columns for healthy heatmaps', healthy_ncols)
    print('Number of columns for phylogenetic trees', tree_ncols)
    print('Number of rows for Gram positive', grampos_nrows)
    print('Number of rows for Gram negative', gramneg_nrows)
    print('Number of rows for perturbations', pert_nrows)

    # Make the subplots
    # -----------------
    # gram positive
    ax_grampos_tree = fig.add_subplot(gs[
        pert_nrows:pert_nrows+grampos_nrows,
        0         :tree_branch_ncols])
    ax_grampos_tree_full = fig.add_subplot(gs[
        pert_nrows:pert_nrows+grampos_nrows,
        0         :tree_branch_ncols], facecolor='none')
    ax_grampos_healthy_pert = fig.add_subplot(gs[
        pert_start         :pert_nrows,
        gap_inoc+tree_ncols+inoc_cols:gap_inoc+tree_ncols+inoc_cols
        +healthy_ncols])
    ax_grampos_uc_pert = fig.add_subplot(gs[
        pert_start                      :pert_nrows,
        gap_inoc*2 + gap+tree_ncols+healthy_ncols+inoc_cols*2:
        tree_ncols+healthy_ncols+uc_ncols+inoc_cols*2+gap+gap_inoc*2])
    ax_grampos_pert = fig.add_subplot(gs[
        pert_start          :pert_nrows,
        gap_inoc+tree_ncols+inoc_cols:1+tree_ncols+healthy_ncols+
        uc_ncols+inoc_cols*2+gap+gap_inoc*2],
        facecolor='none')
    ax_grampos_healthy_abund = fig.add_subplot(gs[
        pert_nrows:pert_nrows+grampos_nrows,
        gap_inoc+tree_ncols+inoc_cols:tree_ncols+healthy_ncols+inoc_cols
        +gap_inoc])
    ax_grampos_uc_abund = fig.add_subplot(gs[
        pert_nrows              :pert_nrows+grampos_nrows,
        gap_inoc*2 + gap + tree_ncols+healthy_ncols+inoc_cols*2:tree_ncols
        +healthy_ncols+uc_ncols
        +inoc_cols*2+gap+ gap_inoc*2])
    ax_grampos_healthy_inoc = fig.add_subplot(gs[
        pert_nrows:pert_nrows+grampos_nrows,
        tree_ncols: tree_ncols+inoc_cols])
    ax_grampos_uc_inoc = fig.add_subplot(gs[
        pert_nrows:pert_nrows+grampos_nrows,
        gap_inoc+gap+tree_ncols+healthy_ncols+ inoc_cols:gap_inoc+
        tree_ncols+healthy_ncols + gap+inoc_cols*2])


    # gram negative
    ax_gramneg_tree = fig.add_subplot(gs[
        pert_nrows:pert_nrows+gramneg_nrows,
        56        :56+tree_branch_ncols])
    ax_gramneg_tree_full = fig.add_subplot(gs[
        pert_nrows :pert_nrows+grampos_nrows,
        56         :56+tree_branch_ncols], facecolor='none')
    ax_gramneg_healthy_pert = fig.add_subplot(gs[
        pert_start            :pert_nrows,
        gap_inoc+56+tree_ncols+inoc_cols:56+tree_ncols+healthy_ncols
        +inoc_cols+ gap_inoc])
    ax_gramneg_uc_pert = fig.add_subplot(gs[
        pert_start:pert_nrows, gap+56+tree_ncols+healthy_ncols+inoc_cols*2+gap_inoc*2:
        gap_inoc*2+gap+56+tree_ncols+healthy_ncols+uc_ncols+inoc_cols])
    ax_gramneg_pert = fig.add_subplot(gs[
        pert_start            :pert_nrows,
        56+tree_ncols + gap_inoc+ inoc_cols:56+tree_ncols+healthy_ncols+
        uc_ncols+inoc_cols*2+gap+gap_inoc*2], facecolor='none')
    ax_gramneg_healthy_abund = fig.add_subplot(gs[
        pert_nrows   :pert_nrows+gramneg_nrows,
        56+tree_ncols+gap_inoc+inoc_cols:56+tree_ncols+healthy_ncols+
        gap_inoc+inoc_cols])
    ax_gramneg_uc_abund = fig.add_subplot(gs[
        pert_nrows                  :pert_nrows+gramneg_nrows,
        gap+gap_inoc*2+56+tree_ncols+healthy_ncols+inoc_cols*2:56+
        tree_ncols+healthy_ncols+
        uc_ncols+inoc_cols+gap_inoc*2+gap])
    ax_gramneg_healthy_inoc = fig.add_subplot(gs[
        pert_nrows:pert_nrows+gramneg_nrows,
        56+tree_ncols:56+tree_ncols+inoc_cols])
    ax_gramneg_uc_inoc = fig.add_subplot(gs[
        pert_nrows:pert_nrows+gramneg_nrows,
        gap_inoc+gap+56+tree_ncols+healthy_ncols+inoc_cols:gap_inoc+
        56+tree_ncols+healthy_ncols+inoc_cols*2+gap])

    bayes_rows = 20
    bayes_row_gap = 20
    ax_bayes_healthy = fig.add_subplot(gs[67:88, 96:100])
    ax_bayes_uc = fig.add_subplot(gs[91:112, 96:100])

    ax_healthy_network = fig.add_subplot(gs[67:88, 58:79])
    ax_healthy_network.set_title("G", fontsize=50, fontweight="bold", loc="left")

    ax_uc_network = fig.add_subplot(gs[91:112, 58:79])
    ax_uc_network.set_title("I", fontsize=50, fontweight="bold", loc="left")

    _remove_border(ax_healthy_network)
    _remove_border(ax_uc_network)


    healthy_counts = get_cycle_counts(mcmc=md2.BaseMCMC.load(chain_healthy),
        bayes_filter=10)
    uc_counts = get_cycle_counts(mcmc=md2.BaseMCMC.load(chain_uc), bayes_filter=10)
    ax_bayes_healthy = generate_plot(healthy_counts, 10, ax_bayes_healthy, "H")
    ax_bayes_uc = generate_plot(uc_counts, 10, ax_bayes_uc, "J", pad=15)

    # Plot gram positive subplots
    # ---------------------------
    ax_grampos_tree, grampos_taxaname_order = _make_phylogenetic_tree(
        tree_fname=tree_fname, names=grampos_taxanames, taxa=taxa, fig=fig,
        ax=ax_grampos_tree, figlabel='E', figlabelax=ax_grampos_tree_full)

    ax_grampos_healthy_abund, grampos_healthy_colorder = _make_cluster_membership_heatmap(
        chainname=chain_healthy, study_pkl=healthy_pkl, ax=ax_grampos_healthy_abund,
        order=grampos_taxaname_order, vmax=max_, vmin=min_,
        filter_taxa=healthy_filter, binary=False, fig=fig, make_colorbar=False, count=1)

    ax_grampos_uc_abund, grampos_uc_colorder = _make_cluster_membership_heatmap(
        chainname=chain_uc, study_pkl=uc_pkl, ax=ax_grampos_uc_abund, order=grampos_taxaname_order,
        vmax=max_, vmin=min_, filter_taxa=uc_filter, binary=False, fig=fig,
        make_colorbar=False, count=2)

    ax_grampos_healthy_pert = _make_perturbation_heatmap(chainname=chain_healthy,
        min_bayes_factor=10, ax=ax_grampos_healthy_pert, colorder=grampos_healthy_colorder,
        fig=fig, make_colorbar=False, figlabel='A', render_labels=True)
    ax_grampos_uc_pert = _make_perturbation_heatmap(chainname=chain_uc,
        min_bayes_factor=10, ax=ax_grampos_uc_pert, colorder=grampos_uc_colorder,
        fig=fig, make_colorbar=False, figlabel='B', render_labels=False)

    ax_grampos_healthy_inoc = _make_inoc_heatmap(pkl=inoc_pkl, subject_name="Healthy",
        order=grampos_taxaname_order, ax=ax_grampos_healthy_inoc, fig=fig,
        vmax=max_, vmin=min_, make_colorbar=False)
    ax_grampos_uc_inoc = _make_inoc_heatmap(pkl=inoc_pkl, subject_name="Ulcerative Colitis",
        order=grampos_taxaname_order, ax=ax_grampos_uc_inoc, fig=fig,
        vmax=max_, vmin=min_, make_colorbar=False)


    ax_grampos_tree = _remove_border(ax_grampos_tree)
    ax_grampos_pert = _remove_border(ax_grampos_pert)
    ax_grampos_pert.set_title('Gram + Bacteria', fontsize=50, pad=80)

    # Plot gram negative subplots
    # ---------------------------
    ax_gramneg_tree, gramneg_taxaname_order = _make_phylogenetic_tree(
        tree_fname=tree_fname, names=gramneg_taxanames, taxa=taxa, fig=fig,
        ax=ax_gramneg_tree, figlabel='F', figlabelax=ax_gramneg_tree_full)

    ax_gramneg_healthy_abund, gramneg_healthy_colorder = _make_cluster_membership_heatmap(
        chainname=chain_healthy, study_pkl=healthy_pkl, ax=ax_gramneg_healthy_abund,
        order=gramneg_taxaname_order, vmax=max_, vmin=min_, filter_taxa=healthy_filter,
        binary=False, fig=fig, make_colorbar=False, count=3)

    ax_gramneg_uc_abund, gramneg_uc_colorder = _make_cluster_membership_heatmap(
        chainname=chain_uc, study_pkl=uc_pkl,ax=ax_gramneg_uc_abund, order=gramneg_taxaname_order,
        vmax=max_, vmin=min_, filter_taxa=uc_filter, binary=False, fig=fig,
        make_colorbar=True, count=4)

    ax_gramneg_healthy_pert = _make_perturbation_heatmap(chainname=chain_healthy,
        min_bayes_factor=10, ax=ax_gramneg_healthy_pert, colorder=gramneg_healthy_colorder,
        fig=fig, make_colorbar=True, figlabel='C', render_labels=True)
    ax_gramneg_uc_pert = _make_perturbation_heatmap(chainname=chain_uc,
        min_bayes_factor=10, ax=ax_gramneg_uc_pert, colorder=gramneg_uc_colorder,
        fig=fig, make_colorbar=False, figlabel='D', render_labels=False)

    ax_gramneg_healthy_inoc = _make_inoc_heatmap(pkl=inoc_pkl, subject_name="Healthy",
        order=gramneg_taxaname_order, ax=ax_gramneg_healthy_inoc, fig=fig,
        vmax=max_, vmin=min_, make_colorbar=False)
    ax_gramneg_uc_inoc = _make_inoc_heatmap(pkl=inoc_pkl, subject_name="Ulcerative Colitis",
        order=gramneg_taxaname_order, ax=ax_gramneg_uc_inoc, fig=fig,
        vmax=max_, vmin=min_, make_colorbar=False)

    ax_gramneg_tree = _remove_border(ax_gramneg_tree)
    ax_gramneg_pert = _remove_border(ax_gramneg_pert)
    ax_gramneg_tree_full = _remove_border(ax_gramneg_tree_full)
    ax_grampos_tree_full = _remove_border(ax_grampos_tree_full)

    ax_gramneg_pert.set_title('Gram - Bacteria', fontsize=50, pad=80)

    suffix_taxon = {'genus': '*',
        'family': '**', 'order': '***', 'class': '****', 'phylum': '*****',
        'kingdom': '******'}
    text = '$\\bf{Taxonomy }\, \\bf{ Key}$\n'
    for taxon in suffix_taxon:
        text += ' {} : {},'.format(suffix_taxon[taxon], taxon)
    text = text[0:-1]
    fig.text(x=0.51, y=0.42, s=text, fontsize=35)

    fig.subplots_adjust(wspace=0.00, left=0.005, right=0.995, hspace=0.05,
        top=.96, bottom=0.02)
    plt.savefig(outfile)
    plt.close()

def _remove_border(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel('')
    ax.set_ylabel('')
    return ax

def get_cycle_counts(mcmc: BaseMCMC, bayes_filter: float = 100.0):

    clustering = mcmc.graph[STRNAMES.CLUSTERING_OBJ]
    #M = pl.summary(mcmc.graph[STRNAMES.INTERACTIONS_OBJ], set_nan_to_0=True, section='posterior')['mean']
    #M_condensed = condense_fixed_clustering_interaction_matrix(M, clustering=clustering)
    print("Generating Bayes factors.")
    bf = md2.util.generate_interation_bayes_factors_posthoc(mcmc=mcmc, section='posterior') # (n_taxa, n_taxa)
    bf_condensed = md2.util.condense_fixed_clustering_interaction_matrix(bf, clustering=clustering)
    interactions = md2.util.condense_fixed_clustering_interaction_matrix(
        mcmc.graph[STRNAMES.INTERACTIONS_OBJ].get_trace_from_disk(section="posterior"),
        clustering=clustering
    )
    # Get interaction sign based on majority vote
    n_pos = np.sum(interactions > 0, axis=0)
    n_neg = np.sum(interactions < 0, axis=0)
    adjacencies = np.zeros(n_pos.shape)
    adjacencies[n_pos > n_neg] = +1
    adjacencies[n_pos < n_neg] = -1

    # Filter by bayes
    print("Filtering by Bayes > {}.".format(bayes_filter))
    A = np.zeros(adjacencies.shape)
    A[bf_condensed > bayes_filter] = adjacencies[bf_condensed > bayes_filter]

    n_clusters = A.shape[0]
    n_edges = np.sum(A != 0)
    total_pairs = n_clusters * (n_clusters - 1)

    plus = np.zeros(A.shape)
    plus[A > 0] = 1
    minus = np.zeros(A.shape)
    minus[A < 0] = 1
    zeroes = np.zeros(A.shape)
    zeroes[A == 0] = 1
    counts = {
        '++': np.trace(plus @ plus) / 2,
        '--': np.trace(minus @ minus) / 2,
        '-+': np.trace(minus @ plus),
        '+o': np.trace(plus @ zeroes),
        '-o': np.trace(minus @ zeroes)
    }
    return counts

def generate_plot(counts, bayes_filter, ax, title, pad=0):
    # Load into dataframe
    df = pd.DataFrame([
        {"Sign": sgn, "Count": count} for sgn, count in counts.items()
    ]).set_index("Sign")

    # Generate frequencies
    df["Frequency"] = df["Count"] / np.sum(df["Count"])
    df = df[["Frequency"]].transpose()
    df["X"] = 0

    ax = df.plot(
        x="X",
        kind='bar',
        stacked=True,
        mark_right=True,
        ax=ax
    )
    ax.xaxis.set_visible(False)
    ax.set_ylim([0.0, 1.0])
    ax.yaxis.grid()
    ax.set_axisbelow(True)
    #ax.set_yticklabels(ax.get_yticks(), fontsize=30)
    ax.tick_params(axis="y", labelsize=30)
    ax.set_title(title, fontsize=50, fontweight="bold", pad=pad)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=25, title_fontsize=30,
        title="$\\bf{Sign}$")
    return ax

def _make_inoc_heatmap(pkl, subject_name, ax, order, fig, vmax, vmin, make_colorbar):

    taxa_idx_dict = {}
    taxas = pkl.taxa
    for i in range(len(taxas)):
        taxa_idx_dict[taxas[i].name] = i

    subj = pkl[subject_name]
    inoc_abundance = subj.matrix()["rel"]
    inoc_abundance_order = []
    for otu in order:
        inoc_abundance_order.append(inoc_abundance[taxa_idx_dict[otu.strip()]])
    inoc_abundance_order = np.asarray(inoc_abundance_order)

    inoc_abundance_order += vmin -1e-10
    inoc_abundance_order = np.where(inoc_abundance_order>=vmax, vmax,
        inoc_abundance_order)

    df = pd.DataFrame(inoc_abundance_order, index=order, columns=["inoc"])
    kwargs = {'norm': LogNorm(vmin=vmin, vmax=vmax)}

    cmap = sns.cubehelix_palette(n_colors=100, as_cmap=True, start=2, rot=0,
        dark=0, light=0.5)
    cmap.set_bad(color='silver')
    cmap.set_under(color='white')

    cbaxes= None
    if make_colorbar:
        print("make colorbar inoc")
        cbaxes = fig.add_axes([0.705, 0.45, 0.015, 0.1]) # left, bottom, width, height
        heatmap = sns.heatmap(df, yticklabels = False, cmap = cmap,
            ax = ax, norm = LogNorm(vmin=vmin, vmax=vmax), xticklabels=True,
            linewidth = 0.1, linecolor = "grey", cbar=True, cbar_ax=cbaxes)
        cbar = heatmap.collections[0].colorbar
        cbar.ax.set_title('Relative\nAbundance\n (Inoculum)\n', fontsize=30, fontweight='bold')
        cbar.ax.tick_params(labelsize=30)
        cbaxes.set_ylim(1e-1, 1e-6)
    else:
        heatmap = sns.heatmap(df, yticklabels = False, cmap = cmap,
            ax = ax, norm = LogNorm(vmin=vmin, vmax=vmax),xticklabels=True,
            linewidth = 0.1, linecolor = "indianred", cbar=False)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize = 25,
    rotation = 90, color="red")

    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color("indianred")

    return ax

def _make_phylogenetic_tree(tree_fname, names, taxa, ax, fig, figlabel=None,
    figlabelax=None):

    tree = ete3.Tree(tree_fname)
    tree.prune(names, True)
    tree.write(outfile='tmp/temp.nhx')
    fontsize=19.5

    taxonomies = ['family', 'order', 'class', 'phylum', 'kingdom']
    suffix_taxa = {'genus': '*', 'family': '**', 'order': '***', 'class': '****'
        , 'phylum': '*****', 'kingdom': '******'}
    extra_taxa_added = set([])

    tree = Phylo.read('tmp/temp.nhx', format='newick')
    Phylo.draw(tree, axes=ax, do_show=False, show_confidence=False)
    taxa_order = []
    for text in ax.texts:
        taxa_order.append(text._text)
        # Substitute the name of the taxon with the species/genus if possible
        taxonname = str(text._text).replace(' ','')
        taxon = taxa[taxonname]
        suffix = '' # for defining taxonomic level outside genus
        if taxon.tax_is_defined('genus'):
            taxonname = ' ' + taxon.taxonomy['genus']
            if taxon.tax_is_defined('species'):
                spec = taxon.taxonomy['species']
                l = spec.split('/')
                if len(l) < 5:
                    spec = '/'.join(l)
                    taxonname = taxonname + ' {}'.format(spec)
                elif len(l) >= 5:
                    spec = '/'.join(l[:2])
                    taxonname = taxonname + ' {}'.format(spec)
            else:
                suffix = suffix_taxa['genus']
        else:
            found = False
            for taxlevel in taxonomies:
                if found:
                    break
                if taxon.tax_is_defined(taxlevel):
                    found = True
                    taxonname = ' ' + taxon.taxonomy[taxlevel]
                    suffix = suffix_taxa[taxlevel]
                    extra_taxa_added.add(taxlevel)

            if not found:
                taxonname = '#'*40

        taxonname += ' ' + taxon.name
        taxonname = ' ' + suffix + taxonname
        text._text = taxonname
        text._text = text._text + '- ' * (95 - len(text._text))
        text.set_fontsize(fontsize)

    if figlabel is not None:
        figlabelax.text(x=0.15, y=1.0, s=figlabel, fontsize=50, fontweight='bold',
            transform=ax.transAxes)
    #ax.set_aspect("equal")
    return ax, taxa_order

def _make_perturbation_heatmap(chainname, min_bayes_factor, ax, colorder, fig,
    make_colorbar=True, figlabel=None, render_labels=True):

    chain = md2.BaseMCMC.load(chainname)
    subjset = chain.graph.data.subjects
    clustering = chain.graph[STRNAMES.CLUSTERING_OBJ]

    matrix = np.zeros(shape=(len(chain.graph.perturbations), len(clustering)))
    bf_matrix = np.zeros(shape=(len(chain.graph.perturbations), len(clustering)))

    index = []
    for pidx, perturbation in enumerate(chain.graph.perturbations):
        index.append(perturbation.name)

        bayes_factor = md2.generate_perturbation_bayes_factors_posthoc(mcmc=chain,
            perturbation=perturbation, section='posterior')
        bf_condensed = md2.condense_fixed_clustering_perturbation(bayes_factor,
            clustering=clustering)
        values = md2.summary(perturbation, section='posterior', only=["mean"])['mean']

        for cidx in range(len(clustering)):
            aidx = list(clustering.clusters[clustering.order[cidx]].members)[0]
            if bayes_factor[aidx] >= min_bayes_factor:
                matrix[pidx, cidx] = values[aidx]
        bf_matrix[pidx] = bf_condensed


    df = pd.DataFrame(matrix, columns=_make_names_of_clusters(len(clustering)),
        index=index)
    bf_df = pd.DataFrame(bf_matrix, columns=_make_names_of_clusters(len(clustering)),
        index=index)

    #df = df[colorder]
    #bayes = pd.read_csv('healthy_bayes_factors.csv', index_col=0)
    bf_df = bf_df.mask(bf_df < np.sqrt(10), np.NaN)
    bf_df = bf_df.mask((bf_df>=np.sqrt(10)) & (bf_df<10), .01)
    bf_df = bf_df.mask((bf_df>=10) & (bf_df<100), .1)
    bf_df = bf_df.mask((bf_df>=100), 1)

    np_df = df.to_numpy()
    np_df = np.where(np_df<-5, -5, np_df)
    np_df = np.where(np_df>5, 5, np_df)

    N = np_df.shape[1]

    x=np.append(np.append(np.arange(1, N+1), np.arange(1,N+1)), np.arange(1,N+1))
    y=np.append(np.append(np.repeat('High Fat Diet',N),
        np.repeat('Vancomycin',N)), np.repeat('Gentamicin',N))

    bvec=np.append(np.append(bf_df.iloc[0,:], bf_df.iloc[1,:]), bf_df.iloc[2,:])

    if make_colorbar:
        print("Making bar")
        cax_strength = fig.add_axes([0.615, 0.45, 0.012, 0.1])
        cax_bf = fig.add_axes([0.685, 0.4875, 0.01, 0.1])

        heatmap(x=pd.Series(x), y=pd.Series(y), ax = ax, cax=cax_strength,
        render_labels=render_labels, size_scale = 500, size=bvec,
        size_range=[0, 1], y_order=['Gentamicin', 'Vancomycin', 'High Fat Diet'],
        color=np.ravel(np_df), color_range=[-5, 5],
        palette=sns.diverging_palette(240, 10, n=256)[::-1], marker='o')

        txt=['$\sqrt{10} < K_\mathrm{BF}  \leq 10$',
            '$ 10 < K_\mathrm{BF}\leq 100$', '$K_\mathrm{BF}>100$' ]
        heatmap(x=[1, 1, 1], y=txt, ax=cax_bf, cax=None, fsize=25,
        size_scale = 500, size_range=[0,1], size=[.01, .1, 1],
        y_order=txt, color=[200, 200, 200], color_range=[-5,5],
        palette=sns.diverging_palette(240, 10, n=100)[::-1], marker='o',
        render_labels=True, title="Bayes\nFactor\n", tick_right=True)

    else:
        heatmap(x=pd.Series(x), y=pd.Series(y), ax=ax, cax=None,
        render_labels=render_labels, size_scale = 500, size=bvec,
        size_range=[0, 1], y_order=['Gentamicin', 'Vancomycin', 'High Fat Diet'],
        color=np.ravel(np_df), color_range=[-5, 5],
        palette=sns.diverging_palette(240, 10, n=256)[::-1], marker='o')

    if figlabel is not None:
        ax.set_title(figlabel, fontsize=50, fontweight='bold',
            loc="left")


    return ax

def heatmap(x, y, ax, cax=None, render_labels=False, title=None, tick_right=False,
    fsize=25, **kwargs):

    if 'color' in kwargs:
        color = kwargs['color']
    else:
        color = [1]*len(x)

    if 'palette' in kwargs:
        palette = kwargs['palette']
        n_colors = len(palette)
    else:
        n_colors = 256 # Use 256 colors for the diverging color palette
        palette = sns.color_palette("Blues", n_colors)

    if 'color_range' in kwargs:
        color_min, color_max = kwargs['color_range']
    else:
        color_min, color_max = min(color), max(color) # Range of values that will be mapped to the palette, i.e. min and max possible correlation

    def value_to_color(val):
        if color_min == color_max:
            return palette[-1]
        else:
            val_position = float((val - color_min)) / (color_max - color_min) # position of value in the input range, relative to the length of the input range
            val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1
            ind = int(val_position * (n_colors - 1)) # target index in the color palette
            return palette[ind]

    if 'size' in kwargs:
        size = kwargs['size']
    else:
        size = [1]*len(x)

    if 'size_range' in kwargs:
        size_min, size_max = kwargs['size_range'][0], kwargs['size_range'][1]
    else:
        size_min, size_max = min(size), max(size)

    size_scale = kwargs.get('size_scale', 500)

    def value_to_size(val):
        if size_min == size_max:
            return 1 * size_scale
        else:
            val_position = (val - size_min) * 0.99 / (size_max - size_min) + 0.01 # position of value in the input range, relative to the length of the input range
            val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1
            return val_position * size_scale

    if 'x_order' in kwargs:
        x_names = [t for t in kwargs['x_order']]
    else:
        x_names = [t for t in sorted(set([v for v in x]))]
    x_to_num = {p[1]:p[0] for p in enumerate(x_names)}

    if 'y_order' in kwargs:
        y_names = [t for t in kwargs['y_order']]
    else:
        y_names = [t for t in sorted(set([v for v in y]))]
    y_to_num = {p[1]:p[0] for p in enumerate(y_names)}


    marker = kwargs.get('marker', 'o')

    kwargs_pass_on = {k:v for k,v in kwargs.items() if k not in [
         'color', 'palette', 'color_range', 'size', 'size_range', 'size_scale', 'marker', 'x_order', 'y_order', 'xlabel', 'ylabel'
    ]}

    ax.scatter(
        x=[x_to_num[v] for v in x],
        y=[y_to_num[v] for v in y],
        marker=marker,
        s=[value_to_size(v) for v in size],
        c=[value_to_color(v) for v in color],
        **kwargs_pass_on
    )
    ax.set_xticks([v for k,v in x_to_num.items()])
    #ax.set_xticklabels([k for k in x_to_num], horizontalalignment='right')
    if render_labels:
        ax.set_yticks([v for k,v in y_to_num.items()])
        ax.set_yticklabels([k for k in y_to_num], fontsize=fsize)
        ax.tick_params(axis='both', which='major', pad=15)
    else:
        ax.set_yticks([v for k,v in y_to_num.items()])
        ax.set_yticklabels([])
    ax.grid(False, 'major')
    ax.grid(b=True, which='minor', color='black', linestyle='-')

    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)

    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])
    ax.set_facecolor('none')

    ax.set_xticklabels([])
    ax.tick_params(axis="both", length=0, width=0)

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(0.5)
        ax.spines[axis].set_visible(True)
        ax.spines[axis].set_color('black')

    ax.set_xlabel(kwargs.get('xlabel', ''))
    ax.set_ylabel(kwargs.get('ylabel', ''))
    if title is not None:
        ax.set_title(title, fontsize=30, fontweight="bold")
    if tick_right:
        ax.yaxis.tick_right()

    ax.set_aspect("equal")

    # Add color legend on the right side of the plot
    #if color_min < color_max:
    if cax is not None and color_min<color_max:
        print("Colorbar legends")
        col_x = [0]*len(palette) # Fixed x coordinate for the bars
        bar_y=np.linspace(color_min, color_max, n_colors) # y coordinates for each of the n_colors bar
        #print(bar_y)
        bar_height = bar_y[1] - bar_y[0]
        print("height:", bar_height)
        cax.barh(
            y=bar_y,
            width=[5]*len(palette), # Make bars 5 units wide
            left=col_x, # Make bars start at 0
            height=bar_height,
            color=palette,
            linewidth=0.5)
        cax.set_xlim(1, 2) # Bars are going from 0 to 5, so lets crop the plot somewhere in the middle
        cax.set_ylim(-5, 5)
        cax.grid(False) # Hide grid
        cax.set_facecolor('none') # Make background white
        cax.set_xticks([]) # Remove horizontal ticks
        cax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3)) # Show vertical ticks for min, middle and max
        cax.yaxis.tick_right() # Show vertical ticks on the right
        cax.set_title('Perturbation\nEffect\n', fontsize=30, fontweight='bold')
        cax.tick_params(axis='y', labelsize=30)

def get_scale(name):

    chain = md2.BaseMCMC.load(name)
    subjset = chain.graph.data.subjects
    rel_abund = np.zeros(len(subjset.taxa))
    for subj in subjset:
        M = subj.matrix()['rel']
        start_idx = np.searchsorted(subj.times, 14)
        end_idx = np.searchsorted(subj.times, 21.5)

        rel_abund += np.mean(M[:,start_idx:end_idx], axis=1)
    rel_abund /= len(subjset)
    temp = np.where(rel_abund==0, np.inf, rel_abund)
    max_rel = np.max(rel_abund)
    min_rel = np.min(temp)

    print("scale:", max_rel, min_rel)
    return max_rel, min_rel

def is_zero_abundance(order, studyset, dict_):
    zero_abundant = {}
    for subj in studyset:
        M = subj.matrix()['rel']
        #print(M.shape)
        for otu in order:
            idx = dict_[otu.strip()]
            if np.sum(M[idx, :]) == 0:
                if otu.strip() not in zero_abundant:
                    zero_abundant[otu.strip()] = 0
                zero_abundant[otu.strip()] += 1
    final_dict = {}
    for otu in zero_abundant:
        if zero_abundant[otu] == len(studyset):
            final_dict[otu] = zero_abundant[otu]

    return final_dict


def _make_cluster_membership_heatmap(chainname, study_pkl, ax, order, binary, fig,
    vmax, vmin, filter_taxa, make_colorbar=True, figlabel=None, count=0):
    '''Make the heatmap of the cluster membership
    If `binary` is True, then we just have a binary vector of the membership of
    the taxon in the cluster. If `binary` is False, then the coloring is the average relative
    abundance of the ASV and the colorbar is on a log scale.
    '''
    chain = md2.BaseMCMC.load(chainname)
    taxas = [otu.name for otu in chain.graph.data.taxa]
    subjset = chain.graph.data.subjects
    clusters = chain.graph[STRNAMES.CLUSTERING_OBJ].tolistoflists()
    all_taxa = study_pkl.taxa
    taxa_idx_dict = {all_taxa[i].name : i for i in range(len(all_taxa))}
    zero_abundant_count = is_zero_abundance(order, study_pkl, taxa_idx_dict)


    # If binary is False, then get the relative abundances of the ASVS
    rel_abund = None
    if not binary:
        rel_abund = np.zeros(len(subjset.taxa))
        for subj in subjset:
            M = subj.matrix()['rel']
            start_idx = np.searchsorted(subj.times, 14)
            end_idx = np.searchsorted(subj.times, 21.5)

            rel_abund += np.mean(M[:,start_idx:end_idx], axis=1)
        rel_abund /= len(subjset)
    max_rel = np.max(rel_abund)
    min_rel = np.min(rel_abund)

    if min_rel < 1e-6:
        min_rel = 1e-6
    rel_abund = np.where(rel_abund>=vmax, vmax, rel_abund)
    rel_abund[rel_abund == 0] = vmax+1e-1

    matrix = np.zeros(shape=(len(subjset.taxa), len(clusters)))
    for cidx, cluster in enumerate(clusters):
        for oidx in cluster:
            if binary:
                matrix[oidx, cidx] = 1
            else:
                matrix[oidx, cidx] = rel_abund[oidx]

    index = [taxon.name for taxon in subjset.taxa]

    iii = 0
    for taxonnew in order:
        if taxonnew not in index:
            index.append(taxonnew)
            iii += 1

    # Add in nan rows in places that order is not in index
    matrix = np.vstack((matrix, np.zeros(shape=(iii, matrix.shape[1]))*np.nan))
    order = [str(a).replace(' ', '') for a in order]

    df = pd.DataFrame(matrix,
        columns=_make_names_of_clusters(len(clusters)),
        index=index)

    df = df.reindex(order)

    cols = np.arange(len(df.columns))
    colnames = df.columns
    newcolnames = []
    for idx in cols:
        newcolnames.append(colnames[idx])
    df = df[newcolnames]

    if not binary:
        kwargs = {'norm': LogNorm(vmin=vmin, vmax=vmax+1e-4)}
    else:
        kwargs = {}

    cmap = sns.cubehelix_palette(n_colors=100, as_cmap=True, start=2, rot=0,
        dark=0, light=0.5)
    cmap.set_bad(color='gray')
    cmap.set_under(color='white')
    cmap.set_over(color="yellowgreen")

    cmap_masked = colors.ListedColormap(['silver', "green"])
    vals = df.values
    vals += min_rel - 1e-10
    new_vals = []
    for x in range(vals.shape[0]):
        otu = order[x].strip()

        if otu in zero_abundant_count:
            print(otu, "present")
            for y in range(vals.shape[1]):

                ax.plot(y, x, marker="o", color="black", markersize=5)
            new_vals.append([1] * vals.shape[1])

        else:
            if otu not in filter_taxa and otu not in taxas:
                #for y in range(vals.shape[1]):
                #    ax.plot(y, x, marker="^", markersize=10, color="black")
                new_vals.append([1] * vals.shape[1])
            else:
                new_vals.append([np.nan] * vals.shape[1])
    new_vals = np.asarray(new_vals)
    masked_array = np.ma.masked_where(np.isnan(new_vals), new_vals)

    im = ax.imshow(vals, cmap=cmap, aspect='auto', **kwargs)
    im_masked = ax.imshow(masked_array, cmap=cmap_masked, aspect="auto")
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    xticklabels = np.arange(1, len(df.columns)+1, step=2, dtype=int)
    ax.set_xticks(ticks=np.arange(len(df.columns), step=2))
    ax.set_xticklabels(labels=xticklabels)
    for i, tick in enumerate(ax.xaxis.get_major_ticks()):
        tick.label.set_rotation(90)
        tick.label.set_fontsize(25)
        # tick.label.set_horizontalalignment('right')

    # Make grid
    ax.set_xticks(np.arange(0.5, len(df.columns), 1), minor=True)
    ax.set_yticks(np.arange(0.5, len(df.index), 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.1)

    ax.tick_params(axis='both', which='minor', left=False, bottom=False)

    if make_colorbar:
        print("Making colorbar cluster")
        cbaxes = fig.add_axes([0.53, 0.45, 0.015, 0.1]) # left, bottom, width, height
        cbar = plt.colorbar(im, cax=cbaxes, orientation='vertical')
        cbar.ax.set_title('Relative\nAbundance\n', fontsize=30, fontweight='bold')
        cbar.ax.tick_params(labelsize=30, length=10, which="major")
        cbar.ax.tick_params(length=0, which="minor")

        legend_elements = [Line2D([0], [0], marker= "s", color="white",
        label= "Colonized consistently but not \ndetected in Days 14 - 21", markerfacecolor="yellowgreen",
        markersize=35), Line2D([0], [0], marker= "s", color="white",
        label= "Did not colonize consistently", markerfacecolor="gray",
        markersize=35), Line2D([0], [0], marker= "s", color="white",
        label= "Not detected consecutively for 3 \ndays in at least 1 mice",
        markerfacecolor="silver",
        markersize=35), Line2D([0], [0], marker= "o", color="white",
        label= "Undetected throughout the\nexperiment", markerfacecolor="black",
        markersize=25)]

        lgd_ax = fig.add_axes([0.875, 0.485, 0.05, 0.05])
        lgd_ax.legend(handles=legend_elements, loc="center", fontsize=30,
            frameon=False, labelspacing=2)
        lgd_ax = _remove_border(lgd_ax)

    return ax, newcolnames

def _make_names_of_clusters(n_clusters):
    '''Standardize the way to name the clusters so that they are all the same between
    eachother
    '''
    return ['Cluster {}'.format(i) for i in range(n_clusters)]

def parse_args():

    parser = argparse.ArgumentParser(description = "files needed for making"\
    "figure 4")
    parser.add_argument("-chain1", "--chain_healthy", required = True,
        help = "pl.base.BaseMMCMC file for healthy cohort")
    parser.add_argument("-chain2", "--chain_uc", required = True,
        help = "pl.base.BaseMMCMC file for UC cohort")
    parser.add_argument("-t", "--tree_fname", required=True,
        help="file used to make the phylogenetic tree")
    parser.add_argument("-study1", "--study_healthy", required=True,
        help="pl.Base.Study file for healthy cohort")
    parser.add_argument("-study2", "--study_uc", required=True,
        help="pl.Base.Study file for UC cohort")
    parser.add_argument("-study3", "--study_inoc", required=True,
        help="pl.Base.Study file for the inoculum")

    parser.add_argument("-study4", "--detected_study_healthy", required=True,
        help="pl.Base.Study file for consistently detected OTUs in healthy")
    parser.add_argument("-study5", "--detected_study_uc", required=True,
        help="pl.Base.Study file for consistently detected OTUs in UC")
    parser.add_argument("-o", "--output_loc", required = "True",
        help = "path to the folder where the output is saved")

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    study_inoc = md2.Study.load(args.study_inoc)
    study_healthy = md2.Study.load(args.study_healthy)
    study_uc = md2.Study.load(args.study_uc)
    healthy_filtered_study = md2.Study.load(args.detected_study_healthy)
    uc_filtered_study = md2.Study.load(args.detected_study_uc)
    taxa = study_inoc.taxa

    os.makedirs(args.output_loc, exist_ok=True)
    outfile = args.output_loc + "/figure4.pdf"

    phylogenetic_heatmap_gram_split(inoc_pkl=study_inoc, healthy_pkl=study_healthy,
       uc_pkl=study_uc, chain_healthy=args.chain_healthy, chain_uc=args.chain_uc,
       outfile=outfile, tree_fname=args.tree_fname, taxa=taxa,
       healthy_filter=[otu.name for otu in healthy_filtered_study.taxa],
       uc_filter=[otu.name for otu in uc_filtered_study.taxa])
