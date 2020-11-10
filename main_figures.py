import sys
import logging
import os
import time
import pandas as pd
import shutil
import argparse
import pickle

import numpy as np
import scipy
import scipy.stats
import scipy.sparse
import scipy.spatial
from scipy.cluster.hierarchy import linkage
import skbio
import skbio.diversity
import skbio.stats

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as plticker
from matplotlib.ticker import ScalarFormatter, LogFormatter, LogFormatterSciNotation, FixedLocator
import matplotlib.patches as patches
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
# from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import TransformedBbox, BboxPatch, BboxConnector 
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.legend as mlegend
from matplotlib.colors import LogNorm
import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox

import ete3
# from ete3 import TreeStyle
import Bio
from Bio import Phylo

import pylab as pl
import diversity
import synthetic
import base
import names
import names as MDSINE2_names

DATAPATH = './pickles/real_subjectset.pkl'
DATAPATH_INOCULUM = './pickles/inoculum_subjectset.pkl'
BASEPATH = './output_figures/'
HEALTHY_SUBJECTS = ['2','3','4','5']
UNHEALTHY_SUBJECTS = ['6','7','8','9','10']

TAXLEVEL_PLURALS = {
    'genus': 'Genera', 'Genus': 'Genera',
    'family': 'Families', 'Family': 'Families',
    'order': 'Orders', 'Order': 'Orders',
    'class': 'Classes', 'Class': 'Classes',
    'phylum': 'Phyla', 'Phylum': 'Phyla',
    'kingdom': 'Kingdoms', 'Kingdom': 'Kingdoms'
}

TAXLEVEL_INTS = ['species', 'genus', 'family', 'order', 'class', 'phylum', 'kingdom']
TAXLEVEL_REV_IDX = {
    'species': 0,
    'genus': 1,
    'family': 2,
    'order': 3,
    'class': 4,
    'phylum': 5,
    'kingdom': 6
}

DATA_FIGURE_COLORS = {}

XKCD_COLORS1 = sns.color_palette('muted', n_colors=10)
XKCD_COLORS2 = sns.color_palette("dark", n_colors=10)

XKCD_COLORS = []
for lst in [XKCD_COLORS1, XKCD_COLORS2]:
    # lst = lst[::-1]
    for c in lst:
        XKCD_COLORS.append(c)


XKCD_COLORS_IDX = 0

def loaddata(healthy):
    '''Load the subjectset data

    Parameters
    ----------
    healthy : bool, None
        If True, return the healthy subjects
        If False, return the non-healthy subjects
        If None, return all of the subjects

    Returns
    -------
    pl.base.SubjectSet
    '''
    subjset = pl.base.SubjectSet.load(DATAPATH)
    # subjset.pop_times([0])
    if healthy is not None:
        if not pl.isbool(healthy):
            raise TypeError('`healthy` ({}) must be a bool'.format(healthy))
        if healthy:
            subjset.pop_subject(UNHEALTHY_SUBJECTS)
        else:
            subjset.pop_subject(HEALTHY_SUBJECTS)
    return subjset

def loadinoculum():
    '''Load the inoculum data

    Returns
    -------
    pl.base.SubjectSet
    '''
    subjset = pl.base.SubjectSet.load(DATAPATH_INOCULUM)
    return subjset

def unbias_var_estimate(vals):
    vals = np.asarray(vals)
    mean = np.mean(vals)
    a = np.sum((vals - mean)**2)
    return a / (len(vals)-1)

def _set_type_of_point_bc(t, subj):
    '''Sets the type of marker that is used in beta diversity figure
    each perturbation, post perturbation, and colonization gets a different number.

    Here we know there are three perturbations
    '''
    p0 = subj.perturbations[0]
    p1 = subj.perturbations[1]
    p2 = subj.perturbations[2]

    if t < p0.start:
        return 0
    elif t >= p0.start and t <= p0.end:
        return 1
    elif t > p0.end and t < p1.start:
        return 2
    elif t >= p1.start and t <= p1.end:
        return 3
    elif t > p1.end and t < p2.start:
        return 4
    elif t >= p2.start and t <= p2.end:
        return 5
    else:
        return 6

def mark_inset(parent_axes, inset_axes, loc1a=1, loc1b=1, loc2a=2, loc2b=2, **kwargs):
    rect = TransformedBbox(inset_axes.viewLim, parent_axes.transData)

    pp = BboxPatch(rect, fill=False, **kwargs)
    parent_axes.add_patch(pp)

    p1 = BboxConnector(inset_axes.bbox, rect, loc1=loc1a, loc2=loc1b, **kwargs)
    inset_axes.add_patch(p1)
    p1.set_clip_on(False)
    p2 = BboxConnector(inset_axes.bbox, rect, loc1=loc2a, loc2=loc2b, **kwargs)
    inset_axes.add_patch(p2)
    p2.set_clip_on(False)

def _cnt_times(df, times, times_cnts, t2idx):
    for col in df.columns:
        if col in times:
            times_cnts[t2idx[col]] += 1
    return times_cnts

def _add_unequal_col_dataframes(df, dfother, times, times_cnts, t2idx):

    times_cnts = _cnt_times(dfother, times, times_cnts, t2idx)
    if df is None:
        return dfother, times_cnts

    cols_toadd_df = []
    cols_toadd_dfother = []
    for col in dfother.columns:
        if col not in df.columns:
            cols_toadd_df.append(col)
    for col in df.columns:
        if col not in dfother.columns:
            cols_toadd_dfother.append(col)

    df = pd.concat([df,
        pd.DataFrame(np.zeros(shape=(len(df.index), len(cols_toadd_df))), 
            index=df.index, columns=cols_toadd_df)], axis=1)
    dfother = pd.concat([dfother,
        pd.DataFrame(np.zeros(shape=(len(dfother.index), len(cols_toadd_dfother))), 
            index=dfother.index, columns=cols_toadd_dfother)], axis=1)


    return dfother.reindex(df.index) + df, times_cnts

def tablelegend(ax, col_labels=None, row_labels=None, title_label="", *args, **kwargs):
    '''Place a table legend on the axes.

    Creates a legend where the labels are not directly placed with the artists, 
    but are used as row and column headers, looking like this:

    title_label   | col_labels[1] | col_labels[2] | col_labels[3]
    -------------------------------------------------------------
    row_labels[1] |
    row_labels[2] |              <artists go there>
    row_labels[3] |


    Parameters
    ----------

    ax : `matplotlib.axes.Axes`
        The artist that contains the legend table, i.e. current axes instant.

    col_labels : list of str, optional
        A list of labels to be used as column headers in the legend table.
        `len(col_labels)` needs to match `ncol`.

    row_labels : list of str, optional
        A list of labels to be used as row headers in the legend table.
        `len(row_labels)` needs to match `len(handles) // ncol`.

    title_label : str, optional
        Label for the top left corner in the legend table.

    ncol : int
        Number of columns.


    Other Parameters
    ----------------
    Refer to `matplotlib.legend.Legend` for other parameters.
    '''
    #################### same as `matplotlib.axes.Axes.legend` #####################
    handles, labels, extra_args, kwargs = mlegend._parse_legend_args([ax], *args, **kwargs)
    if len(extra_args):
        raise TypeError('legend only accepts two non-keyword arguments')

    if col_labels is None and row_labels is None:
        ax.legend_ = mlegend.Legend(ax, handles, labels, **kwargs)
        ax.legend_._remove_method = ax._remove_legend
        return ax.legend_
    #################### modifications for table legend ############################
    else:
        ncol = kwargs.pop('ncol')
        handletextpad = kwargs.pop('handletextpad', 0 if col_labels is None else -2)
        title_label = [title_label]

        # blank rectangle handle
        extra = [Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)]

        # empty label
        empty = [""]

        # number of rows infered from number of handles and desired number of columns
        nrow = len(handles) // ncol

        # organise the list of handles and labels for table construction
        if col_labels is None:
            assert nrow == len(row_labels), "nrow = len(handles) // ncol = %s, but should be equal to len(row_labels) = %s." % (nrow, len(row_labels))
            leg_handles = extra * nrow
            leg_labels  = row_labels
        elif row_labels is None:
            assert ncol == len(col_labels), "ncol = %s, but should be equal to len(col_labels) = %s." % (ncol, len(col_labels))
            leg_handles = []
            leg_labels  = []
        else:
            assert nrow == len(row_labels), "nrow = len(handles) // ncol = %s, but should be equal to len(row_labels) = %s." % (nrow, len(row_labels))
            assert ncol == len(col_labels), "ncol = %s, but should be equal to len(col_labels) = %s." % (ncol, len(col_labels))
            leg_handles = extra + extra * nrow
            leg_labels  = title_label + row_labels
        for col in range(ncol):
            if col_labels is not None:
                leg_handles += extra
                leg_labels  += [col_labels[col]]
            leg_handles += handles[col*nrow:(col+1)*nrow]
            leg_labels  += empty * nrow

        # Create legend
        ax.legend_ = mlegend.Legend(ax, leg_handles, leg_labels, ncol=ncol+int(row_labels is not None), handletextpad=handletextpad, **kwargs)
        ax.legend_._remove_method = ax._remove_legend
        return ax.legend_

def colorplot():

    # colors = sns.xkcd_palette(XKCD_COLORS)
    # colors = sns.cubehelix_palette(6, start=1, rot=-0.75, light=0.6)
    colors = sns.cubehelix_palette(6, light=0.6 )
    sns.palplot(colors)
    plt.show()

def _get_top(df, cutoff_frac_abundance, taxlevel, asvname_map=None):
    matrix = df.values
    abunds = np.sum(matrix, axis=1)
    namemap = {}

    a = abunds / abunds.sum()
    a = np.sort(a)[::-1]

    cutoff_num = None
    for i in range(len(a)):
        if a[i] < cutoff_frac_abundance:
            cutoff_num = i
            break
    if cutoff_num is None:
        raise ValueError('Error')
    print('Cutoff Num:', cutoff_num)

    idxs = np.argsort(abunds)[-cutoff_num:][::-1]
    dfnew = df.iloc[idxs, :]

    if asvname_map is not None:
        indexes = df.index
        for idx in idxs:
            namemap[indexes[idx]] = asvname_map[indexes[idx]]

    # Add everything else as 'Other'
    vals = None
    for idx in range(len(df.index)):
        if idx not in idxs:
            if vals is None:
                vals = df.values[idx, :]
            else:
                vals += df.values[idx, :]
        
    dfother = pd.DataFrame([vals], columns=df.columns, index=['{} with <{}% total abund'.format(
        TAXLEVEL_PLURALS[taxlevel], cutoff_frac_abundance*100)])
    df = dfnew.append(dfother)

    if asvname_map is not None:
        print('\n\n')
        print('Name map')
        print('--------')
        for k,v in namemap.items():
            print()
            print(k)
            print(v)

    return df


# Phylogenetic and clustering heatmap
# -----------------------------------
def _make_names_of_clusters(n_clusters):
    '''Standardize the way to name the clusters so that they are all the same between
    eachother
    '''
    return ['Cluster {}'.format(i) for i in range(n_clusters)]

def _make_cluster_membership_heatmap(chainname, ax, order, binary, fig, make_colorbar=True, figlabel=None):
    '''Make the heatmap of the cluster membership
    If `binary` is True, then we just have a binary vector of the membership of
    the asv in the cluster. If `binary` is False, then the coloring is the average relative
    abundance of the ASV and the colorbar is on a log scale.
    '''
    chain = pl.inference.BaseMCMC.load(chainname)
    subjset = chain.graph.data.subjects
    clusters = chain.graph[names.STRNAMES.CLUSTERING_OBJ].toarray()

    # If binary is False, then get the relative abundances of the ASVS
    rel_abund = None
    if not binary:
        rel_abund = np.zeros(len(subjset.asvs))
        for subj in subjset:
            M = subj.matrix()['rel']
            start_idx = np.searchsorted(subj.times, 7)
            end_idx = np.searchsorted(subj.times, 20)

            rel_abund += np.mean(M[:,start_idx:end_idx], axis=1)
        rel_abund /= len(subjset)
        max_rel = np.max(rel_abund)
        min_rel = np.min(rel_abund)

    # print('here')
    # print(min_rel)
    if min_rel < 0.001:
        min_rel = 0.001
    rel_abund[rel_abund == 0] = min_rel
    # print(max_rel)

    matrix = np.zeros(shape=(len(subjset.asvs), len(clusters)))
    for cidx, cluster in enumerate(clusters):
        for oidx in cluster:
            if binary:
                matrix[oidx, cidx] = 1
            else:
                matrix[oidx, cidx] = rel_abund[oidx]

    index = [str(name) for name in subjset.asvs.names.order]

    iii = 0
    for asvnew in order:
        if asvnew not in index:
            index.append(asvnew)
            iii += 1

    # Add in nan rows in places that order is not in index
    matrix = np.vstack((matrix, np.zeros(shape=(iii, matrix.shape[1]))*np.nan))
    order = [str(a).replace(' ', '') for a in order]

    df = pd.DataFrame(matrix,
        columns=_make_names_of_clusters(len(clusters)),
        index=index)

    
    # indices = {}
    # for name in df.index:
    #     indices[name] = name.replace('OTU', 'ASV')
    # df1 = df.rename(index=indices)
    
    # with open('cluster_heatmap.pkl', 'wb') as handle:
    #     pickle.dump(df1, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # print(df1.head())
    
    # Reindex the dataframe based on the order of the phylogenetic tree
    df = df.reindex(order)

    # # Reindex the columns based off of a dendrogram
    # values = df.values
    # Z = linkage(values.T, 'ward', optimal_ordering=True)
    # n_cols = df.shape[1]
    # cols = []
    # for row in range(Z.shape[0]):
    #     z1 = Z[row,0]
    #     z2 = Z[row,1]
    #     if z1 < n_cols:
    #         cols.append(z1)
    #     if z2 < n_cols:
    #         cols.append(z2)
    # cols = np.asarray(cols)[::-1]
    cols = np.arange(len(df.columns))

    # print(Z)
    # print(cols)
    colnames = df.columns
    newcolnames = []
    for idx in cols:
        newcolnames.append(colnames[idx])
    df = df[newcolnames]

    if not binary:
        kwargs = {'norm': LogNorm(vmin=min_rel, vmax=max_rel)}
    else:
        kwargs = {}

    cmap = sns.cubehelix_palette(n_colors=100, as_cmap=True, start=2, rot=0, dark=0, light=0.5)
    cmap.set_bad(color='silver')
    cmap.set_under(color='white')
    vals = df.values
    vals += min_rel - 1e-10
    im = ax.imshow(vals, cmap=cmap, aspect='auto', **kwargs)
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
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.1)

    ax.tick_params(axis='both', which='minor', left=False, bottom=False)

    if make_colorbar:
        cbaxes = fig.add_axes([0.92, 0.1, 0.02, 0.1]) # left, bottom, width, height
        cbar = plt.colorbar(im, cax=cbaxes, orientation='vertical')
        cbar.ax.set_title('Relative\nAbundance', fontsize=30, fontweight='bold')
        cbar.ax.tick_params(labelsize=40)

    return ax, newcolnames

def _make_perturbation_heatmap(chainname, min_bayes_factor, ax, colorder, fig, make_colorbar=True, figlabel=None,
    render_labels=True):
    chain = pl.inference.BaseMCMC.load(chainname)
    subjset = chain.graph.data.subjects
    clustering = chain.graph[names.STRNAMES.CLUSTERING_OBJ]

    matrix = np.zeros(shape=(len(chain.graph.perturbations), len(clustering)))

    index = []
    for pidx, perturbation in enumerate(chain.graph.perturbations):
        index.append(perturbation.name)

        indicator_trace = ~np.isnan(perturbation.get_trace_from_disk(section='posterior'))
        bayes_factor = pl.variables.summary(indicator_trace, only=['mean'])['mean']
        bayes_factor = bayes_factor/(1. - bayes_factor)
        bayes_factor = bayes_factor * (perturbation.probability.prior.b.value + 1) / \
            (perturbation.probability.prior.a.value + 1)
        values = pl.variables.summary(perturbation, section='posterior', 
            only=['mean'])['mean']

        for cidx in range(len(clustering)):
            aidx = list(clustering.clusters[clustering.order[cidx]].members)[0]
            if bayes_factor[aidx] >= min_bayes_factor:
                matrix[pidx, cidx] = values[aidx]

    df = pd.DataFrame(matrix, columns=_make_names_of_clusters(len(clustering)), 
        index=index)

    # with open('perturbation_heatmap.pkl', 'wb') as handle:
    #     pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # print(df.head())

    df = df[colorder]

    max_val = np.max(np.absolute(df.values))
    max_val = 5
    im = ax.imshow(df.values, cmap='bwr_r', aspect='auto', vmin=-max_val, 
        vmax=max_val)
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.xaxis.set_major_locator(plt.NullLocator())
    
    # Make grid
    ax.set_xticks(np.arange(0.5, len(df.columns), 1), minor=True)
    ax.set_yticks(np.arange(0.5, len(subjset.perturbations), 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.1)

    ax.tick_params(axis='both', which='minor', left=False, bottom=False)
    
    if make_colorbar:
        cbaxes = fig.add_axes([0.92, 0.3, 0.02, 0.1]) # left, bottom, width, height
        cbar = plt.colorbar(im, cax=cbaxes, orientation='vertical', ticks=[-5,-2.5,0,2.5,5])
        cbar.ax.set_yticklabels(['<-5', '-2.5', '0', '2.5', '>5'], fontsize=30)
        cbar.ax.set_title('Perturbation\nEffect', fontsize=35, fontweight='bold')

    if render_labels:
        ax.set_yticks(np.arange(len(subjset.perturbations)), minor=False)
        ax.set_yticklabels(list(df.index))
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_rotation(0)
            tick.label.set_fontsize(30)
    else:
        ax.set_yticks([])

    if figlabel is not None:
        ax.text(x=0.0, y=1.05, s=figlabel, fontsize=50, fontweight='bold',
            transform=ax.transAxes)

    return ax

def _make_phylogenetic_tree(treename, names, asvs, ax, fig, figlabel=None, figlabelax=None):

    tree = ete3.Tree(treename)
    tree.prune(names, True)
    tree.write(outfile='tmp/temp.nhx')
    fontsize=22

    taxonomies = ['family', 'order', 'class', 'phylum', 'kingdom']
    suffix_taxa = {'genus': '*',
        'family': '**', 'order': '***', 'class': '****', 'phylum': '*****', 'kingdom': '******'}
    extra_taxa_added = set([])

    tree = Phylo.read('tmp/temp.nhx', format='newick')
    Phylo.draw(tree, axes=ax, do_show=False, show_confidence=False)
    asv_order = []
    for text in ax.texts:
        asv_order.append(text._text)
        # Substitute the name of the asv with the species/genus if possible
        asvname = str(text._text).replace(' ','')
        asv = asvs[asvname]
        suffix = '' # for defining taxonomic level outside genus
        if asv.tax_is_defined('genus'):
            asvname = ' ' + asv.taxonomy['genus']
            if asv.tax_is_defined('species'):
                spec = asv.taxonomy['species']
                l = spec.split('/')
                if len(l) < 3:
                    spec = '/'.join(l)
                    asvname = asvname + ' {}'.format(spec)
                elif len(l) >= 3:
                    spec = '/'.join(l[:2])
                    asvname = asvname + ' {}'.format(spec)
            else:
                suffix = suffix_taxa['genus']
        else:
            found = False
            for taxa in taxonomies:
                if found:
                    break
                if asv.tax_is_defined(taxa):
                    found = True
                    asvname = ' ' + asv.taxonomy[taxa]
                    suffix = suffix_taxa[taxa]
                    extra_taxa_added.add(taxa)

            if not found:
                asvname = '#'*40

        asvname += ' ' + asv.name
        asvname = ' ' + suffix + asvname
        text._text = str(asvname.replace('OTU_', 'ASV'))   
        text._text = text._text + '- ' * 55
        text.set_fontsize(fontsize)

    if figlabel is not None:
        figlabelax.text(x=0.15, y=1.0, s=figlabel, fontsize=50, fontweight='bold',
            transform=ax.transAxes)

    # # Make the taxnonmic key on the right hand side
    # text = '$\\bf{Taxonomy} \\bf{Key}$\n'
    # for taxa in suffix_taxa:
    #     text += '{} - {}\n'.format(suffix_taxa[taxa], taxa)
    # if side_by_side is not None:
    #     if not side_by_side:
    #         fig.text(0.1, 0.875, text, fontsize=18)
    #     else:
    #         fig.text(0.9, 0.3, text, fontsize=18)
    

    return ax, asv_order

def phylogenetic_heatmap_side_by_side():

    chainnamehealthy = 'output_real/pylab24/real_runs/strong_priors/fixed_top/healthy1_5_0.0001_rel_2_5/' \
        'ds0_is0_b5000_ns15000_mo-1_logTrue_pertsmult/graph_leave_out-1/mcmc.pkl'
    chainnameuc = 'output_real/pylab24/real_runs/strong_priors/fixed_top/healthy0_5_0.0001_rel_2_5/' \
            'ds0_is3_b5000_ns15000_mo-1_logTrue_pertsmult/graph_leave_out-1/mcmc.pkl'

    names = set([])
    for chainname in [chainnamehealthy, chainnameuc]:
        chain = pl.inference.BaseMCMC.load(chainname)
        asvs = chain.graph.data.asvs
        for asvname in asvs.names.order:
            names.add(str(asvname))

    names = list(names)

    fig = plt.figure(figsize=(30,45))
    gs = fig.add_gridspec(17,21)
    ax_phyl_healthy = fig.add_subplot(gs[2+1:,:4])
    ax_clus_healthy = fig.add_subplot(gs[2+1:,9:15])
    ax_pert_healthy = fig.add_subplot(gs[2+0,9:15])

    ax_clus_uc = fig.add_subplot(gs[2+1:,15:20])
    ax_pert_uc = fig.add_subplot(gs[2+0,15:20])

    ax_network_healthy = fig.add_subplot(gs[0:2, 9:14], zorder=-50)
    ax_network_uc = fig.add_subplot(gs[0:2, 15:20], zorder=-50)

    arr_man = mpimg.imread('figures/hairball_network_healthy.jpg')
    imagebox = OffsetImage(arr_man, zoom=0.13)
    ab = AnnotationBbox(imagebox, (0.5, 0.475), pad=0, box_alignment=(0.5,0.5))
    ax_network_healthy.add_artist(ab)
    ax_network_healthy.text(x=0, y=0.9, s='A', fontsize=45, fontweight='bold',
        transform=ax_network_healthy.transAxes)

    arr_man = mpimg.imread('figures/hairball_network_uc.jpg')
    imagebox = OffsetImage(arr_man, zoom=0.13)
    ab = AnnotationBbox(imagebox, (0.5, 0.475), pad=0, box_alignment=(0.5,0.5))
    ax_network_uc.add_artist(ab)
    ax_network_uc.text(x=0, y=0.9, s='B', fontsize=45, fontweight='bold',
        transform=ax_network_uc.transAxes)

    treename = 'raw_data/newick_tree_full.nhx'

    # Healthy
    subjset = loaddata(None)
    asvs = subjset.asvs
    ax_phyl_healthy, order = _make_phylogenetic_tree(treename=treename, names=names, asvs=asvs, ax=ax_phyl_healthy, 
        fig=fig, healthy=True, side_by_side=True, figlabel='C')
    plt.savefig(BASEPATH + 'phylo_clustering_heatmap_side_by_side_RDP_alignment.pdf')
    plt.savefig(BASEPATH + 'phylo_clustering_heatmap_side_by_side_RDP_alignment.png')

    ax_clus_healthy, colorder = _make_cluster_membership_heatmap(chainname=chainnamehealthy, ax=ax_clus_healthy, 
        order=order, binary=False, fig=fig, make_colorbar=False)
    ax_pert_healthy = _make_perturbation_heatmap(chainname=chainnamehealthy, min_bayes_factor=10, 
        ax=ax_pert_healthy, colorder=colorder, fig=fig, make_colorbar=False)

    ax_phyl_healthy.spines['top'].set_visible(False)
    ax_phyl_healthy.spines['bottom'].set_visible(False)
    ax_phyl_healthy.spines['left'].set_visible(False)
    ax_phyl_healthy.spines['right'].set_visible(False)
    ax_phyl_healthy.xaxis.set_major_locator(plt.NullLocator())
    ax_phyl_healthy.xaxis.set_minor_locator(plt.NullLocator())
    ax_phyl_healthy.yaxis.set_major_locator(plt.NullLocator())
    ax_phyl_healthy.yaxis.set_minor_locator(plt.NullLocator())
    ax_phyl_healthy.set_xlabel('')
    ax_phyl_healthy.set_ylabel('')
    ax_network_healthy.set_title('Healthy Cohort', fontsize=40)
    ax_pert_healthy.text(x=0, y=1.05, s='D', fontsize=45, fontweight='bold', 
        transform=ax_pert_healthy.transAxes)

    ax_network_uc.set_title('Ulcerative Colitis Cohort', fontsize=40)

    # UC
    ax_clus_uc, colorder = _make_cluster_membership_heatmap(chainname=chainnameuc, ax=ax_clus_uc, 
        order=order, binary=False, fig=fig)
    ax_pert_uc = _make_perturbation_heatmap(chainname=chainnameuc, min_bayes_factor=10, 
        ax=ax_pert_uc, colorder=colorder, fig=fig, render_labels=False)
    ax_pert_uc.text(x=0, y=1.05, s='E', fontsize=45, fontweight='bold', 
        transform=ax_pert_uc.transAxes)

    fig.subplots_adjust(wspace=0.2, left=0.02, right=0.92, hspace=0.05,
        top=.98, bottom=0.02)

    # fig.suptitle('Test', fontsize=30)

    ax_network_healthy.spines['top'].set_visible(False)
    ax_network_healthy.spines['bottom'].set_visible(False)
    ax_network_healthy.spines['left'].set_visible(False)
    ax_network_healthy.spines['right'].set_visible(False)
    ax_network_healthy.xaxis.set_major_locator(plt.NullLocator())
    ax_network_healthy.xaxis.set_minor_locator(plt.NullLocator())
    ax_network_healthy.yaxis.set_major_locator(plt.NullLocator())
    ax_network_healthy.yaxis.set_minor_locator(plt.NullLocator())
    ax_network_healthy.set_xlabel('')
    ax_network_healthy.set_ylabel('')

    ax_network_uc.spines['top'].set_visible(False)
    ax_network_uc.spines['bottom'].set_visible(False)
    ax_network_uc.spines['left'].set_visible(False)
    ax_network_uc.spines['right'].set_visible(False)
    ax_network_uc.xaxis.set_major_locator(plt.NullLocator())
    ax_network_uc.xaxis.set_minor_locator(plt.NullLocator())
    ax_network_uc.yaxis.set_major_locator(plt.NullLocator())
    ax_network_uc.yaxis.set_minor_locator(plt.NullLocator())
    ax_network_uc.set_xlabel('')
    ax_network_uc.set_ylabel('')

    plt.savefig(BASEPATH + 'phylo_clustering_heatmap_side_by_side_RDP_alignment.pdf')
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

def phylogenetic_heatmap_gram_split():
    '''Split the phylogenetic tree into gram negative and gram positive side by side.

    Gram positive is on left, gram negative is on right.
    There are approximately 3 times as many gram positive than negative. Make the subplot for 
    gram engative proportionally small on the right so it is the same scale as on the left and 
    put the networks underneath.

    '''
    # Load data and get asv names
    # ---------------------------
    import util as MDISNE2_util
    chainnamehealthy = 'output_real/pylab24/real_runs/strong_priors/fixed_top/healthy1_5_0.0001_rel_2_5/' \
        'ds0_is0_b5000_ns15000_mo-1_logTrue_pertsmult/graph_leave_out-1/mcmc.pkl'
    chainnameuc = 'output_real/pylab24/real_runs/strong_priors/fixed_top/healthy0_5_0.0001_rel_2_5/' \
            'ds0_is3_b5000_ns15000_mo-1_logTrue_pertsmult/graph_leave_out-1/mcmc.pkl'
    treename = 'raw_data/phylogenetic_tree_branch_len_preserved.nhx'

    names = set([])
    for chainname in [chainnamehealthy, chainnameuc]:
        chain = pl.inference.BaseMCMC.load(chainname)
        asvs = chain.graph.data.asvs
        for asvname in asvs.names.order:
            names.add(str(asvname))

    names = list(names)
    subjset_real = loaddata(None)
    asvs = subjset_real.asvs

    # Split up into gram negative and gram positive
    # ---------------------------------------------
    gramneg_asvnames = []
    grampos_asvnames = []
    for name in names:
        asv = asvs[name]
        if MDISNE2_util.is_gram_negative(asv=asv):
            gramneg_asvnames.append(name)
        else:
            grampos_asvnames.append(name)

    # Divide up the figure into a 100x100 grid
    fig = plt.figure(figsize=(40,55))
    gs = fig.add_gridspec(100,100)

    # Make the number of rows of the heatmap proportional
    # ---------------------------------------------------
    pert_nrows = 7
    grampos_nrows = 100 - pert_nrows
    gramneg_nrows = int(grampos_nrows * len(gramneg_asvnames)/len(grampos_asvnames))
    ntwk_buffer = 0
    ntwk_row_start = 10 + gramneg_nrows + ntwk_buffer
    ntwk_nrows = int((100 - ntwk_row_start)/2)

    # Make the number of columns of the heatmap proportional to healthy and uc
    tree_ncols = 24
    tree_branch_ncols = 5
    heatmap_width = 50 - tree_ncols

    mcmc = pl.inference.BaseMCMC.load(chainnamehealthy)
    healthy_nclusters = len(mcmc.graph[MDSINE2_names.STRNAMES.CLUSTERING_OBJ])
    mcmc = pl.inference.BaseMCMC.load(chainnameuc)
    uc_nclusters = len(mcmc.graph[MDSINE2_names.STRNAMES.CLUSTERING_OBJ])
    
    uc_ncols = int(heatmap_width*uc_nclusters/(uc_nclusters + healthy_nclusters))
    healthy_ncols = heatmap_width - uc_ncols

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
        0         :pert_nrows,
        tree_ncols:tree_ncols+healthy_ncols])
    ax_grampos_uc_pert = fig.add_subplot(gs[
        0                       :pert_nrows,
        tree_ncols+healthy_ncols:tree_ncols+healthy_ncols+uc_ncols])
    ax_grampos_pert = fig.add_subplot(gs[
        0          :pert_nrows,
        tree_ncols:tree_ncols+healthy_ncols+uc_ncols], facecolor='none')
    ax_grampos_healthy_abund = fig.add_subplot(gs[
        pert_nrows:pert_nrows+grampos_nrows,
        tree_ncols:tree_ncols+healthy_ncols])
    ax_grampos_uc_abund = fig.add_subplot(gs[
        pert_nrows              :pert_nrows+grampos_nrows,
        tree_ncols+healthy_ncols:tree_ncols+healthy_ncols+uc_ncols])
    ax_healthy_network = fig.add_subplot(gs[
        ntwk_row_start:ntwk_row_start+ntwk_nrows,
        50            :90], zorder=-50)

    # gram negative
    ax_gramneg_tree = fig.add_subplot(gs[
        pert_nrows:pert_nrows+gramneg_nrows,
        50        :50+tree_branch_ncols])
    ax_gramneg_tree_full = fig.add_subplot(gs[
        pert_nrows :pert_nrows+grampos_nrows,
        50         :50+tree_branch_ncols], facecolor='none')
    ax_gramneg_healthy_pert = fig.add_subplot(gs[
        0            :pert_nrows,
        50+tree_ncols:50+tree_ncols+healthy_ncols])
    ax_gramneg_uc_pert = fig.add_subplot(gs[
        0                          :pert_nrows,
        50+tree_ncols+healthy_ncols:50+tree_ncols+healthy_ncols+uc_ncols])
    ax_gramneg_pert = fig.add_subplot(gs[
        0            :pert_nrows,
        50+tree_ncols:50+tree_ncols+healthy_ncols+uc_ncols], facecolor='none')
    ax_gramneg_healthy_abund = fig.add_subplot(gs[
        pert_nrows   :pert_nrows+gramneg_nrows,
        50+tree_ncols:50+tree_ncols+healthy_ncols])
    ax_gramneg_uc_abund = fig.add_subplot(gs[
        pert_nrows                  :pert_nrows+gramneg_nrows,
        50+tree_ncols+healthy_ncols:50+tree_ncols+healthy_ncols+uc_ncols])
    ax_uc_network = fig.add_subplot(gs[
        ntwk_row_start+ntwk_nrows:100,
        50                       :90], zorder=-50)

    # Plot gram positive subplots
    # ---------------------------
    ax_grampos_tree, grampos_asvname_order = _make_phylogenetic_tree(
        treename=treename, names=grampos_asvnames, asvs=asvs, fig=fig,
        ax=ax_grampos_tree, figlabel='E', figlabelax=ax_grampos_tree_full)
    
    ax_grampos_healthy_abund, grampos_healthy_colorder = _make_cluster_membership_heatmap(
        chainname=chainnamehealthy, ax=ax_grampos_healthy_abund, order=grampos_asvname_order,
        binary=False, fig=fig, make_colorbar=False)

    ax_grampos_uc_abund, grampos_uc_colorder = _make_cluster_membership_heatmap(
        chainname=chainnameuc, ax=ax_grampos_uc_abund, order=grampos_asvname_order,
        binary=False, fig=fig, make_colorbar=False)

    ax_grampos_healthy_pert = _make_perturbation_heatmap(chainname=chainnamehealthy,
        min_bayes_factor=10, ax=ax_grampos_healthy_pert, colorder=grampos_healthy_colorder,
        fig=fig, make_colorbar=False, figlabel='A', render_labels=True)
    ax_grampos_uc_pert = _make_perturbation_heatmap(chainname=chainnameuc,
        min_bayes_factor=10, ax=ax_grampos_uc_pert, colorder=grampos_uc_colorder,
        fig=fig, make_colorbar=False, figlabel='B', render_labels=False)

    ax_grampos_tree = _remove_border(ax_grampos_tree)
    ax_grampos_pert = _remove_border(ax_grampos_pert)
    ax_grampos_pert.set_title('Gram + Bacteria', fontsize=50, pad=80)

    # Plot gram negative subplots
    # ---------------------------
    ax_gramneg_tree, gramneg_asvname_order = _make_phylogenetic_tree(
        treename=treename, names=gramneg_asvnames, asvs=asvs, fig=fig,
        ax=ax_gramneg_tree, figlabel='F', figlabelax=ax_gramneg_tree_full)
    
    ax_gramneg_healthy_abund, gramneg_healthy_colorder = _make_cluster_membership_heatmap(
        chainname=chainnamehealthy, ax=ax_gramneg_healthy_abund, order=gramneg_asvname_order,
        binary=False, fig=fig, make_colorbar=False)

    ax_gramneg_uc_abund, gramneg_uc_colorder = _make_cluster_membership_heatmap(
        chainname=chainnameuc, ax=ax_gramneg_uc_abund, order=gramneg_asvname_order,
        binary=False, fig=fig, make_colorbar=True)

    ax_gramneg_healthy_pert = _make_perturbation_heatmap(chainname=chainnamehealthy,
        min_bayes_factor=10, ax=ax_gramneg_healthy_pert, colorder=gramneg_healthy_colorder,
        fig=fig, make_colorbar=False, figlabel='C', render_labels=True)
    ax_gramneg_uc_pert = _make_perturbation_heatmap(chainname=chainnameuc,
        min_bayes_factor=10, ax=ax_gramneg_uc_pert, colorder=gramneg_uc_colorder,
        fig=fig, make_colorbar=True, figlabel='D', render_labels=False)

    ax_gramneg_tree = _remove_border(ax_gramneg_tree)
    ax_gramneg_pert = _remove_border(ax_gramneg_pert)
    ax_gramneg_tree_full = _remove_border(ax_gramneg_tree_full)
    ax_grampos_tree_full = _remove_border(ax_grampos_tree_full)
    
    ax_gramneg_pert.set_title('Gram - Bacteria', fontsize=50, pad=80)

    # Render the networks below the gram negative
    # -------------------------------------------
    # healthy
    ax_healthy_network = _remove_border(ax_healthy_network)
    arr_man = mpimg.imread('figures/hairball_network_healthy.jpg')
    imagebox = OffsetImage(arr_man, zoom=0.325)
    ab = AnnotationBbox(imagebox, (0.5, 0.475), pad=0, box_alignment=(0.5,0.5))
    ax_healthy_network.add_artist(ab)
    ax_healthy_network.text(x=0.3, y=0.9, s='G', fontsize=50, fontweight='bold',
        transform=ax_healthy_network.transAxes, zorder=50)

    # uc
    ax_uc_network = _remove_border(ax_uc_network)
    arr_man = mpimg.imread('figures/hairball_network_uc.jpg')
    imagebox = OffsetImage(arr_man, zoom=0.325)
    ab = AnnotationBbox(imagebox, (0.5, 0.475), pad=0, box_alignment=(0.5,0.5))
    ax_uc_network.add_artist(ab)
    ax_uc_network.text(x=0.3, y=0.9, s='H', fontsize=50, fontweight='bold',
        transform=ax_uc_network.transAxes, zorder=50)
    
    # Add in the taxonomic key
    suffix_taxa = {'genus': '*',
        'family': '**', 'order': '***', 'class': '****', 'phylum': '*****', 'kingdom': '******'}
    text = '$\\bf{Taxonomy} \\bf{Key}$\n'
    for taxa in suffix_taxa:
        text += '{} - {}\n'.format(suffix_taxa[taxa], taxa)
    fig.text(x=0.875, y=0.45, s=text, fontsize=35)


    fig.subplots_adjust(wspace=0.2, left=0.015, right=0.985, hspace=0.05,
        top=.96, bottom=0.02)

    plt.savefig('output_figures/phylo_clustering_heatmap_gramsplit.pdf')
    plt.close()
    sys.exit()

# Alpha Diversity
# ---------------
def alpha_diversity_figure(ax=None, figlabel=None):
    '''Alpha diversity figure
    '''
    subjset = loaddata(healthy=None)
    
    if ax is None:
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111)

    healthy_found = False
    not_healthy_found = False

    cnt_healthy = {}
    cnt_unhealthy = {}
    val_healthy = {}
    val_unhealthy = {}

    for subj in subjset:
        label = None
        if subj.name in HEALTHY_SUBJECTS:
            color = 'slateblue'
            if not healthy_found:
                label = 'True'
                healthy_found = True
        else:
            color = 'forestgreen'
            if not not_healthy_found:
                label = 'False'
                not_healthy_found = True

        for i,t in enumerate(subj.times):
            aaa = diversity.alpha.normalized_entropy(subj.reads[t])

            if subj.name in HEALTHY_SUBJECTS:
                if t not in cnt_healthy:
                    cnt_healthy[t] = 0
                    val_healthy[t] = []
                cnt_healthy[t] = 1
                val_healthy[t].append(aaa)
            else:
                if t not in cnt_unhealthy:
                    cnt_unhealthy[t] = 0
                    val_unhealthy[t] = []
                cnt_unhealthy[t] = 1
                val_unhealthy[t].append(aaa)

    # Plot unhealthy
    times = np.sort(list(cnt_unhealthy.keys()))
    means_unhealthy = np.zeros(len(times))
    std_unhealthy = np.zeros(len(times))

    means_healthy = np.zeros(len(times))
    std_healthy = np.zeros(len(times))

    for i,t in enumerate(times):
        means_unhealthy[i] = np.mean(val_unhealthy[t])
        std_unhealthy[i] = np.sqrt(unbias_var_estimate(val_unhealthy[t]))

        means_healthy[i] = np.mean(val_healthy[t])
        std_healthy[i] = np.sqrt(unbias_var_estimate(val_healthy[t]))
    
    ax.plot(times, means_unhealthy, marker='o', markersize=4, color='forestgreen', 
        label='Ulcerative Colitis')
    ax.fill_between(times, y1=means_unhealthy-std_unhealthy, y2=means_unhealthy+std_unhealthy,
        alpha=0.2, color='forestgreen')

    ax.plot(times, means_healthy, marker='o', markersize=4, color='slategray', 
        label='Healthy')
    ax.fill_between(times, y1=means_healthy-std_healthy, y2=means_healthy+std_healthy,
        alpha=0.2, color='slategray')


    # ax.errorbar(times, means_unhealthy, yerr=std_unhealthy, ecolor='forestgreen', color='forestgreen', 
    #     label='unhealthy')


        
        # ax.plot(times, vals, marker='o', markersize=4, color=color,
        #     label=label)

    pl.visualization.shade_in_perturbations(ax, perturbations=subjset.perturbations, 
        textsize=15, alpha=0.25)
    for perturbation in subjset.perturbations:
        ax.axvline(x=perturbation.start, color='black', linestyle=':', lw=3)
        ax.axvline(x=perturbation.end, color='black', linestyle=':', linewidth=3)
    ax.legend(loc='upper left')
    ax.set_ylabel('nat', size=20, fontweight='bold')
    ax.set_xlabel('Days', size=20, fontweight='bold')
    ax.set_title('Normalized Entropy', size=15, fontsize=35, fontweight='bold')

    # l = np.arange(times[-1], step=5)
    # axbottom.set_xticks(ticks=)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(13)
        tick.label.set_fontweight('bold')
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(13)
        tick.label.set_fontweight('bold')

    if figlabel is not None:
        ax.text(x = -0.1, y= 1.1,
            s=figlabel, fontsize=30, fontweight='bold',
            transform=ax.transAxes)
    # plt.savefig(BASEPATH + 'alpha_diversity_figure.pdf')
    # plt.close()

def alpha_diversity_figure_boxplot(ax=None, figlabel=None):
    '''Alpha diversity figure
    '''
    subjset = loaddata(healthy=None)
    
    if ax is None:
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111)

    healthy_found = False
    not_healthy_found = False

    cnt_healthy = {}
    cnt_unhealthy = {}
    val_healthy = {}
    val_unhealthy = {}

    for subj in subjset:
        label = None
        if subj.name in HEALTHY_SUBJECTS:
            color = 'slateblue'
            if not healthy_found:
                label = 'True'
                healthy_found = True
        else:
            color = 'forestgreen'
            if not not_healthy_found:
                label = 'False'
                not_healthy_found = True

        for i,t in enumerate(subj.times):
            aaa = diversity.alpha.normalized_entropy(subj.reads[t])

            if subj.name in HEALTHY_SUBJECTS:
                if t not in cnt_healthy:
                    cnt_healthy[t] = 0
                    val_healthy[t] = []
                cnt_healthy[t] = 1
                val_healthy[t].append(aaa)
            else:
                if t not in cnt_unhealthy:
                    cnt_unhealthy[t] = 0
                    val_unhealthy[t] = []
                cnt_unhealthy[t] = 1
                val_unhealthy[t].append(aaa)

    # Plot unhealthy
    times = np.sort(list(cnt_unhealthy.keys()))
    val_unhealthy_ = []
    val_healthy_ = []
    times_unhealthy_ = []

    # Set the right indexes for the perturbations (converting from time to index)
    for perturbation in subjset.perturbations:
        start_t = perturbation.start
        end_t = perturbation.end

        startidx = np.searchsorted(times, start_t)
        endidx = np.searchsorted(times, end_t)
        perturbation.start = startidx - 0.5
        perturbation.end = endidx + 0.5

    vals = []
    types = []
    times_ = []

    for i,t in enumerate(times):

        vals += val_unhealthy[t]
        types += ['Ulcerative Colitis']*len(val_unhealthy[t])
        vals += val_healthy[t]
        types += ['Healthy']*len(val_healthy[t])
        times_ += [t]* (len(val_unhealthy[t] + val_healthy[t]))

    vun = np.asarray(vals).reshape(-1,1)
    tun = np.asarray(times_).reshape(-1,1)
    types = np.asarray(types).reshape(-1,1)
    M = np.hstack((tun,vun, types))
    a = pd.DataFrame(M, columns=['Times (d)', 'Values', 'Dataset'])

    a[['Times (d)', 'Values']] = a[['Times (d)', 'Values']].apply(pd.to_numeric)

    print(a.head())

    sns.boxplot(data=a, x='Times (d)', y='Values', hue='Dataset')

    pl.visualization.shade_in_perturbations(ax, perturbations=subjset.perturbations, 
        textsize=15, alpha=0.0)
    for perturbation in subjset.perturbations:
        ax.axvline(x=perturbation.start, color='black', linestyle=':', lw=3)
        ax.axvline(x=perturbation.end, color='black', linestyle=':', linewidth=3)
    ax.legend(loc='upper left')
    ax.set_ylabel('nat', size=20, fontweight='bold')
    ax.set_xlabel('Days', size=20, fontweight='bold')
    ax.set_title('Normalized Entropy', size=15, fontsize=35, fontweight='bold')

    aaa = np.arange(0, len(times), step= int(len(times)/15))
    ax.set_xticks(ticks=aaa)
    ax.set_xticklabels(labels=times[aaa])
    



    if figlabel is not None:
        ax.text(x = -0.1, y= 1.1,
            s=figlabel, fontsize=30, fontweight='bold',
            transform=ax.transAxes)
    # plt.savefig(BASEPATH + 'alpha_diversity_figure_boxplots.pdf')
    # plt.close()

def alpha_diversity_mean_std(ax=None, figlabel=None):
    '''Alpha diversity figure
    '''
    subjset = loaddata(healthy=None)
    subjset_innoc = loadinoculum()
    
    if ax is None:
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111)
        
    val_healthy = {}
    val_unhealthy = {}
    val_innoc_healthy = diversity.alpha.normalized_entropy(subjset_innoc['healthy'].reads[0])
    val_innoc_uc = diversity.alpha.normalized_entropy(subjset_innoc['ulcerative colitis'].reads[0])

    for subj in subjset:
        for t in subj.times:
            aaa = diversity.alpha.normalized_entropy(subj.reads[t])
            if subj.name in HEALTHY_SUBJECTS:
                if t not in val_healthy:
                    val_healthy[t] = []
                val_healthy[t].append(aaa)
            else:
                if t not in val_unhealthy:
                    val_unhealthy[t] = []
                val_unhealthy[t].append(aaa)

    times = np.sort(list(val_unhealthy.keys()))
    means_unhealthy = np.zeros(len(times))
    std_unhealthy = np.zeros(len(times))

    means_healthy = np.zeros(len(times))
    std_healthy = np.zeros(len(times))

    for i,t in enumerate(times):
        means_unhealthy[i] = np.mean(val_unhealthy[t])
        std_unhealthy[i] = np.sqrt(unbias_var_estimate(val_unhealthy[t]))

        means_healthy[i] = np.mean(val_healthy[t])
        std_healthy[i] = np.sqrt(unbias_var_estimate(val_healthy[t]))

    colors = sns.color_palette('muted')
    colors_healthy = colors[0]
    colors_unhealthy = colors[1]

    times_idxs = np.arange(len(times))

    print('times:', times)

    times_idxs_healthy = times_idxs-(0.25/2)
    times_idxs_unhealthy = times_idxs+(0.25/2)
    
    # Plot the lines
    ax.errorbar(times_idxs_healthy, means_healthy, std_healthy, 
        ecolor=colors_healthy, color=colors_healthy, capsize=3, fmt='none')
    ax.plot(times_idxs_healthy, means_healthy, marker='o', color=colors_healthy, linewidth=0, 
        markersize=5)

    ax.errorbar(times_idxs_unhealthy, means_unhealthy, std_unhealthy, 
        ecolor=colors_unhealthy, color=colors_unhealthy, capsize=3, fmt='none')
    ax.plot(times_idxs_unhealthy, means_unhealthy, marker='o', color=colors_unhealthy, linewidth=0, 
        markersize=5)

    # Add in the Inoculum
    ax.plot([-1.5], val_innoc_healthy, marker='*', color=colors_healthy, markersize=10)
    ax.plot([-1.5], val_innoc_uc, marker='*', color=colors_unhealthy, markersize=10)

    # Set the xticklabels
    locs = np.arange(0, len(times),step=10)
    ticklabels= times[locs]
    plt.xticks(ticks=locs, labels=ticklabels)

    # Plot the perturbation markers
    for perturbation in subjset.perturbations:
        start_t = perturbation.start
        end_t = perturbation.end

        startidx = np.searchsorted(times, start_t)
        endidx = np.searchsorted(times, end_t)
        perturbation.start = startidx - 0.5
        perturbation.end = endidx + 0.5
    pl.visualization.shade_in_perturbations(ax, perturbations=subjset.perturbations, textsize=18, 
        alpha=0)
    for perturbation in subjset.perturbations:
        ax.axvline(x=perturbation.start, color='black', linestyle='--', lw=2)
        ax.axvline(x=perturbation.end, color='black', linestyle='--', linewidth=2)



    # Set the ticks to be bold
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
        # tick.label.set_fontweight('bold')
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(16)
        # tick.label.set_fontweight('bold')

    # Set the labels
    ax.set_ylabel('nat', size=20, fontweight='bold')
    ax.set_xlabel('Time (days)', size=20, fontweight='bold')

    # Make legend
    axlegend = fig.add_subplot(111, facecolor='none')
    handles = []
    l = mlines.Line2D([],[], color=colors_healthy,
        linestyle='-', label='Healthy')
    handles.append(l)
    l = mlines.Line2D([],[], color=colors_unhealthy,
        linestyle='-', label='Ulcerative Colitis')
    handles.append(l)
    l = mlines.Line2D([], [], color='black', marker='*', 
        label='Inoculum', linestyle='none', markersize=10)
    handles.append(l)
    lgnd2 = plt.legend(handles=handles, fontsize=18)
    axlegend.add_artist(lgnd2)
    axlegend.spines['top'].set_visible(False)
    axlegend.spines['bottom'].set_visible(False)
    axlegend.spines['left'].set_visible(False)
    axlegend.spines['right'].set_visible(False)
    axlegend.xaxis.set_major_locator(plt.NullLocator())
    axlegend.xaxis.set_minor_locator(plt.NullLocator())
    axlegend.yaxis.set_major_locator(plt.NullLocator())
    axlegend.yaxis.set_minor_locator(plt.NullLocator())

    plt.subplots_adjust(bottom=0.18)

    # caption = 'Mean and standard deviation of normalized entropy measure within each consortium.'
    # axlegend.text(0.5, -0.2, caption, horizontalalignment='center', fontsize=17)

    plt.savefig(BASEPATH + 'alpha_diversity_mean_std.pdf')
    plt.savefig(BASEPATH + 'alpha_diversity_mean_std.png')
    # plt.show()
    plt.close()


# Beta Diversity
# --------------
def beta_diversity_figure(axleft=None, axright=None, axcenter=None, figlabel=None):
    '''beta diversity figure

    Project into the within and between difference
    '''
    # First we need to add all of the data into a matrix and record the labels
    subjset = loaddata(None)
    inoculum_subjset = loadinoculum()

    data = None
    labels = []
    labels_float = {}
    # for subj in [subjset.iloc(0)]:
    for subj in subjset:
        for t in subj.times:
            ts = str(float(t)).replace('.5', 'PM').replace('.0', 'AM')
            labels.append('{}-{}'.format(subj.name, ts))
            labels_float[labels[-1]] = (subj, t)
        
        d = subj.matrix()['raw'].T
        if data is None:
            data = d
        else:
            data = np.vstack((data,d))
    
    for subj in inoculum_subjset:
        labels.append('inoculum {}'.format(subj.name))
        labels_float[labels[-1]] = 1000
        m = subj.matrix()['raw'].T
        data = np.vstack((data, m))

    # Perform pcoa on the distance matrix
    bc_dm = skbio.diversity.beta_diversity("braycurtis", data, labels)
    bc_pcoa = skbio.stats.ordination.pcoa(bc_dm)

    # print('proportion explained')
    # print(bc_pcoa.proportion_explained)
    # print('eigvals')
    # print(bc_pcoa.eigvals)
    # print('samples')
    # print(bc_pcoa.samples)

    data = bc_pcoa.samples.to_numpy()

    if axleft is None:
        fig = plt.figure(figsize=(16,7))
        axright = fig.add_subplot(121)
        axleft = fig.add_subplot(122)

    colors = sns.color_palette('muted')
    colorshealthy = colors[0]
    colorsunhealthy = colors[1]

    colorinoculumhealthy = colors[0]
    colorinoculumuc = colors[1]

    x_healthy = None
    y_healthy = None
    x_uc = None
    y_uc = None

    xs = []
    ys = []
    cs = []
    # ecs = []

    # Make the (xs,ys) for each subject and assign the colors
    subj_colors = {}
    subj_perts = {}

    for subj in subjset:
        if subj.name in HEALTHY_SUBJECTS:
            subj_colors[subj.name] = colorshealthy
        else:
            subj_colors[subj.name] = colorsunhealthy
        subj_perts[subj.name] = [[[],[]] for i in range(7)] # 1 colonization, 3 perts, 3 post perts


    for row in range(data.shape[0]):
        if 'inoculum' in labels[row]:
            if 'healthy' in labels[row]:
                x_healthy = data[row,0]
                y_healthy = data[row,1]
            else:
                x_uc = data[row,0]
                y_uc = data[row,1]
        else:
            subj,t = labels_float[labels[row]]
            mi = _set_type_of_point_bc(t, subj)

            subj_perts[subj.name][mi][0].append(data[row,0])
            subj_perts[subj.name][mi][1].append(data[row,1])


    PERT_MARKERS = ['+', 'd', 'o', 's', 'v', 'x', 'X']
    # PERT_MARKERS = ['+', 'd', 'v', 's', 'v', 'x', 'v']
    INOCULUM_MARKER = '*'
    for ix, ax in enumerate([axright, axleft]):
        print('ix', ix)

        # Plot the points
        for subj in subjset:
            for mi in range(len(subj_perts[subj.name])):
                xs = subj_perts[subj.name][mi][0]
                ys = subj_perts[subj.name][mi][1]

                ax.plot(xs, ys, PERT_MARKERS[mi], color=subj_colors[subj.name], 
                    markersize=6, alpha=0.75)
        
        # Plot the inoculum
        ax.plot([x_healthy], [y_healthy], INOCULUM_MARKER, color=colorinoculumhealthy,
            markersize=15, alpha=0.75)
        ax.plot([x_uc], [y_uc], INOCULUM_MARKER, color=colorinoculumuc,
            markersize=15, alpha=0.75)

        # if ix == 0:
        #     # Draw rec around unhealthy
        #     rect = patches.Rectangle((-0.38, -0.18), 0.38, 0.35,
        #         linewidth=1, edgecolor='red', facecolor='none')
        #     ax.add_patch(rect)

            # Make the legends
            # lgnd1 = plt.legend(healthy_lines[1], healthy_lines[0], 
            #     title='$\\bf{Healthy}$', bbox_to_anchor=(1.05, 1))
            # lgnd2 = plt.legend(unhealthy_lines[1], unhealthy_lines[0], 
            #     title='$\\bf{Unhealthy}$', bbox_to_anchor=(1.05, 0.6))
            # lgnd3 = plt.legend(pert_lines[1], pert_lines[0], 
            #     title='$\\bf{Perturbations}$', bbox_to_anchor=(1.05, 0.2))

            # ax.add_artist(lgnd1)
            # ax.add_artist(lgnd2)
            # ax.add_artist(lgnd3)


    axleft.set_xlim(left=-.38, right=0)
    axleft.set_ylim(bottom=-0.18, top=-0.18+0.35)

    if axcenter is None:
        axcenter = fig.add_subplot(111, facecolor='none')
    axcenter.spines['top'].set_visible(False)
    axcenter.spines['bottom'].set_visible(False)
    axcenter.spines['left'].set_visible(False)
    axcenter.spines['right'].set_visible(False)
    axcenter.xaxis.set_major_locator(plt.NullLocator())
    axcenter.xaxis.set_minor_locator(plt.NullLocator())
    axcenter.yaxis.set_major_locator(plt.NullLocator())
    axcenter.yaxis.set_minor_locator(plt.NullLocator())

    if figlabel is not None:
        axcenter.text(x = -0.1, y= 1.1,
            s=figlabel, fontsize=30, fontweight='bold',
            transform=axcenter.transAxes)

    # Make the legend for the right side
    # Ulcerative colitis
    handles = []
    l = mlines.Line2D([],[], color=colorshealthy,
        linestyle='-', label='Healthy', linewidth=5)
    handles.append(l)
    l = mlines.Line2D([],[], color=colorsunhealthy,
        linestyle='-', label='Ulcerative Colitis', linewidth=5)
    handles.append(l)
    lgnd2 = plt.legend(handles=handles, title='$\\bf{Dataset}$', 
        bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.,
        fontsize=17, title_fontsize=18)
    axcenter.add_artist(lgnd2)

    # Perturbations
    handles = []
    l = mlines.Line2D([],[],
        marker=INOCULUM_MARKER,
        markeredgecolor='black',
        markerfacecolor='none',
        linestyle='none', 
        markersize=15,
        label='Inoculum')
    handles.append(l)
    l = mlines.Line2D([],[],
        marker=PERT_MARKERS[0],
        markeredgecolor='black',
        markerfacecolor='none',
        linestyle='none', 
        markersize=15,
        label='Colonization')
    handles.append(l)
    for pidx in range(1,7):

        pert_name = subjset.perturbations[(pidx-1)//2].name
        # If pidx-1 is odd, then it is post perturbation
        if (pidx-1)%2 == 1:
            pert_name = 'Post-' + pert_name

        l = mlines.Line2D([],[],
            marker=PERT_MARKERS[pidx],
            markeredgecolor='black',
            markerfacecolor='none',
            linestyle='none', 
            markersize=15,
            label=pert_name)
        handles.append(l)
    lgnd3 = plt.legend(handles=handles, title='$\\bf{Markers}$', 
        bbox_to_anchor=(1.05, 0.0), loc='lower left', borderaxespad=0.,
        fontsize=17, title_fontsize=18)
    axcenter.add_artist(lgnd3)

    axcenter.set_xlabel('PC1: {:.3f}'.format(bc_pcoa.proportion_explained[0]),
        fontsize=20, fontweight='bold')
    axcenter.xaxis.set_label_coords(0.5,-0.08)

    axright.set_ylabel('PC2: {:.3f}'.format(bc_pcoa.proportion_explained[1]),
        fontsize=20, fontweight='bold')

    # mark_inset(parent_axes=axright, inset_axes=axleft, loc1a=2, loc1b=1, 
    #     loc2a=3, loc2b=4, fc='none', ec='crimson')
    axleft.spines['top'].set_color('gray')
    axleft.spines['bottom'].set_color('gray')
    axleft.spines['left'].set_color('gray')
    axleft.spines['right'].set_color('gray')
    axleft.tick_params(axis='both', color='gray')

    rect =[patches.Rectangle(xy=(-0.38, -0.18), width=0.38, height=.35)]

    pc = PatchCollection(rect, facecolor='none', alpha=0.8, edgecolor='gray')
    axright.add_collection(pc)

    axleft.set_xticks([-0.35, -0.25, -0.15, -0.05])
    axleft.set_yticks([-0.15, -0.05, 0.05, 0.15])
    for tick in axleft.yaxis.get_major_ticks():
        tick.label.set_fontsize(18)
        # tick.label.set_fontweight('bold')
    for tick in axleft.xaxis.get_major_ticks():
        tick.label.set_fontsize(18)
        # tick.label.set_fontweight('bold')

    axright.set_yticks([-0.4, -0.2, 0, 0.2, 0.4])
    for tick in axright.yaxis.get_major_ticks():
        tick.label.set_fontsize(18)
        # tick.label.set_fontweight('bold')
    for tick in axright.xaxis.get_major_ticks():
        tick.label.set_fontsize(18)
        # tick.label.set_fontweight('bold')

    fig.subplots_adjust(left=0.09, right=0.775, top=0.86)

    axright.set_title('A', fontsize=25, fontweight='bold')
    axleft.set_title('B', fontsize=25, fontweight='bold')

    # # Label the left and right side
    # axcenter.text(x=0., y=1.01, s='A', fontsize=25, fontweight='bold')
    # axcenter.text(x=0.545, y=1.01, s='B', fontsize=25, fontweight='bold')

    # bc_pcoa.plot()
    plt.savefig(BASEPATH + 'pcoa_braycurtis_w_zoom.pdf')
    plt.savefig(BASEPATH + 'pcoa_braycurtis_w_zoom.png')
    plt.close()
    # plt.show()

def beta_diversity_figure_no_zoom_2_colors(axleft=None, axright=None, axcenter=None, figlabel=None):
    '''beta diversity figure

    Project into the within and between difference
    '''
    # First we need to add all of the data into a matrix and record the labels
    subjset = loaddata(None)
    inoculum_subjset = loadinoculum()

    data = None
    labels = []
    labels_float = {}
    # for subj in [subjset.iloc(0)]:
    for subj in subjset:
        for t in subj.times:
            ts = str(float(t)).replace('.5', 'PM').replace('.0', 'AM')
            labels.append('{}-{}'.format(subj.name, ts))
            labels_float[labels[-1]] = (subj, t)
        
        d = subj.matrix()['raw'].T
        if data is None:
            data = d
        else:
            data = np.vstack((data,d))
    
    for subj in inoculum_subjset:
        labels.append('inoculum {}'.format(subj.name))
        labels_float[labels[-1]] = 1000
        m = subj.matrix()['raw'].T
        data = np.vstack((data, m))

    # Perform pcoa on the distance matrix
    bc_dm = skbio.diversity.beta_diversity("braycurtis", data, labels)
    bc_pcoa = skbio.stats.ordination.pcoa(bc_dm)

    # print('proportion explained')
    # print(bc_pcoa.proportion_explained)
    # print('eigvals')
    # print(bc_pcoa.eigvals)
    # print('samples')
    # print(bc_pcoa.samples)

    data = bc_pcoa.samples.to_numpy()

    if axleft is None:
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111)

    colors = sns.color_palette('muted')
    colorshealthy = colors[0]
    colorsunhealthy = colors[1]

    colorinoculumhealthy = colors[0]
    colorinoculumuc = colors[1]

    x_healthy = None
    y_healthy = None
    x_uc = None
    y_uc = None

    xs = []
    ys = []
    cs = []
    # ecs = []

    # Make the (xs,ys) for each subject and assign the colors
    subj_colors = {}
    subj_perts = {}

    for subj in subjset:
        if subj.name in HEALTHY_SUBJECTS:
            subj_colors[subj.name] = colorshealthy
        else:
            subj_colors[subj.name] = colorsunhealthy
        subj_perts[subj.name] = [[[],[]] for i in range(7)] # 1 colonization, 3 perts, 3 post perts


    for row in range(data.shape[0]):
        if 'inoculum' in labels[row]:
            if 'healthy' in labels[row]:
                x_healthy = data[row,0]
                y_healthy = data[row,1]
            else:
                x_uc = data[row,0]
                y_uc = data[row,1]
        else:
            subj,t = labels_float[labels[row]]
            mi = _set_type_of_point_bc(t, subj)

            subj_perts[subj.name][mi][0].append(data[row,0])
            subj_perts[subj.name][mi][1].append(data[row,1])


    PERT_MARKERS = ['+', 'd', 'o', 's', 'v', 'x', 'X']
    # PERT_MARKERS = ['+', 'd', 'v', 's', 'v', 'x', 'v']
    INOCULUM_MARKER = '*'

    # Plot the points
    for subj in subjset:
        for mi in range(len(subj_perts[subj.name])):
            xs = subj_perts[subj.name][mi][0]
            ys = subj_perts[subj.name][mi][1]

            ax.plot(xs, ys, PERT_MARKERS[mi], color=subj_colors[subj.name], 
                markersize=6, alpha=0.75)
    
    # Plot the inoculum
    ax.plot([x_healthy], [y_healthy], INOCULUM_MARKER, color=colorinoculumhealthy,
        markersize=15, alpha=0.75)
    ax.plot([x_uc], [y_uc], INOCULUM_MARKER, color=colorinoculumuc,
        markersize=15, alpha=0.75)


        # if ix == 0:
        #     # Draw rec around unhealthy
        #     rect = patches.Rectangle((-0.38, -0.18), 0.38, 0.35,
        #         linewidth=1, edgecolor='red', facecolor='none')
        #     ax.add_patch(rect)

            # Make the legends
            # lgnd1 = plt.legend(healthy_lines[1], healthy_lines[0], 
            #     title='$\\bf{Healthy}$', bbox_to_anchor=(1.05, 1))
            # lgnd2 = plt.legend(unhealthy_lines[1], unhealthy_lines[0], 
            #     title='$\\bf{Unhealthy}$', bbox_to_anchor=(1.05, 0.6))
            # lgnd3 = plt.legend(pert_lines[1], pert_lines[0], 
            #     title='$\\bf{Perturbations}$', bbox_to_anchor=(1.05, 0.2))

            # ax.add_artist(lgnd1)
            # ax.add_artist(lgnd2)
            # ax.add_artist(lgnd3)


    # axleft.set_xlim(left=-.38, right=0)
    # axleft.set_ylim(bottom=-0.18, top=-0.18+0.35)

    # if axcenter is None:
    #     axcenter = fig.add_subplot(111, facecolor='none')
    # axcenter.spines['top'].set_visible(False)
    # axcenter.spines['bottom'].set_visible(False)
    # axcenter.spines['left'].set_visible(False)
    # axcenter.spines['right'].set_visible(False)
    # axcenter.xaxis.set_major_locator(plt.NullLocator())
    # axcenter.xaxis.set_minor_locator(plt.NullLocator())
    # axcenter.yaxis.set_major_locator(plt.NullLocator())
    # axcenter.yaxis.set_minor_locator(plt.NullLocator())

    if figlabel is not None:
        ax.text(x = -0.1, y= 1.1,
            s=figlabel, fontsize=30, fontweight='bold',
            transform=ax.transAxes)

    # Make the legend for the right side
    # Ulcerative colitis
    handles = []
    l = mlines.Line2D([],[], color=colorshealthy,
        linestyle='-', label='Healthy')
    handles.append(l)
    l = mlines.Line2D([],[], color=colorsunhealthy,
        linestyle='-', label='Ulcerative Colitis')
    handles.append(l)
    lgnd2 = plt.legend(handles=handles, title='$\\bf{Dataset}$', 
        bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax.add_artist(lgnd2)

    # Perturbations
    handles = []
    l = mlines.Line2D([],[],
        marker=INOCULUM_MARKER,
        markeredgecolor='black',
        markerfacecolor='none',
        linestyle='none', 
        label='Inoculum')
    handles.append(l)
    l = mlines.Line2D([],[],
        marker=PERT_MARKERS[0],
        markeredgecolor='black',
        markerfacecolor='none',
        linestyle='none', 
        label='Colonization')
    handles.append(l)
    for pidx in range(1,7):

        pert_name = subjset.perturbations[(pidx-1)//2].name
        # If pidx-1 is odd, then it is post perturbation
        if (pidx-1)%2 == 1:
            pert_name = 'Post ' + pert_name

        l = mlines.Line2D([],[],
            marker=PERT_MARKERS[pidx],
            markeredgecolor='black',
            markerfacecolor='none',
            linestyle='none', 
            label=pert_name)
        handles.append(l)
    lgnd3 = plt.legend(handles=handles, title='$\\bf{Markers}$', 
        bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0.)
    ax.add_artist(lgnd3)

    ax.set_title('Bray-Curtis PCoA', fontsize=35, fontweight='bold')
    ax.set_xlabel('PC1: {:.3f}'.format(bc_pcoa.proportion_explained[0]),
        fontsize=20, fontweight='bold')
    ax.set_ylabel('PC2: {:.3f}'.format(bc_pcoa.proportion_explained[1]),
        fontsize=20, fontweight='bold')
    ax.xaxis.set_label_coords(0.5,-0.08)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(13)
        tick.label.set_fontweight('bold')
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(13)
        tick.label.set_fontweight('bold')

    # axleft.set_facecolor('whitesmoke')
    # axright.set_facecolor('whitesmoke')

    fig.subplots_adjust(left=0.12, right=0.79)

    # bc_pcoa.plot()
    plt.savefig(BASEPATH + 'pcoa_braycurtis_no_zoom.pdf')
    # plt.show()


# Data Figure
# -----------
def data_figure_rel_and_qpcr(horizontal):
    '''Plot the relative abundance and the qpcr plot above it
    '''
    TAXLEVEL = 'family'
    CUTOFF_FRAC_ABUNDANCE = 0.01
    global DATA_FIGURE_COLORS
    global XKCD_COLORS_IDX

    # Make the dataframs for healthy and unhealthy
    df_healthy, asvname_map_healthy = _make_full_df(True, TAXLEVEL, CUTOFF_FRAC_ABUNDANCE)
    df_unhealthy, asvname_map_unhealthy = _make_full_df(False, TAXLEVEL, CUTOFF_FRAC_ABUNDANCE)

    # Set the colors from most to least abundant - only consider healthy
    M = df_healthy.to_numpy()
    a = np.sum(M, axis=1)
    idxs = np.argsort(a)[::-1] # reverse the indexes so it goes from largest to smallest

    asvname_map_keys = list(asvname_map_healthy.keys())
    print('Label, beginning')
    for idx in idxs:
        label = df_healthy.index[idx]
        color = XKCD_COLORS[XKCD_COLORS_IDX]
        XKCD_COLORS_IDX += 1
        DATA_FIGURE_COLORS[label] = color
        print(label)

    if horizontal:
        fig = plt.figure(figsize=(34,12))

        squeeze = 2
        gs = fig.add_gridspec(9,40*squeeze)

        axqpcr1 = fig.add_subplot(gs[2:4,1*squeeze:14*squeeze])
        axrel1 = fig.add_subplot(gs[4:8,1*squeeze:14*squeeze])
        axpert1 = fig.add_subplot(gs[2:8,1*squeeze:14*squeeze], facecolor='none')
        axinoculum1 = fig.add_subplot(gs[4:8,0])
    else:
        fig = plt.figure(figsize=(13,30))

        squeeze = 2
        gs = fig.add_gridspec(19,15*squeeze)

        axqpcr1 = fig.add_subplot(gs[:2,1*squeeze:14*squeeze])
        axrel1 = fig.add_subplot(gs[2:6,1*squeeze:14*squeeze])
        axpert1 = fig.add_subplot(gs[:6,1*squeeze:14*squeeze], facecolor='none')
        axinoculum1 = fig.add_subplot(gs[2:6,0])

    max_qpcr_value1 = _data_figure_rel_and_qpcr(True, TAXLEVEL=TAXLEVEL,
        df=df_healthy, CUTOFF_FRAC_ABUNDANCE=CUTOFF_FRAC_ABUNDANCE,
        axqpcr=axqpcr1, axrel=axrel1, axpert=axpert1,
        axinoculum=axinoculum1, make_ylabels=True,
        figlabelinoculum='D', figlabelqpcr='B', figlabelrel='E',
        make_legend=False, asvname_map=asvname_map_healthy)

    if horizontal:
        axqpcr2 = fig.add_subplot(gs[2:4,17*squeeze:30*squeeze])
        axrel2 = fig.add_subplot(gs[4:8,17*squeeze:30*squeeze])
        axpert2 = fig.add_subplot(gs[2:8,17*squeeze:30*squeeze], facecolor='none')
        axinoculum2 = fig.add_subplot(gs[4:8,16*squeeze])
    else:
        axqpcr2 = fig.add_subplot(gs[7:2+7,1*squeeze:14*squeeze])
        axrel2 = fig.add_subplot(gs[2+7:6+7,1*squeeze:14*squeeze])
        axpert2 = fig.add_subplot(gs[7:6+7,1*squeeze:14*squeeze], facecolor='none')
        axinoculum2 = fig.add_subplot(gs[2+7:6+7,0])

    max_qpcr_value2 = _data_figure_rel_and_qpcr(False, TAXLEVEL=TAXLEVEL, 
        df=df_unhealthy, CUTOFF_FRAC_ABUNDANCE=CUTOFF_FRAC_ABUNDANCE,
        axqpcr=axqpcr2, axrel=axrel2, axpert=axpert2,
        axinoculum=axinoculum2, make_ylabels=True,
        figlabelinoculum='F', figlabelqpcr='C', figlabelrel='G',
        make_legend=False, asvname_map=asvname_map_unhealthy)

    # Set the same max and min value for the qpcr measurements
    max_qpcr_value = np.max([max_qpcr_value1, max_qpcr_value2])
    axqpcr1.set_ylim(bottom=1e9, top=max_qpcr_value*(1.20))
    axqpcr2.set_ylim(bottom=1e9, top=max_qpcr_value*(1.20))
    axqpcr1.set_yscale('log')
    axqpcr2.set_yscale('log')

    axqpcr1.set_yticks([1e10, 1e11])
    for tick in axqpcr1.yaxis.get_major_ticks():
        tick.label.set_fontsize(24)

    axqpcr2.set_yticks([1e10, 1e11])
    for tick in axqpcr2.yaxis.get_major_ticks():
        tick.label.set_fontsize(24)

    # Make animation at the top
    if horizontal:
        axanimation = fig.add_subplot(gs[0, 7*squeeze:35*squeeze], facecolor='none')
        axfigures = fig.add_subplot(gs[0, :], facecolor='none')
        _data_figure_experiment_animation(ax=axanimation, axfigures=axfigures, figlabel='A')


    # Make the legend
    if horizontal:
        axlegend1 = fig.add_subplot(gs[2:8, 33*squeeze: 39*squeeze], facecolor='none')
    else:
        axlegend1 = fig.add_subplot(gs[13:18, 1*squeeze: 13*squeeze], facecolor='none')
    _data_figure_rel_and_qpcr_legend(axlegend=axlegend1, TAXLEVEL=TAXLEVEL,
        CUTOFF_FRAC_ABUNDANCE=CUTOFF_FRAC_ABUNDANCE)

    axpert1.spines['top'].set_visible(False)
    axpert1.spines['bottom'].set_visible(False)
    axpert1.spines['left'].set_visible(False)
    axpert1.spines['right'].set_visible(False)
    axpert1.xaxis.set_major_locator(plt.NullLocator())
    axpert1.xaxis.set_minor_locator(plt.NullLocator())
    axpert1.yaxis.set_major_locator(plt.NullLocator())
    axpert1.yaxis.set_minor_locator(plt.NullLocator())

    axpert2.spines['top'].set_visible(False)
    axpert2.spines['bottom'].set_visible(False)
    axpert2.spines['left'].set_visible(False)
    axpert2.spines['right'].set_visible(False)
    axpert2.xaxis.set_major_locator(plt.NullLocator())
    axpert2.xaxis.set_minor_locator(plt.NullLocator())
    axpert2.yaxis.set_major_locator(plt.NullLocator())
    axpert2.yaxis.set_minor_locator(plt.NullLocator())


    if horizontal:
        fig.subplots_adjust( wspace=0.58, left=0.05, right=0.95, top=0.925, bottom=.075, hspace=0.8)
    else:
        fig.subplots_adjust( wspace=0.58, left=0.075, right=0.955, top=0.925, bottom=.075, hspace=1.0)
    # plt.show()
    plt.savefig(BASEPATH + 'datafigure_rel_horizonal{}.pdf'.format(horizontal))
    plt.savefig(BASEPATH + 'datafigure_rel_horizonal{}.png'.format(horizontal))
    plt.close()

def _data_figure_experiment_animation(ax, axfigures, figlabel):
    subjset_real = pl.base.SubjectSet.load(DATAPATH)
    times = []
    for subj in subjset_real:
        times = np.append(times, subj.times)
    times = np.sort(np.unique(times))
    print(times)

    y = [-0.1 for t in times]

    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_minor_locator(plt.NullLocator())

    axfigures.spines['top'].set_visible(False)
    axfigures.spines['bottom'].set_visible(False)
    axfigures.spines['left'].set_visible(False)
    axfigures.spines['right'].set_visible(False)
    axfigures.xaxis.set_major_locator(plt.NullLocator())
    axfigures.xaxis.set_minor_locator(plt.NullLocator())
    axfigures.yaxis.set_major_locator(plt.NullLocator())
    axfigures.yaxis.set_minor_locator(plt.NullLocator())
    axfigures.set_xlim(0,1)
    axfigures.set_ylim(0,1)

    markerline, stemlines, baseline = ax.stem(times, y, linefmt='none')
    baseline.set_color('black')
    markerline.set_color('black')
    markerline.set_markersize(5)

    x = np.arange(0, np.max(times), step=10)
    labels = ['Day {}'.format(int(t)) for t in x]
    for ylim in [0.15, -0.35]:
        y = [ylim for t in x]
        markerline, stemlines, baseline = ax.stem(x, y)
        stemlines = [stemline.set_color('black') for stemline in stemlines]
        baseline.set_color('black')
        markerline.set_color('none')
    for i in range(len(labels)):
        label = labels[i]
        xpos = x[i] 
        ax.text(xpos, 0.-.50, label, horizontalalignment='center', fontsize=28)
    x = np.arange(0,np.max(times),2)
    for ylim in [0.07, -0.07]:
        y = [ylim for t in x]
        markerline, stemlines, baseline = ax.stem(x, y)
        stemlines = [stemline.set_color('black') for stemline in stemlines]
        baseline.set_color('black')
        markerline.set_color('none')

    for perturbation in subjset_real.perturbations:
        name = perturbation.name
        # lines += [start,end]
        # ax.text(start, 0.33, 'Start {}'.format(name), horizontalalignment='center')
        ax.text((perturbation.end+perturbation.start)/2, 0.17, name.capitalize(), horizontalalignment='center',
            fontsize=28)

        # ax.arrow(start, 0.3, 0, -0.27, length_includes_head=True, head_width=0.25, head_length=0.05)
        # ax.arrow(end, 0.3, 0, -0.27, length_includes_head=True, head_width=0.25, head_length=0.05)
        starts = np.asarray([perturbation.start])
        ends = np.asarray([perturbation.end])
        ax.barh(y=[0 for i in range(len(starts))], width=ends-starts, height=0.1, left=starts, color='darkgrey')

    ax.text(0, 0.17, 'Colonization', horizontalalignment='center', fontsize=28)

    # ax.arrow(0, 0.17, 0, -0.14, length_includes_head=True, head_width=0.25, head_length=0.05)

    # stool collection
    xpos = np.max(times)* 1.05
    y = 0.05
    ax.scatter([xpos], [y], c='black', s=25)
    ax.text(xpos+1, y, 'Fecal Sample Collection', horizontalalignment='left', fontsize=28, 
        verticalalignment='center')

    ax.set_ylim(-0.15,0.4)

    if figlabel is not None:
        ax.text(x=-0.02, y=1.02, s=figlabel, fontsize=30, fontweight='bold', 
            transform=ax.transAxes)

    # Draw figures ontop of words
    # ---------------------------
    # Load man
    arr_man = mpimg.imread('figures/man.jpg')
    imagebox = OffsetImage(arr_man, zoom=1)
    ab = AnnotationBbox(imagebox, (0, 0), pad=0, box_alignment=(0.5,0.2))
    axfigures.add_artist(ab)

    # Load poop
    arr_poop = mpimg.imread('figures/poop.jpg')
    imagebox = OffsetImage(arr_poop, zoom=1)
    ab = AnnotationBbox(imagebox, (0.01, -0.0075), pad=0, box_alignment=(0,.45))
    axfigures.add_artist(ab)

    # Load arrow
    arr_poop = mpimg.imread('figures/arrow.png')
    imagebox = OffsetImage(arr_poop, zoom=0.4)
    ab = AnnotationBbox(imagebox, (0.04, 0.5), pad=0, box_alignment=(0,.45))
    axfigures.add_artist(ab)
    axfigures.text(s='fmt', x=0.045, y=0.8, fontsize=25, zorder=500)

    # Load mouse
    arr_poop = mpimg.imread('figures/mouse.jpg')
    imagebox = OffsetImage(arr_poop, zoom=0.3)
    ab = AnnotationBbox(imagebox, (0.07, 0.4), pad=0, box_alignment=(0,.45))
    axfigures.add_artist(ab)

    # High fat diet
    arr_poop = mpimg.imread('figures/pizza.png')
    imagebox = OffsetImage(arr_poop, zoom=0.13)
    ab = AnnotationBbox(imagebox, (0.435, 1), pad=0, box_alignment=(0,0))
    axfigures.add_artist(ab)

    # Vancomycin
    arr_poop = mpimg.imread('figures/needle.png')
    imagebox = OffsetImage(arr_poop, zoom=0.13)
    ab = AnnotationBbox(imagebox, (0.56, 1), pad=0, box_alignment=(0,0))
    axfigures.add_artist(ab)

    # Gentamicin
    arr_poop = mpimg.imread('figures/needle.png')
    imagebox = OffsetImage(arr_poop, zoom=0.13)
    ab = AnnotationBbox(imagebox, (0.70, 1), pad=0, box_alignment=(0,0))
    axfigures.add_artist(ab)

    return ax

def _data_figure_rel_and_qpcr_legend(axlegend, TAXLEVEL, CUTOFF_FRAC_ABUNDANCE):
    taxidx = TAXLEVEL_REV_IDX[TAXLEVEL]
    upper_tax = TAXLEVEL_INTS[taxidx+1]
    lower_tax = TAXLEVEL_INTS[taxidx]

    print('TAXLEVEL', TAXLEVEL)
    print(taxidx)
    print(upper_tax)
    print(lower_tax)

    # Last one is the one that aggregates the rest of the taxonomies
    labels = list(DATA_FIGURE_COLORS.keys())
    labels.sort()
    print('here')
    print(labels)

    # Put aggregate last if it is there
    last_label = None
    for label in labels:
        print(label)
        if len(label.split(' ')) > 2:
            last_label = label
            break
    if last_label is not None:
        labels.remove(last_label)
        labels.append(last_label)

    ims = []
    for label in labels:
        im, = axlegend.bar([0],[0],color=DATA_FIGURE_COLORS[label], label=label)
        ims.append(im)

    extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    legend_handle = [extra]
    legend_handle = legend_handle + ims
    extra_col = [extra]*(len(ims)+1)
    legend_handle = legend_handle + extra_col + extra_col
    

    empty_label = ''
    legend_labels = [empty_label]* (len(ims)+1) + ['$\\bf{' + upper_tax.capitalize() + '}$']
    for label in labels[:-1]:
        l1,_ = label.split(' ')
        if l1 == 'nan':
            l1 = 'Uncultured Clone'
        legend_labels = legend_labels + [l1.capitalize()]
    legend_labels = legend_labels + ['Other < {}%'.format(CUTOFF_FRAC_ABUNDANCE*100)]
    legend_labels = legend_labels + ['$\\bf{' + lower_tax.capitalize() + '}$']
    for label in labels[:-1]:
        _,l2 = label.split(' ')
        if l2 == 'nan':
            l2 = ''
        legend_labels = legend_labels + [l2.capitalize()]

    axlegend.legend(legend_handle, legend_labels, ncol=3, loc='upper center', 
        fontsize=22, columnspacing=0.01)

    axlegend.spines['top'].set_visible(False)
    axlegend.spines['bottom'].set_visible(False)
    axlegend.spines['left'].set_visible(False)
    axlegend.spines['right'].set_visible(False)
    axlegend.xaxis.set_major_locator(plt.NullLocator())
    axlegend.xaxis.set_minor_locator(plt.NullLocator())
    axlegend.yaxis.set_major_locator(plt.NullLocator())
    axlegend.yaxis.set_minor_locator(plt.NullLocator())

def _data_figure_rel_and_qpcr(HEALTHY, TAXLEVEL, df, CUTOFF_FRAC_ABUNDANCE, axqpcr, 
    axrel, axpert, axinoculum, asvname_map, figlabelinoculum=None, figlabelqpcr=None,
    figlabelrel=None, make_legend=False, make_ylabels=True):
    '''Data summarization figure for the paper
    '''
    global DATA_FIGURE_COLORS
    global XKCD_COLORS_IDX
    subjset = loaddata(healthy=HEALTHY)

    taxidx = TAXLEVEL_REV_IDX[TAXLEVEL]
    upper_tax = TAXLEVEL_INTS[taxidx+1]
    lower_tax = TAXLEVEL_INTS[taxidx]

    # Plot
    labels = np.asarray(list(df.index))
    labels = labels[::-1]
    matrix = df.values
    matrix = np.flipud(matrix)
    times = np.asarray(list(df.columns))

    print('Plot healthy', HEALTHY)
    print(labels)
    print(times)

    # transform the indicies for the perturbations (we're in index space not time space)
    for perturbation in subjset.perturbations:
        start_t = perturbation.start
        end_t = perturbation.end

        startidx = np.searchsorted(times, start_t)
        endidx = np.searchsorted(times, end_t)
        perturbation.start = startidx - 0.5
        perturbation.end = endidx + 0.5

    # Plot relative abundance, Create a stacked bar chart
    offset = np.zeros(matrix.shape[1])
    # colors = sns.color_palette('muted')

    for row in range(matrix.shape[0]):
        label = labels[row]
        if label in DATA_FIGURE_COLORS:
            color = DATA_FIGURE_COLORS[label]
        else:
            color = XKCD_COLORS[XKCD_COLORS_IDX]
            XKCD_COLORS_IDX += 1
            DATA_FIGURE_COLORS[label] = color

        axrel.bar(np.arange(len(times)), matrix[row,:], bottom=offset, color=color, label=label,
            width=1)
        offset = offset + matrix[row,:]

    # Set the xlabels
    locs = np.arange(0, len(times),step=10)
    ticklabels= times[locs]
    axrel.set_xticks(locs)
    axrel.set_xticklabels(ticklabels)
    axrel.yaxis.set_major_locator(plt.NullLocator())
    axrel.yaxis.set_minor_locator(plt.NullLocator())
    for tick in axrel.xaxis.get_major_ticks():
        tick.label.set_fontsize(24)

    # Plot the qpcr
    qpcr_meas = {}
    for subj in subjset:
        for t in subj.times:
            if t not in qpcr_meas:
                qpcr_meas[t] = []
            qpcr_meas[t].append(subj.qpcr[t].mean())

    # Plot the geometric mean of the subjects' qpcr
    for key in qpcr_meas:
        vals = qpcr_meas[key]
        a = 1
        for val in vals:
            a *= val
        a = a**(1/len(vals))
        qpcr_meas[key] = a

    # Sort and set
    times_qpcr = np.sort(list(qpcr_meas.keys()))
    vals = np.zeros(len(times_qpcr))
    for iii,t  in enumerate(times_qpcr):
        vals[iii] = qpcr_meas[t]
    axqpcr.plot(np.arange(len(times_qpcr)), vals, marker='o', linestyle='-', color='black')
    axqpcr.xaxis.set_major_locator(plt.NullLocator())
    axqpcr.xaxis.set_minor_locator(plt.NullLocator())
    max_qpcr_value = np.max(vals)
    # axqpcr.set_yscale('log')

    # Plot the inoculum
    # #################
    inoculum_subjset = loadinoculum()
    if HEALTHY:
        inoculum = inoculum_subjset['healthy']
    else:
        inoculum = inoculum_subjset['ulcerative colitis']

    # print(inoculum.df()['raw'].head())
    print('Inoculum')
    df, asvname_map_inoc = inoculum.cluster_by_taxlevel(dtype='raw', lca=False, taxlevel=TAXLEVEL,
        index_formatter='%({})s %({})s'.format(upper_tax, lower_tax), smart_unspec=False)
    df = _get_top(df, cutoff_frac_abundance=CUTOFF_FRAC_ABUNDANCE, 
        taxlevel=TAXLEVEL, asvname_map=asvname_map_inoc)

    matrix = df.to_numpy()
    matrix = np.flipud(matrix)
    matrix = matrix / np.sum(matrix)
    labels = np.asarray(list(df.index))
    labels = labels[::-1]

    offset = 0
    for row in range(matrix.shape[0]):
        label = labels[row]
        if label in DATA_FIGURE_COLORS:
            color = DATA_FIGURE_COLORS[label]
        else:
            color = XKCD_COLORS[XKCD_COLORS_IDX]
            XKCD_COLORS_IDX += 1
            DATA_FIGURE_COLORS[label] = color
        axinoculum.bar([0], matrix[row], bottom=[offset], label=labels, width=1, color=color)
        offset += matrix[row,0]
    axinoculum.xaxis.set_major_locator(plt.NullLocator())
    axinoculum.xaxis.set_minor_locator(plt.NullLocator())
    axinoculum.set_ylim(bottom=0, top=1)

    for tick in axinoculum.yaxis.get_major_ticks():
        tick.label.set_fontsize(24)


    # fig.subplots_adjust(left=0.07, right=0.82, hspace=0.1)
    axqpcr.set_ylabel('CFUs/g', size=25, fontweight='bold')
    # axqpcr.yaxis.set_label_coords(-0.06, 0.5)
    axinoculum.set_ylabel('Relative Abundance', size=25, fontweight='bold')
    axrel.set_xlabel('Time (d)', size=25, fontweight='bold')
    # axrel.xaxis.set_label_coords(0.5,-0.1, transform=axrel.transAxes)
    axrel.set_ylim(bottom=0,top=1)
    
    
    if HEALTHY:
        title = 'Healthy Cohort'
    else:
        title = 'Ulcerative Colitis Cohort'
    axqpcr.set_title(title, fontsize=30, fontweight='bold', y=1.3)
        # transform=axqpcr.transAxes)

    axpert.set_xlim(axrel.get_xlim())
    pl.visualization.shade_in_perturbations(axpert, perturbations=subjset.perturbations, textsize=22, 
        alpha=0)
    for perturbation in subjset.perturbations:
        axpert.axvline(x=perturbation.start, color='black', linestyle='--', lw=2)
        axpert.axvline(x=perturbation.end, color='black', linestyle='--', linewidth=2)

    if figlabelinoculum is not None:
        axinoculum.text(x = 0, y= 1.01,
            s=figlabelinoculum, fontsize=25, fontweight='bold',
            transform=axinoculum.transAxes)
    if figlabelqpcr is not None:
        axpert.text(x = 0, y= 1.01,
            s=figlabelqpcr, fontsize=25, fontweight='bold',
            transform=axpert.transAxes)
    if figlabelrel is not None:
        axrel.text(x = 0, y= 1.01,
            s=figlabelrel, fontsize=25, fontweight='bold',
            transform=axrel.transAxes)

    return max_qpcr_value

def _sum_qpcr(healthy):
    subjset = loaddata(healthy)
    qpcr = {}

    for subj in subjset.times:
        for t in subj.times:
            if t not in qpcr:
                qpcr[t] = []
            qpcr[t].append(subj.qpcr[t].mean())

    times = np.asarray(list(qpcr.keys()))
    qpcr = np.asarray(list(qpcr.values()))

    idxs = np.argsort(times)

    times= times[idxs]
    qpcr = qpcr[idxs]
    return times, qpcr

def _make_full_df(HEALTHY, TAXLEVEL, CUTOFF_FRAC_ABUNDANCE):
    subjset = loaddata(HEALTHY)
    # Make the data frame
    taxidx = TAXLEVEL_REV_IDX[TAXLEVEL]
    upper_tax = TAXLEVEL_INTS[taxidx+1]
    lower_tax = TAXLEVEL_INTS[taxidx]

    df = None
    times = []
    for subj in subjset:
        times = np.append(times, subj.times)
    times = np.sort(np.unique(times))
    t2idx = {}
    for i,t in enumerate(times):
        t2idx[t] = i
    times_cnts = np.zeros(len(times))

    for subj in subjset:
        dfnew, asvname_map = subj.cluster_by_taxlevel(dtype='abs', lca=False, taxlevel=TAXLEVEL, 
            index_formatter='%({})s %({})s'.format(upper_tax, lower_tax), smart_unspec=False)
        df, times_cnts = _add_unequal_col_dataframes(df=df, dfother=dfnew, times=times, 
            times_cnts=times_cnts, t2idx=t2idx)

    df = df / df.sum(axis=0)

    # Only plot the OTUs that have a totol percent abundance over a threshold
    if CUTOFF_FRAC_ABUNDANCE is not None:
        df = _get_top(df, cutoff_frac_abundance=CUTOFF_FRAC_ABUNDANCE, taxlevel=TAXLEVEL)

    return df, asvname_map


# Species Heatmap
# ---------------
def species_heatmap():
    N_CONSEC = 5
    MIN_REL_ABUND = 0 #0.001

    fig = plt.figure(figsize=(10,30))
    ax = fig.add_subplot(121)
    df_healthy, df_uc = _get_top_asvs_species_heatmap(n_consec=N_CONSEC,
        min_rel_abund=MIN_REL_ABUND)
    subjset_healthy = loaddata(True)

    df_top = df_healthy.iloc[:200]

    species_heatmap_single(subjset_healthy, df_top, ax)
    # ax.set_title('Healthy Top 200', fontsize=20, fontweight='bold')
    
    ax = fig.add_subplot(122)
    subjset_uc = loaddata(False)
    df_bottom = df_healthy.iloc[200:]
    species_heatmap_single(subjset_healthy, df_bottom, ax, True, cbar=True)
    # ax.set_title('Healthy bottom 200', fontsize=20, fontweight='bold')

    fig.suptitle('Healthy',fontsize=20, fontweight='bold')
    fig.subplots_adjust(wspace=0.2, right=0.96)

    plt.show()

def _combine_dfs(subjset, dtype):
    times = []
    for subj in subjset:
        times = np.append(times, subj.times)
    times = np.sort(np.unique(times))
    t2idx = {}
    for i,t in enumerate(times):
        t2idx[t] = i
    times_cnts = np.zeros(len(times))

    df = None
    for subj in subjset:
        dfnew = subj.df()[dtype]
        df, times_cnts = _add_unequal_col_dataframes(df=df, dfother=dfnew, times=times, 
            times_cnts=times_cnts, t2idx=t2idx)
    df = df / times_cnts
    return df

def _get_top_asvs_species_heatmap(n_consec, min_rel_abund, dtype='abs'):
    subjset_healthy = loaddata(True)
    subjset_uc = loaddata(False)

    df_healthy_rel = _combine_dfs(subjset_healthy, 'rel')
    df_uc_rel = _combine_dfs(subjset_uc, 'rel')

    df_healthy = _combine_dfs(subjset_healthy, dtype)
    df_uc = _combine_dfs(subjset_uc, dtype)


    M_healthy = df_healthy_rel.to_numpy()
    M_uc = df_uc_rel.to_numpy()

    rows_to_keep_uc = []
    for row in range(M_uc.shape[0]):
        n_uc = 0
        for col in range(M_uc.shape[1]):
            if M_uc[row,col] > min_rel_abund:
                n_uc += 1
            else:
                n_uc = 0
            if n_uc == n_consec:
                rows_to_keep_uc.append(row)
                break

    rows_to_keep_healthy = []
    for row in range(M_healthy.shape[0]):
        n_healthy = 0
        for col in range(M_healthy.shape[1]):
            if M_healthy[row,col] > min_rel_abund:
                n_healthy += 1
            else:
                n_healthy = 0
            if n_healthy == n_consec:
                rows_to_keep_healthy.append(row)
                break

    rtk = np.sort(np.unique(
        np.append(rows_to_keep_uc, rows_to_keep_healthy)))

    print(rtk)
    

    df_uc = df_uc.iloc[rtk]
    df_healthy = df_healthy.iloc[rtk]

    labels = []
    for oidx in rtk:
        asv = subjset_uc.asvs[oidx]
        if asv.tax_is_defined(level='genus'):
            label = asv.taxonomy['genus'] + ' '
        else:
            label = ''
        if asv.tax_is_defined(level='species'):
            label += asv.taxonomy['species'] + ' '
        label += asv.name
        labels.append(label)

    d = {}
    for i,idx in enumerate(df_uc.index):
        d[idx] = labels[i]
    df_uc = df_uc.rename(index=d)
    df_healthy = df_healthy.rename(index=d)


    return df_healthy, df_uc

def species_heatmap_single(subjset, df, ax, display_ylabels=True, cbar=False):

    df = np.log10(df)
    df[df == float('-inf')] = np.nan

    # df = df.iloc[:750]s

    if not display_ylabels:
        yticklabels = False
    else:
        yticklabels = 1
    if cbar:
        cbar_kws = {'title': '$\\log_{10}$'}
    else:
        cbar_kws = None
    ax = sns.heatmap(df, robust=False, yticklabels=yticklabels, ax=ax, cbar=cbar,
        cmap='Blues', square=False) #, cbar_kws=cbar_kws)

    ax = pl.visualization.shade_in_perturbations(ax, subjset.perturbations, alpha=0)
    for perturbation in subjset.perturbations:
        ax.axvline(x=perturbation.start, color='black', linestyle='-', lw=2.5)
        ax.axvline(x=perturbation.end, color='black', linestyle='-', linewidth=2.5)


    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(4.5)


    print(df.head())
    print(len(df.index))

# Preprocess Filtering
# --------------------
def preprocess_filtering(healthy):

    # Thresholding
    thresh_rel_abund = 0.0001
    thresh_min_num_subj = 2
    thresh_min_num_consecutive = 5


    thresholds = np.arange(1, 25) / 50000
    min_num_consecutives = np.arange(1,8)
    dtype = 'rel'
    colonization_time = 5
    basepath = 'tmp/'
    os.makedirs(basepath, exist_ok=True)
    subjset_master = loaddata(healthy)

    min_num_subjects = np.arange(1, len(subjset_master)+1)
    matrices = []
    for subj in subjset_master:
        matrices.append(subj.matrix()[dtype])

    # Get the data
    fname = basepath + 'healthy{}_filtering_mns.pkl'.format(healthy)
    if os.path.isfile(fname):
        with open(fname, 'rb') as handle:
            master_n_there = pickle.load(handle)
    else:
        master_n_there = {}
        for i_mns, mns in enumerate(min_num_subjects):
            
            print('\n\nmin_num_subjects', mns)
            print('Top level {}/{}'.format(i_mns, len(min_num_subjects)))
            n_theres = {}
            for mnc in min_num_consecutives:
                n_theres[mnc] = []
                for iii, mthresh in enumerate(thresholds):
                    # subjset = copy.deepcopy(subjset_master)
                    n_theres[mnc].append(
                        _consistency(subjset_master, matrices, dtype, mthresh, mnc, colonization_time,
                        mns))
                    # print(n_theres[mnc][-1])
                    print('{}/{}'.format(iii,len(thresholds)))
            
            master_n_there[mns] = n_theres
        with open(fname, 'wb') as handle:
            pickle.dump(master_n_there, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Plot the data
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111, facecolor=None)

    # caption = 'How many ASVs pass initial filtering given the criteria. To pass the filtering,\nan ASV must ' \
    #     'have be greater than a given relative abundance for a given number of consecutive\n' \
    #     'time points in at least a given number of subjects.'

    ax.text(0.4, -0.08, 'Minimum relative abundance', fontsize=22)
    ax.text(-0.08, 0.3, 'Number of ASVs remaining', fontsize=22, rotation='vertical')

    if healthy:
        title = 'Healthy Cohort'
    else:
        title = 'Ulcerative Colitis Cohort'
    ax.text(0.4, 1.075, s=title, fontsize=30)

    # ax.text(0.5, -0.23, caption, fontsize=18, horizontalalignment='center')

    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_minor_locator(plt.NullLocator())

    axes = {}
    for mns in master_n_there:
        n_theres = master_n_there[mns]

        if not healthy:
            ax = fig.add_subplot(2,3,mns)
        else:
            ax = fig.add_subplot(2,2,mns)
        ax.grid()
        for mnc in min_num_consecutives:
            ax.plot(thresholds, n_theres[mnc], label='{} consecutive'.format(mnc))
        ax.set_title('{} Subject/s'.format(mns), fontsize=22)

        if not healthy and mns == 3:
            ax.legend(bbox_to_anchor=(1.05,1), loc='upper left', fontsize=14)
        if healthy and mns == 2:
            ax.legend(bbox_to_anchor=(1.05,1), loc='upper left', fontsize=14)

        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(16)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(16)
        axes[mns] = ax

    # fig.tight_layout()
    fig.subplots_adjust(left=0.095, right=0.865)

    for mns in master_n_there:
        # Identify our filtering criteria when necessary
        print('mns', mns)
        n_theres = master_n_there[mns]
        if mns == thresh_min_num_subj:
            ax = axes[mns]
            print('in here')
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()

            

            idx = np.searchsorted(thresholds, v=thresh_rel_abund)

            n_left = n_theres[thresh_min_num_consecutive][idx]

            print(thresholds)
            print(idx)
            print(thresholds[idx])
            print(n_theres[thresh_min_num_consecutive])
            print(n_left)

            ax.vlines(x=thresh_rel_abund, ymax=ymax*1.5, ymin=n_left, 
                linestyle='--', color='red', linewidth=3.5)

            ax.hlines(y=n_left, xmax=thresh_rel_abund, xmin=-100, 
                linestyle='--', color='red', linewidth=3.5)

            ax.scatter([thresh_rel_abund], [n_left], c='red', s=60, marker='o')


            ax.set_ylim(ymin,ymax)
            ax.set_xlim(xmin,xmax)



    # fig.suptitle(title, fontsize=30, fontweight='bold')
    # plt.show()
    plt.savefig(BASEPATH + 'preprocess_filtering_healthy{}.pdf'.format(healthy))
    # plt.savefig(BASEPATH + 'preprocess_filtering_healthy{}.png'.format(healthy))
    plt.close()

def _consistency(subjset, matrices, dtype, threshold, min_num_consecutive, colonization_time=None, 
    min_num_subjects=1):
    '''Filters the subjects by looking at the consistency of the 'dtype', which can
    be either 'raw' where we look for the minimum number of counts, 'rel', where we
    look for a minimum relative abundance, or 'abs' where we look for a minium 
    absolute abundance.

    There must be at least `threshold` for at least
    `min_num_consecutive` consecutive timepoints for at least
    `min_num_subjects` subjects for the ASV to be classified as valid.

    If a colonization time is specified, we only look after that timepoint

    Parameters
    ----------
    subjset : str, pylab.base.SubjectSet
        This is the SubjectSet object that we are doing the filtering on
        If it is a str, then it is the location of the saved object.
    dtype : str
        This is the string to say what type of data we are thresholding. Options
        are 'raw', 'rel', or 'abs'.
    threshold : numeric
        This is the threshold for either counts, relative abudnance, or
        absolute abundance
    min_num_consecutive : int
        Number of consecutive timepoints to look for
    colonization_time : numeric
        This is the time we are looking after for colonization. If None we assume 
        there is no colonization time.
    min_num_subjects : int
        This is the minimum number of subjects this needs to be valid for.

    Returns
    -------
    pylab.base.SubjectSet
        This is the filtered subject set.

    Raises
    ------
    ValueError
        If types are not valid or values are invalid
    '''
    if not pl.isstr(dtype):
        raise TypeError('`dtype` ({}) must be a str'.format(type(dtype)))
    if dtype not in ['raw', 'rel', 'abs']:
        raise ValueError('`dtype` ({}) not recognized'.format(dtype))
    if not pl.issubjectset(subjset):
        raise TypeError('`subjset` ({}) must be a pylab.base.SubjectSet'.format(
            type(subjset)))
    if not pl.isnumeric(threshold):
        raise TypeError('`threshold` ({}) must be a numeric'.format(type(threshold)))
    if threshold <= 0:
        raise ValueError('`threshold` ({}) must be > 0'.format(threshold))
    if not pl.isint(min_num_consecutive):
        raise TypeError('`min_num_consecutive` ({}) must be an int'.format(
            type(min_num_consecutive)))
    if min_num_consecutive <= 0:
        raise ValueError('`min_num_consecutive` ({}) must be > 0'.format(min_num_consecutive))
    if colonization_time is None:
        colonization_time = 0
    if not pl.isnumeric(colonization_time):
        raise TypeError('`colonization_time` ({}) must be a numeric'.format(
            type(colonization_time)))
    if colonization_time < 0:
        raise ValueError('`colonization_time` ({}) must be >= 0'.format(colonization_time))
    if min_num_subjects is None:
        min_num_subjects = 1
    if not pl.isint(min_num_subjects):
        raise TypeError('`min_num_subjects` ({}) must be an int'.format(
            type(min_num_subjects)))
    if min_num_subjects > len(subjset) or min_num_subjects <= 0:
        raise ValueError('`min_num_subjects` ({}) value not valid'.format(min_num_subjects))

    # Everything is fine, now we can do the filtering
    talley = np.zeros(len(subjset.asvs), dtype=int)
    for i, subj in enumerate(subjset):
        matrix = np.array(matrices[i]) #subj.matrix(min_rel_abund=None)[dtype]
        tidx_start = None
        for tidx, t in enumerate(subj.times):
            if t >= colonization_time:
                tidx_start = tidx
                break
        if tidx_start is None:
            raise ValueError('Something went wrong')
        matrix = matrix[:, tidx_start:]

        for oidx in range(matrix.shape[0]):
            consecutive = 0
            for tidx in range(matrix.shape[1]):
                if matrix[oidx,tidx] >= threshold:
                    consecutive += 1
                else:
                    consecutive = 0
                if consecutive >= min_num_consecutive:
                    talley[oidx] += 1
                    break

    invalid_oidxs = np.where(talley < min_num_subjects)[0]
    # invalid_oids = subjset.asvs.ids.order[invalid_oidxs]
    # subjset.pop_asvs(invalid_oids)
    return len(subjset.asvs) - len(invalid_oidxs)


# Semi-synthetic model performance
# --------------------------------
def semi_synthetic_benchmark_figure():

    # fname = 'test_df_semi_synthetic.pkl'
    # basepath = 'tmp/'
    # os.makedirs(basepath, exist_ok=True)

    # df = _make_fake_semi_synthetic()
    df = pd.read_csv('semi_synthetic/df.tsv', sep='\t')
    print(df.head())

    try:
        fig = _outer_semi_synthetic(
            df=df, only={'Number of Timepoints': 55, 'Number of Replicates': 5, 
            'Uniform Samples': False},
            x = 'Measurement Noise')
        plt.savefig(BASEPATH + 'semi_synthetic_measurement_noise.pdf')
        plt.savefig(BASEPATH + 'semi_synthetic_measurement_noise.png')
        plt.close()
        print('Measurement noise done')
    except Exception as e:
        print('Failed Measurement noise')
        print(e)

    try:
        fig = _outer_semi_synthetic(
            df=df, only={'Number of Timepoints': 55, 'Measurement Noise': 0.3, 
            'Uniform Samples': False},
            x = 'Number of Replicates')
        plt.savefig(BASEPATH + 'semi_synthetic_replicates.pdf')
        plt.savefig(BASEPATH + 'semi_synthetic_replicates.png')
        plt.close()
        print('Replicates done')
    except Exception as e:
        print('Failed Replicates noise')
        print(e)
    
    try:
        fig = _outer_semi_synthetic(
            df=df, only={'Number of Replicates': 4, 'Measurement Noise': 0.3, 
            'Uniform Samples': True},
            x = 'Number of Timepoints')
        plt.savefig(BASEPATH + 'semi_synthetic_timepoints.pdf')
        plt.savefig(BASEPATH + 'semi_synthetic_timepoints.png')
        plt.close()
        print('timepoints done')
    except Exception as e:
        print('Failed timepoints')
        print(e)

def _srn():
    return np.absolute(np.random.normal())

def _make_fake_semi_synthetic():
    columns = ['Model', 
            'Error Trajectories', 
            'Error Interactions', 
            'Error Perturbations', 
            'Error Topology',
            'Error Growth',
            'Error Clustering',
            'Measurement Noise',
            'Process Variance',
            'Number of Timepoints',
            'Uniform Samples',
            'Number of Replicates']

    n_dataseeds = 10
    models = ['MDSINE2', 'L2', 'cLV']
    n_timepoints = [35, 45, 50, 55, 65]
    n_replicates = [2,3,4,5]
    measurement_noises = [0.1, 0.2, 0.25, 0.3, 0.4]
    pv = 0.1

    data = {
        'Model': [],
        'Error Trajectories': [], 
        'Error Interactions': [], 
        'Error Perturbations': [], 
        'Error Topology': [],
        'Error Growth': [],
        'Error Clustering': [],
        'Measurement Noise': [],
        'Process Variance': [],
        'Number of Timepoints': [],
        'Uniform Samples': [],
        'Number of Replicates': []}
    # Do measurement noise
    nt = 55
    nr = 5
    us = False
    for model in models:
        for mn in measurement_noises:
            for ds in range(n_dataseeds):
                data['Model'].append(model)
                data['Error Trajectories'].append(_srn())
                data['Error Interactions'].append(_srn())
                data['Error Perturbations'].append(_srn())
                data['Error Topology'].append(_srn())
                data['Error Growth'].append(_srn())
                data['Error Clustering'].append(_srn())
                data['Measurement Noise'].append(mn)
                data['Process Variance'].append(pv)
                data['Number of Timepoints'].append(nt)
                data['Uniform Samples'].append(us)
                data['Number of Replicates'].append(nr)

    # Do n-replicates
    nt = 55
    mn = 0.3
    us = False
    for model in models:
        for nr in n_replicates:
            for ds in range(n_dataseeds):
                data['Model'].append(model)
                data['Error Trajectories'].append(_srn())
                data['Error Interactions'].append(_srn())
                data['Error Perturbations'].append(_srn())
                data['Error Topology'].append(_srn())
                data['Error Growth'].append(_srn())
                data['Error Clustering'].append(_srn())
                data['Measurement Noise'].append(mn)
                data['Process Variance'].append(pv)
                data['Number of Timepoints'].append(nt)
                data['Uniform Samples'].append(us)
                data['Number of Replicates'].append(nr)

    # Do n_timepoints
    mn = 0.3
    nr = 4
    us = True
    for model in models:
        for nt in n_timepoints:
            for ds in range(n_dataseeds):
                data['Model'].append(model)
                data['Error Trajectories'].append(_srn())
                data['Error Interactions'].append(_srn())
                data['Error Perturbations'].append(_srn())
                data['Error Topology'].append(_srn())
                data['Error Growth'].append(_srn())
                data['Error Clustering'].append(_srn())
                data['Measurement Noise'].append(mn)
                data['Process Variance'].append(pv)
                data['Number of Timepoints'].append(nt)
                data['Uniform Samples'].append(us)
                data['Number of Replicates'].append(nr)

    df = pd.DataFrame(data)
    return df

def _outer_semi_synthetic(df, only, x):
    fig = plt.figure(figsize=(10,5))

    # Do Measurement noise
    # Only use uniform samples = False, n_timepoitns = 55, n_replicates = 5
    hue = 'Model'
    ax = _inner_semi_synth(df=df, only=only, x=x, y='Error Trajectories', hue=hue,
        ax=fig.add_subplot(2,3,1), title='Forward Simulation Error', 
        ylabel='RMSE', yscale='log', legend=False)

    ax = _inner_semi_synth(df=df, only=only, x=x, y='Error Growth', hue=hue,
        ax=fig.add_subplot(2,3,2), title='Error Growth Rates', 
        ylabel='RMSE', yscale='linear', legend=False)
    
    ax = _inner_semi_synth(df=df, only=only, x=x, y='Error Interactions', hue=hue,
        ax=fig.add_subplot(2,3,3), title='Error Interactions', 
        ylabel='RMSE', yscale='log', legend=True)

    ax = _inner_semi_synth(df=df, only=only, x=x, y='Error Perturbations', hue=hue,
        ax=fig.add_subplot(2,3,4), title='Error Perturbations', 
        ylabel='RMSE', yscale='linear', legend=False)
    
    ax = _inner_semi_synth(df=df, only=only, x=x, y='Error Topology', hue=hue,
        ax=fig.add_subplot(2,3,5), title='Error Topology', 
        ylabel='AUC ROC', yscale='linear', legend=False)
    
    ax = _inner_semi_synth(df=df, only=only, x=x, y='Error Clustering', hue=hue,
        ax=fig.add_subplot(2,3,6), title='Error Cluster Assignment', 
        ylabel='Normalized Mutual Information', yscale='linear', legend=False)

    fig.suptitle(x, fontsize=22, fontweight='bold')
    fig.tight_layout()
    fig.subplots_adjust(top=0.87)
    return fig

def _inner_semi_synth(df, only, x, y, hue, ax, title, ylabel, yscale, legend):
    
    dftemp = df
    if only is not None:
        for col, val in only.items():
            dftemp = dftemp[dftemp[col] == val]

        # print(df.columns)
        # print(dftemp['Measurement Noise'])
        # sys.exit()

    print(dftemp)
    
    sns.boxplot(data=dftemp, x=x, y=y, hue=hue, ax=ax)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_yscale(yscale)

    if not legend:
        ax.get_legend().remove()
    else:
        ax.get_legend().remove()
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    return ax
 
# Bayes Factors
# -------------
def make_bayes_factor():
    '''Make the bayes factors
    '''

    chainhealthy = pl.inference.BaseMCMC.load('output_real/pylab24/real_runs/strong_priors/healthy1_5_0.0001_rel_2_5/ds0_is0_b5000_ns15000_mo-1_logTrue_pertsmult/graph_leave_out-1/mcmc.pkl')
    chainuc = pl.inference.BaseMCMC.load('output_real/pylab24/real_runs/strong_priors/healthy0_5_0.0001_rel_2_5/ds0_is1_b5000_ns15000_mo-1_logTrue_pertsmult/graph_leave_out-1/mcmc.pkl')

    _bayes_factor(chainhealthy, healthy=True)
    _bayes_factor(chainuc, healthy=False)

def _bayes_factor(chain, healthy):
    # Get the bayes factors
    subjset = chain.graph.data.subjects
    asvs = subjset.asvs
    interactions = chain.graph[names.STRNAMES.INTERACTIONS_OBJ]
    bayes_factors = interactions.generate_bayes_factors_posthoc(
        prior=chain.graph[names.STRNAMES.CLUSTER_INTERACTION_INDICATOR].prior,
        section='posterior')
    
    # Get the asv order (same as coclustering)
    if healthy:
        fname = 'tmp/network_matrix_order/Healthy_order_co_cluster.txt'
    else:
        fname = 'tmp/network_matrix_order/UC_order_co_clusters.txt'
    f = open(fname, 'r')
    txt = f.read()
    f.close()
    order = txt.split('\n')
    for i in range(len(order)):
        order[i] = order[i].replace(' ', '_')
    print(order)

    # Make the labels
    ylabels = []
    reorder = []
    for aidx, asvname in enumerate(order):
        print(asvname)
        asv = asvs[asvname]
        reorder.append(asv.idx)
        if asv.tax_is_defined('species'):
            species = asv.taxonomy['species']
            species = species.split('/')
            if len(species) >= 3:
                species = species[:2]
            species = '/'.join(species)
            label = pl.asvname_formatter(
                format='%(genus)s {spec} %(name)s | {idx}'.format(
                    spec=species, idx=aidx+1), 
                asv=asv, asvs=asvs)
        elif asv.tax_is_defined('genus'):
            label = pl.asvname_formatter(
                format='* %(genus)s %(name)s | {idx}'.format(idx=aidx+1),
                asv=asv, asvs=asvs)
        elif asv.tax_is_defined('family'):
            label = pl.asvname_formatter(
                format='** %(family)s %(name)s | {idx}'.format(idx=aidx+1),
                asv=asv, asvs=asvs)
        elif asv.tax_is_defined('order'):
            label = pl.asvname_formatter(
                format='*** %(order)s %(name)s | {idx}'.format(idx=aidx+1),
                asv=asv, asvs=asvs)
        elif asv.tax_is_defined('class'):
            label = pl.asvname_formatter(
                format='**** %(class)s %(name)s | {idx}'.format(idx=aidx+1),
                asv=asv, asvs=asvs)
        elif asv.tax_is_defined('phylum'):
            label = pl.asvname_formatter(
                format='***** %(phylum)s %(name)s | {idx}'.format(idx=aidx+1),
                asv=asv, asvs=asvs)
        elif asv.tax_is_defined('kingdom'):
            label = pl.asvname_formatter(
                format='****** %(kingdom)s %(name)s | {idx}'.format(idx=aidx+1),
                asv=asv, asvs=asvs)
        else:
            raise ValueError('Something went wrong: {}'.format(str(asv)))

        label = label.replace('_', ' ')
        ylabels.append(label)

    bayes_factors = bayes_factors[reorder, :]
    bayes_factors = bayes_factors[:, reorder]

    # Make the plot
    fig = plt.figure(figsize=(40,55))
    ax = fig.add_subplot(111)
    for i in range(bayes_factors.shape[0]):
        bayes_factors[i,i] = np.nan
    cmap = sns.color_palette("Blues", as_cmap=True)
    im = ax.imshow(bayes_factors, cmap=cmap, vmax=10, aspect='auto')

    xticks = np.arange(len(asvs), step=2, dtype=int)
    xlabels = [str(i+1) for i in xticks]

    ax.set_xticks(xticks)
    ax.set_yticks(np.arange(len(asvs)))
    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(22)
        tick.label.set_rotation(90)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(24)

    if healthy:
        ax.set_title('Healthy Cohort', fontsize=50)
    else:
        ax.set_title('Ulcerative Colitis Cohort', fontsize=50)

    s = 'Taxonomy Key\n'
    for taxidx, tax in enumerate(TAXLEVEL_INTS[1:]):
        s += '*'*(taxidx+1) + ': ' + tax
        if taxidx < len(TAXLEVEL_INTS) -2:
            s += ', '
    ax.text(x=-0.25, y=-0.05, s=s, fontsize=40, fontweight='bold', 
        transform=ax.transAxes)


    

    cbar = fig.colorbar(im, orientation='vertical', shrink=0.75)
    cbar.ax.get_yaxis().set_ticks([0,2,4,6,8,10])
    cbar.ax.get_yaxis().set_ticklabels(['0', '2', '4', '6', '8', '>10'], fontsize=40)
    cbar.ax.set_title('Bayes Factor', fontsize=40, fontweight='bold', y=1.03)

    
    fig.subplots_adjust(top=0.95, bottom=0.08,right=1., left=0.25)
    plt.savefig('output_figures/bayes_factors_healthy{}.pdf'.format(healthy))
    plt.close()

# Interactions
# ------------
def make_interactions():
    '''Make the bayes factors
    '''

    chainhealthy = pl.inference.BaseMCMC.load('output_real/pylab24/real_runs/strong_priors/healthy1_5_0.0001_rel_2_5/ds0_is0_b5000_ns15000_mo-1_logTrue_pertsmult/graph_leave_out-1/mcmc.pkl')
    # chainuc = pl.inference.BaseMCMC.load('output_real/pylab24/real_runs/strong_priors/healthy0_5_0.0001_rel_2_5/ds0_is1_b5000_ns15000_mo-1_logTrue_pertsmult/graph_leave_out-1/mcmc.pkl')

    _interactions(chainhealthy, healthy=True)
    # _interactions(chainuc, healthy=False)

def _interactions(chain, healthy):
    # Get the bayes factors
    subjset = chain.graph.data.subjects
    asvs = subjset.asvs
    interactions = chain.graph[names.STRNAMES.INTERACTIONS_OBJ]
    bayes_factors = interactions.generate_bayes_factors_posthoc(
        prior=chain.graph[names.STRNAMES.CLUSTER_INTERACTION_INDICATOR].prior,
        section='posterior')
    interactions = pl.variables.summary(interactions, only='mean', section='posterior',
        set_nan_to_0=True)['mean']
    self_interactions = chain.graph[names.STRNAMES.SELF_INTERACTION_VALUE]
    self_interactions = pl.variables.summary(self_interactions, only='mean', section='posterior')['mean']
    self_interactions = -np.absolute(self_interactions)

    

    for i in range(interactions.shape[0]):
        for j in range(interactions.shape[1]):
            if bayes_factors[i,j] < 10:
                interactions[i,j] = 0
    
    # Get the asv order (same as coclustering)
    if healthy:
        fname = 'tmp/network_matrix_order/Healthy_order_co_cluster.txt'
    else:
        fname = 'tmp/network_matrix_order/UC_order_co_clusters.txt'
    f = open(fname, 'r')
    txt = f.read()
    f.close()
    order = txt.split('\n')
    for i in range(len(order)):
        order[i] = order[i].replace(' ', '_')
    print(order)

    # Make the labels
    ylabels = []
    reorder = []
    for aidx, asvname in enumerate(order):
        asv = asvs[asvname]
        reorder.append(asv.idx)
        if asv.tax_is_defined('species'):
            species = asv.taxonomy['species']
            species = species.split('/')
            if len(species) >= 3:
                species = species[:2]
            species = '/'.join(species)
            label = pl.asvname_formatter(
                format='%(genus)s {spec} %(name)s | {idx}'.format(
                    spec=species, idx=aidx+1), 
                asv=asv, asvs=asvs)
        elif asv.tax_is_defined('genus'):
            label = pl.asvname_formatter(
                format='* %(genus)s %(name)s | {idx}'.format(idx=aidx+1),
                asv=asv, asvs=asvs)
        elif asv.tax_is_defined('family'):
            label = pl.asvname_formatter(
                format='** %(family)s %(name)s | {idx}'.format(idx=aidx+1),
                asv=asv, asvs=asvs)
        elif asv.tax_is_defined('order'):
            label = pl.asvname_formatter(
                format='*** %(order)s %(name)s | {idx}'.format(idx=aidx+1),
                asv=asv, asvs=asvs)
        elif asv.tax_is_defined('class'):
            label = pl.asvname_formatter(
                format='**** %(class)s %(name)s | {idx}'.format(idx=aidx+1),
                asv=asv, asvs=asvs)
        elif asv.tax_is_defined('phylum'):
            label = pl.asvname_formatter(
                format='***** %(phylum)s %(name)s | {idx}'.format(idx=aidx+1),
                asv=asv, asvs=asvs)
        elif asv.tax_is_defined('kingdom'):
            label = pl.asvname_formatter(
                format='****** %(kingdom)s %(name)s | {idx}'.format(idx=aidx+1),
                asv=asv, asvs=asvs)
        else:
            raise ValueError('Something went wrong: {}'.format(str(asv)))

        label = label.replace('_', ' ')
        ylabels.append(label)

    for i in range(interactions.shape[0]):
        interactions[i,i] = self_interactions[i]
    interactions = interactions[reorder, :]
    interactions = interactions[:, reorder]

    # Make the plot
    fig = plt.figure(figsize=(40,55))
    ax = fig.add_subplot(111)
    for i in range(interactions.shape[0]):
        interactions[i,i] = np.nan
    maxval = np.nanmax(np.absolute(interactions))
    im = ax.imshow(interactions, cmap='RdBu', aspect='auto',
        norm=mcolors.SymLogNorm(linthresh=1e-14, linscale=1, vmax=maxval,
            vmin=-maxval))

    xticks = np.arange(len(asvs), step=2, dtype=int)
    xlabels = [str(i+1) for i in xticks]

    ax.set_xticks(xticks)
    ax.set_yticks(np.arange(len(asvs)))
    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(22)
        tick.label.set_rotation(90)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(24)

    if healthy:
        ax.set_title('Healthy Cohort', fontsize=50)
    else:
        ax.set_title('Ulcerative Colitis Cohort', fontsize=50)

    

    cbar = fig.colorbar(im, orientation='vertical', shrink=0.75)
    cbar.ax.get_yaxis().set_ticks([-1e-6, -1e-8, -1e-10, 1e-10, 1e-8, 1e-6])
    cbar.ax.get_yaxis().set_ticklabels(['-1e-6', '-1e-8', '-1e-10', '1e-10', '1e-8', '1e-6'], fontsize=40)
    cbar.ax.set_title('Interaction\nStrength', fontsize=40, fontweight='bold', y=1.03)

    
    fig.subplots_adjust(top=0.95, bottom=0.08,right=1., left=0.25)
    plt.savefig('output_figures/interactions_healthy{}.pdf'.format(healthy))
    plt.close()

# Model performance on new dataset
# --------------------------------
def model_performance_benchmark_figure():

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    df = pd.read_csv('results/full_predictive_performance.tsv', sep=',')
    ax = sns.boxplot(hue='Consortium', x='Model', y='Mean-Predictive', data=df, ax=ax,
        order=['MDSINE2', 'gLV', 'gLV-RA', 'cLV'])
    ax.set_ylabel('Spearman Correlation', fontsize=15)
    ax.set_xlabel('Model', fontsize=15)
    ax.get_legend().remove()
    ax.legend(fontsize=15)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)

    fig.tight_layout()
    fig.subplots_adjust(top=.889)

    plt.savefig(BASEPATH + 'model_performance_benchmark.pdf')
    plt.savefig(BASEPATH + 'model_performance_benchmark.png')
    plt.close()

# Time lookahead predictive performance
def time_lookahead_figure():

    def _subplot(ax, df, title):
        
        
        ax = sns.boxplot(data=df, x='Lookahead', y='Error', hue='Model', ax=ax)
        ax.set_yscale('log')
        ax.set_ylabel('')
        ax.set_xlabel('')

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(14)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(14)

        ax.set_title(title, fontsize=15)
        
        return ax


    path = 'results/time_lookahead_all_models.tsv'
    df = pd.read_csv(path, sep='\t', index_col=0)    
    fig = plt.figure(figsize=(10,10))
    

    # Only get days 1,3,5,7
    lookaheads = df['Lookahead'].to_numpy()
    rows = []
    for row in range(len(lookaheads)):
        if lookaheads[row] in [1,3,5,7]:
            rows.append(row)
    df = df.iloc[rows, :]

    axhealthy = fig.add_subplot(2,1,1)
    dfhealthy = df[df['Dataset'] == 'Healthy']
    axhealthy = _subplot(axhealthy, dfhealthy, title='Healthy')

    axhealthy.get_legend().remove()
    axhealthy.legend(fontsize=15, bbox_to_anchor=(1., 1))

    axuc = fig.add_subplot(2,1,2)
    dfuc = df[df['Dataset'] == 'Ulcerative Colitis']
    auc = _subplot(axuc, dfuc, title='Ulcerative Colitis')
    axuc.get_legend().remove()

    ax = fig.add_subplot(111, facecolor='none')
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_minor_locator(plt.NullLocator())
    
    ax.text(s='Time Look Ahead (Days)', fontsize=15,
        x=0.35, y=-0.075, transform=ax.transAxes)
    ax.text(s='Relative RMSE', fontsize=15,
        x=-0.125, y=0.5, transform=ax.transAxes, rotation='vertical')
    fig.subplots_adjust(right=0.75, left=0.11, bottom=0.12, top=0.87, hspace=0.225)
    plt.savefig('output_figures/time_look_ahead.pdf')
    plt.close()


os.makedirs('output_figures/', exist_ok=True)
# Alpha diversity
# alpha_diversity_mean_std()

# Beta diversity
# beta_diversity_figure()

# Data figure
# data_figure_rel_and_qpcr(horizontal=True)

# Species heatmap
# species_heatmap()

# Preprocess filtering
# preprocess_filtering(True)
# preprocess_filtering(False)

# Phylogenetic heatmap
# phylogenetic_heatmap_side_by_side()
phylogenetic_heatmap_gram_split()

# Semi-synthetic benchmarking
# semi_synthetic_benchmark_figure()

# Model performance benchmarking
# model_performance_benchmark_figure()

# Time lookahead
# time_lookahead_figure()

# Bayes Factors
# make_bayes_factor()

# Interactions
# make_interactions()
