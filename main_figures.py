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
# from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import TransformedBbox, BboxPatch, BboxConnector 
from matplotlib.patches import Rectangle
import matplotlib.legend as mlegend
from matplotlib.colors import LogNorm

import ete3
from ete3 import TreeStyle
import Bio
from Bio import Phylo

import pylab as pl
import diversity
import synthetic
import main_base
import names

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

# XKCD_COLORS_ = [
#     'windows blue',
#     'pale red',
#     'medium green',
#     'denim blue',
#     'amber',
#     'greyish',
#     'faded green',
#     'butter yellow',
#     'dusty purple',
#     'rose',
#     'blue green',
#     'slate',
#     'cream',
#     'bluish purple',
#     'snot green',
#     'deep pink',
#     'pastel blue',
#     'orchid',
#     'dull yellow',
#     'milk chocolate',

#     'black',
#     'white',
#     'red',
#     'orange']
# XKCD_COLORS = sns.xkcd_palette(XKCD_COLORS_)
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

def _get_top(df, cutoff_frac_abundance, taxlevel):
    matrix = df.values
    abunds = np.sum(matrix, axis=1)

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
    return df


# Phylogenetic and clustering heatmap
# -----------------------------------
def _make_names_of_clusters(n_clusters):
    '''Standardize the way to name the clusters so that they are all the same between
    eachother
    '''
    return ['Cluster {}'.format(i) for i in range(n_clusters)]

def _make_cluster_membership_heatmap(chainname, ax, order, binary, fig):
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
    im = ax.imshow(df.values, cmap=cmap, aspect='auto', **kwargs)
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    xticklabels = np.arange(1, len(df.columns)+1, dtype=int)
    ax.set_xticks(ticks=np.arange(len(xticklabels)))
    ax.set_xticklabels(labels=xticklabels)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_rotation(90)
        # tick.label.set_horizontalalignment('right')
    
    # Make grid
    ax.set_xticks(np.arange(0.5, len(df.columns), 1), minor=True)
    ax.set_yticks(np.arange(0.5, len(subjset.asvs), 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.1)

    ax.tick_params(axis='both', which='minor', left=False, bottom=False)

    cbaxes = fig.add_axes([0.92, 0.5, 0.02, 0.1]) # left, bottom, width, height
    cbar = plt.colorbar(im, cax=cbaxes, orientation='vertical')
    cbar.ax.set_xlabel('Relative Abund')

    return ax, newcolnames

def _make_perturbation_heatmap(chainname, min_bayes_factor, ax, colorder, fig):
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
    # max_val = 5
    im = ax.imshow(df.values, cmap='bwr_r', aspect='auto', vmin=-max_val, 
        vmax=max_val)
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.xaxis.set_major_locator(plt.NullLocator())
    
    # Make grid
    ax.set_xticks(np.arange(0.5, len(df.columns), 1), minor=True)
    ax.set_yticks(np.arange(0.5, len(subjset.perturbations), 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.1)

    ax.tick_params(axis='both', which='minor', left=False, bottom=False)
    
    cbaxes = fig.add_axes([0.92, 0.75, 0.02, 0.1]) # left, bottom, width, height
    cbar = plt.colorbar(im, cax=cbaxes, orientation='vertical')
    cbar.ax.set_xlabel('Pert Effect')

    ax.set_yticks(np.arange(len(subjset.perturbations)), minor=False)
    ax.set_yticklabels(list(df.index))

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_rotation(0)
    return ax

def _make_phylogenetic_tree(treename, chainname, ax):
    chain = pl.inference.BaseMCMC.load(chainname)
    asvs = chain.graph.data.asvs
    names = [str(name) for name in asvs.names.order]

    tree = ete3.Tree(treename)
    tree.prune(names, True)
    tree.write(outfile='temp.nhx')



    tree = Phylo.read('temp.nhx', format='newick')
    Phylo.draw(tree, axes=ax, do_show=False, show_confidence=False)
    asv_order = []
    for text in ax.texts:
        asv_order.append(text._text)
        # Substitute the name of the asv with the species/genus if possible
        asvname = str(text._text).replace(' ','')
        asv = asvs[asvname]
        if asv.tax_is_defined('genus'):
            asvname = ' ' + asv.taxonomy['genus']
            if asv.tax_is_defined('species'):
                spec = asv.taxonomy['species']
                l = spec.split('/')
                if len(l) <= 3:
                    spec = '/'.join(l[:3])
                    asvname = asvname + ' {}'.format(spec)
            
            text._text = asvname

        text._text = str(text._text).replace('OTU_', 'ASV_')
        
        text._text = text._text + '- ' * 75
        text.set_fontsize(6)

    return ax, asv_order

def phylogenetic_heatmap(healthy):

    if healthy:
        chainname = 'output_real/pylab24/real_runs/strong_priors/fixed_top/healthy1_5_0.0001_rel_2_5/' \
            'ds0_is0_b5000_ns15000_mo-1_logTrue_pertsmult/graph_leave_out-1/mcmc.pkl'
        # treename = 'raw_data/phylogenetic_tree_branch_len_preserved.nhx'
    else:
        chainname = 'output_real/pylab24/real_runs/strong_priors/fixed_top/healthy0_5_0.0001_rel_2_5/' \
            'ds0_is3_b5000_ns15000_mo-1_logTrue_pertsmult/graph_leave_out-1/mcmc.pkl'
        # treename = 'raw_data/phylogenetic_tree_branch_len_preserved.nhx'
    treename = 'raw_data/phylogenetic_tree_branch_len_preserved.nhx'

    fig = plt.figure(figsize=(12,20))
    gs = fig.add_gridspec(5,2)
    ax_phyl = fig.add_subplot(gs[1:,0])
    ax_clus = fig.add_subplot(gs[1:,1])
    ax_pert = fig.add_subplot(gs[0,1])



    ax_phyl, order = _make_phylogenetic_tree(treename=treename, chainname=chainname, ax=ax_phyl)
    ax_clus, colorder = _make_cluster_membership_heatmap(chainname=chainname, ax=ax_clus, 
        order=order, binary=False, fig=fig)
    ax_pert = _make_perturbation_heatmap(chainname=chainname, min_bayes_factor=10, 
        ax=ax_pert, colorder=colorder, fig=fig)

    ax_phyl.spines['top'].set_visible(False)
    ax_phyl.spines['bottom'].set_visible(False)
    ax_phyl.spines['left'].set_visible(False)
    ax_phyl.spines['right'].set_visible(False)
    ax_phyl.xaxis.set_major_locator(plt.NullLocator())
    ax_phyl.xaxis.set_minor_locator(plt.NullLocator())
    ax_phyl.yaxis.set_major_locator(plt.NullLocator())
    ax_phyl.yaxis.set_minor_locator(plt.NullLocator())
    ax_phyl.set_xlabel('')
    ax_phyl.set_ylabel('')

    fig.subplots_adjust(wspace=0.40, left=0.06, right=0.87, hspace=0.01)

    if healthy:
        title = 'Healthy'
        s = 'healthy'
    else:
        title = 'Ulcerative Colitis'
        s = 'uc'
    fig.suptitle(title, fontsize=30)

    # Make the caption
    txt = 'Average abundance from day 7-21 (pre-perturbation). Fixed topology inference.'
    # axtext = fig.add_subplot(gs[6:, 10:20])
    fig.text(0.5, 0.03, txt, ha='center', fontsize=12, wrap=True)

    plt.savefig(BASEPATH + 'phylo_clustering_heatmap_{}_RDP_alignment.pdf'.format(s))
    plt.close()
    # plt.show()

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
    
    if ax is None:
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111)
        
    val_healthy = {}
    val_unhealthy = {}
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
    pl.visualization.shade_in_perturbations(ax, perturbations=subjset.perturbations, textsize=10, 
        alpha=0)
    for perturbation in subjset.perturbations:
        ax.axvline(x=perturbation.start, color='black', linestyle='--', lw=2)
        ax.axvline(x=perturbation.end, color='black', linestyle='--', linewidth=2)


    # Set the ticks to be bold
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(13)
        tick.label.set_fontweight('bold')
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(13)
        tick.label.set_fontweight('bold')

    # Set the labels
    ax.set_title('Normalized Entropy', size=15, fontsize=35, fontweight='bold')
    ax.set_ylabel('nat', size=20, fontweight='bold')
    ax.set_xlabel('Days', size=20, fontweight='bold')

    # Make legend
    axlegend = fig.add_subplot(111, facecolor='none')
    handles = []
    l = mlines.Line2D([],[], color=colors_healthy,
        linestyle='-', label='Healthy')
    handles.append(l)
    l = mlines.Line2D([],[], color=colors_unhealthy,
        linestyle='-', label='Ulcerative Colitis')
    handles.append(l)
    lgnd2 = plt.legend(handles=handles)
    axlegend.add_artist(lgnd2)
    axlegend.spines['top'].set_visible(False)
    axlegend.spines['bottom'].set_visible(False)
    axlegend.spines['left'].set_visible(False)
    axlegend.spines['right'].set_visible(False)
    axlegend.xaxis.set_major_locator(plt.NullLocator())
    axlegend.xaxis.set_minor_locator(plt.NullLocator())
    axlegend.yaxis.set_major_locator(plt.NullLocator())
    axlegend.yaxis.set_minor_locator(plt.NullLocator())

    plt.savefig(BASEPATH + 'alpha_diversity_mean_std.pdf')


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
        axright = fig.add_subplot(122)
        axleft = fig.add_subplot(121)

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
        linestyle='-', label='Healthy')
    handles.append(l)
    l = mlines.Line2D([],[], color=colorsunhealthy,
        linestyle='-', label='Ulcerative Colitis')
    handles.append(l)
    lgnd2 = plt.legend(handles=handles, title='$\\bf{Dataset}$', 
        bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    axcenter.add_artist(lgnd2)

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
    axcenter.add_artist(lgnd3)

    axcenter.set_title('Bray-Curtis PCoA', fontsize=35, fontweight='bold')
    axcenter.set_xlabel('PC1: {:.3f}'.format(bc_pcoa.proportion_explained[0]),
        fontsize=20, fontweight='bold')
    axcenter.xaxis.set_label_coords(0.5,-0.08)

    axleft.set_ylabel('PC2: {:.3f}'.format(bc_pcoa.proportion_explained[1]),
        fontsize=20, fontweight='bold')

    mark_inset(parent_axes=axright, inset_axes=axleft, loc1a=1, loc1b=2, 
        loc2a=4, loc2b=3, fc='none', ec='crimson')
    axleft.spines['top'].set_color('crimson')
    axleft.spines['bottom'].set_color('crimson')
    axleft.spines['left'].set_color('crimson')
    axleft.spines['right'].set_color('crimson')
    axleft.tick_params(axis='both', color='crimson')

    for tick in axleft.yaxis.get_major_ticks():
        tick.label.set_fontsize(13)
        tick.label.set_fontweight('bold')
    for tick in axleft.xaxis.get_major_ticks():
        tick.label.set_fontsize(13)
        tick.label.set_fontweight('bold')

    for tick in axright.yaxis.get_major_ticks():
        tick.label.set_fontsize(13)
        tick.label.set_fontweight('bold')
    for tick in axright.xaxis.get_major_ticks():
        tick.label.set_fontsize(13)
        tick.label.set_fontweight('bold')

    # axleft.set_facecolor('whitesmoke')
    # axright.set_facecolor('whitesmoke')

    fig.subplots_adjust(left=0.09, right=0.83)

    # bc_pcoa.plot()
    plt.savefig(BASEPATH + 'pcoa_braycurtis_w_zoom.pdf')
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
def data_figure(HEALTHY, axtop=None, axmiddle=None, axbottom=None, axpert=None, figlabel=None):
    '''Dada summarization figure for the paper
    '''
    
    TAXLEVEL = 'family'
    DTYPE = 'abs'
    # if not HEALTHY:
    #     CUTOFF_FRAC_ABUNDANCE = 0.001
    # else:
    CUTOFF_FRAC_ABUNDANCE = 0.001

    subjset = loaddata(healthy=HEALTHY)

    # Make the data frame
    df = None
    times = []
    for subj in subjset:
        times = np.append(times, subj.times)
    times = np.sort(np.unique(times))
    t2idx = {}
    for i,t in enumerate(times):
        t2idx[t] = i
    times_cnts = np.zeros(len(times))
        
    # Aggregate at a taxonomic level and average over all of the subjects
    df = None
    # for subj in [subjset.iloc(1)]:
    for subj in subjset:
        dfnew = subj.cluster_by_taxlevel(dtype=DTYPE, lca=False, taxlevel=TAXLEVEL, 
            index_formatter='%({})s'.format(TAXLEVEL), smart_unspec=True)
        df, times_cnts = _add_unequal_col_dataframes(df=df, dfother=dfnew, times=times, 
            times_cnts=times_cnts, t2idx=t2idx)
        print('name', subj.name)
    df = df / times_cnts

    # Only plot the OTUs that have a totol percent abundance over a threshold
    if CUTOFF_FRAC_ABUNDANCE is not None:
        df = _get_top(df, cutoff_frac_abundance=CUTOFF_FRAC_ABUNDANCE, taxlevel=TAXLEVEL)

    # Plot
    labels = np.asarray(list(df.index))
    labels = labels[::-1]
    matrix = df.values
    matrix = np.flipud(matrix)
    times = np.asarray(list(df.columns))

    # transform the indicies for the perturbations (we're in index space not time space)
    for perturbation in subj.perturbations:
        start_t = perturbation.start
        end_t = perturbation.end

        startidx = np.searchsorted(times, start_t)
        endidx = np.searchsorted(times, end_t)
        perturbation.start = startidx - 0.5
        perturbation.end = endidx + 0.5

    # Make a gridspec into 4 rows, with the middle taking up 2
    if axtop is None:
        fig = plt.figure(figsize=(20,10))
        gs = fig.add_gridspec(4,1)
        verticle_gridspec_offset = 0
        axtop = fig.add_subplot(gs[verticle_gridspec_offset,:])
        axmiddle = fig.add_subplot(gs[verticle_gridspec_offset+1:verticle_gridspec_offset+3, :])
        axbottom = fig.add_subplot(gs[verticle_gridspec_offset+3,:])

    # Plot on every axis
    for ix, ax in enumerate([axbottom, axmiddle, axtop]):

        # Create a stacked bar chart
        offset = np.zeros(matrix.shape[1])
        colors = sns.xkcd_palette(colors=XKCD_COLORS)
        # colors = sns.color_palette('muted')

        for row in range(matrix.shape[0]):
            # if labels[row] == 'Other':
            #     color = OTHERCOLOR
            # else:
            color = colors[row]
            ax.bar(np.arange(len(times)), matrix[row,:], bottom=offset, color=color, label=labels[row],
                width=1)
            offset = offset + matrix[row,:]

        # Set ylims
        ix0top = 5e9
        ix1top = 5e10
        if ix == 0:
            ax.set_ylim(bottom=0, top=ix0top)
        elif ix == 1:
            ax.set_ylim(bottom=0, top=ix1top)
        else:
            # ax.set_ylim(bottom=ix1top)
            pass

        ax.set_xlim(-1, len(times))

        # Set Y tickers and tick labels
        #### TODO : Why is this getting rid of the ticks????????
        if ix == 0:
            ax.yaxis.set_major_locator(FixedLocator(np.arange(0, ix0top, step=ix0top/4)))
        elif ix == 1:
            step = ix1top/5
            ax.yaxis.set_major_locator(FixedLocator(np.arange(0, ix1top+step, step=step)))
        # else:
        #     _,top = ax.get_ylim()
        #     ax.yaxis.set_major_locator(FixedLocator(np.arange(ix1top, top, step=(top-ix1top)/4)))


        ax.yaxis.set_major_formatter(LogFormatter(labelOnlyBase=False))
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(13)
            tick.label.set_fontweight('bold')
        

        # Color in perturbations, only set label for the top
        label = ix == 2
        pl.visualization.shade_in_perturbations(ax, perturbations=subjset.perturbations, textsize=15, 
            label=label, alpha=0)
        
    
    # Set X tickers and tick labels
    l = np.arange(len(times), step=5)
    axbottom.set_xticks(ticks=l)
    axbottom.set_xticklabels(labels=times[l], fontsize=13, fontweight='bold')
    axtop.xaxis.set_major_locator(plt.NullLocator())
    axmiddle.xaxis.set_major_locator(plt.NullLocator())
    axmiddle.spines['top'].set_visible(False)
    axmiddle.spines['bottom'].set_visible(False)
    axbottom.spines['top'].set_visible(False)
    axtop.spines['bottom'].set_visible(False)

    # Set legend - make \n if there is an unspecified
    axtop.legend()
    handles, labels = axtop.get_legend_handles_labels()
    to_replace_label = '{} NA'.format(TAXLEVEL.capitalize())
    replace_label = '\n{} NA'.format(TAXLEVEL.capitalize())
    for iii in range(len(labels)):
        labels[iii] = labels[iii].replace(to_replace_label, replace_label)
            
    axtop.legend(handles[::-1], labels[::-1], 
        bbox_to_anchor=(1,1), 
        title='$\\bf{' + TAXLEVEL.capitalize() + '}$', 
        title_fontsize=14, fontsize=15)

    # # Make breakpoints between top and middle axes
    # d = .005  # how big to make the diagonal lines in axes coordinates
    # # arguments to pass to plot, just so we don't keep repeating them
    # kwargs = dict(transform=axtop.transAxes, color='k', clip_on=False)
    # axtop.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    # axtop.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    # kwargs.update(transform=axmiddle.transAxes)  # switch to the bottom axes
    # axmiddle.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    # axmiddle.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    # Make lines for perturbations
    if axpert is None:
        axpert = fig.add_subplot(gs[
            verticle_gridspec_offset:verticle_gridspec_offset+4,:], facecolor='none')
    axpert.set_xlim(axmiddle.get_xlim())
    axpert.spines['top'].set_visible(False)
    axpert.spines['bottom'].set_visible(False)
    axpert.spines['left'].set_visible(False)
    axpert.spines['right'].set_visible(False)
    axpert.xaxis.set_major_locator(plt.NullLocator())
    axpert.xaxis.set_minor_locator(plt.NullLocator())
    axpert.yaxis.set_major_locator(plt.NullLocator())
    axpert.yaxis.set_minor_locator(plt.NullLocator())
    for perturbation in subjset.perturbations:
        axpert.axvline(x=perturbation.start, color='black', linestyle=':', lw=3)
        axpert.axvline(x=perturbation.end, color='black', linestyle=':', linewidth=3)

    if figlabel is not None:
        axpert.text(x = -0.1, y= 1.1,
            s=figlabel, fontsize=30, fontweight='bold',
            transform=axpert.transAxes)


    # fig.subplots_adjust(left=0.07, right=0.82, hspace=0.1)

    # Set titles
    if DTYPE == 'abs':
        label = 'CFUs/g'
    elif DTYPE == 'rel':
        label = 'Relative Abundance'
    else:
        label = 'Counts'
    axmiddle.set_ylabel(label, size=20, fontweight='bold')
    axmiddle.yaxis.set_label_coords(-0.06, 0.5)
    axbottom.set_xlabel('Time (d)', size=20, fontweight='bold', va='center')
    axbottom.xaxis.set_label_coords(0.5,-0.5)
    
    if HEALTHY:
        title = 'Healthy Microbes'
    else:
        title = 'Ulcerative Colitis Microbes'
    axtop.set_title(title, fontsize=35, fontweight='bold')
    # plt.savefig(BASEPATH + 'datafigure_healthy{}_{}.pdf'.format(HEALTHY, TAXLEVEL))

    # plt.show()

def data_figure_both():

    fig = plt.figure(figsize=(20,10))

    gs = fig.add_gridspec(2,18)
    axinoculum = fig.add_subplot(gs[0,0:2])
    axleft = fig.add_subplot(gs[0,3:10])
    axright = fig.add_subplot(gs[0,11:])
    axwhole = fig.add_subplot(gs[0, :], facecolor='none')

    data_figure_side_by_side(True, fig, axleft=axleft, axright=axright, axinoculum=axinoculum, 
        axwhole=axwhole, figlabelleft='B', figlabelright='C', figlabelinoculum='A',
        make_legend=True)

    axinoculum = fig.add_subplot(gs[1,0:2])
    axleft = fig.add_subplot(gs[1,3:10])
    axright = fig.add_subplot(gs[1,11:])
    axwhole = fig.add_subplot(gs[1, :], facecolor='none')

    data_figure_side_by_side(False, fig, axleft=axleft, axright=axright, axinoculum=axinoculum, 
        axwhole=axwhole, figlabelleft='E', figlabelright='F', figlabelinoculum='D')


    fig.subplots_adjust(right=0.68, wspace=0.31, hspace=0.38, left=0.08)

def data_figure_side_by_side(HEALTHY, fig, axleft, axright, axinoculum, axwhole, figlabelleft=None,
    figlabelright=None, figlabelinoculum=None, make_legend=False):
    '''Dada summarization figure for the paper
    '''
    global DATA_FIGURE_COLORS
    global XKCD_COLORS_IDX
    
    TAXLEVEL = 'family'

    taxidx = TAXLEVEL_REV_IDX[TAXLEVEL]
    upper_tax = TAXLEVEL_INTS[taxidx+1]
    lower_tax = TAXLEVEL_INTS[taxidx]

    DTYPE = 'abs'
    # if not HEALTHY:
    #     CUTOFF_FRAC_ABUNDANCE = 0.001
    # else:
    CUTOFF_FRAC_ABUNDANCE = 0.01

    subjset = loaddata(healthy=HEALTHY)

    # Make the data frame
    df = None
    times = []
    for subj in subjset:
        times = np.append(times, subj.times)
    times = np.sort(np.unique(times))
    t2idx = {}
    for i,t in enumerate(times):
        t2idx[t] = i
    times_cnts = np.zeros(len(times))
        
    # Aggregate at a taxonomic level and average over all of the subjects
    df = None
    # for subj in [subjset.iloc(1)]:
    for subj in subjset:
        dfnew = subj.cluster_by_taxlevel(dtype=DTYPE, lca=False, taxlevel=TAXLEVEL, 
            index_formatter='%({})s %({})s'.format(upper_tax, lower_tax), smart_unspec=False)
        df, times_cnts = _add_unequal_col_dataframes(df=df, dfother=dfnew, times=times, 
            times_cnts=times_cnts, t2idx=t2idx)
        print('name', subj.name)
    df = df / times_cnts

    # Only plot the OTUs that have a totol percent abundance over a threshold
    if CUTOFF_FRAC_ABUNDANCE is not None:
        df = _get_top(df, cutoff_frac_abundance=CUTOFF_FRAC_ABUNDANCE, 
            taxlevel=TAXLEVEL)

    # Plot
    labels = np.asarray(list(df.index))
    labels = labels[::-1]
    matrix = df.values
    matrix = np.flipud(matrix)
    times = np.asarray(list(df.columns))

    # transform the indicies for the perturbations (we're in index space not time space)
    for perturbation in subj.perturbations:
        start_t = perturbation.start
        end_t = perturbation.end

        startidx = np.searchsorted(times, start_t)
        endidx = np.searchsorted(times, end_t)
        perturbation.start = startidx - 0.5
        perturbation.end = endidx + 0.5        

    # Plot on every axis
    for ix, ax in enumerate([axleft, axright]):

        # Create a stacked bar chart
        offset = np.zeros(matrix.shape[1])
        # colors = sns.xkcd_palette(colors=XKCD_COLORS)
        # colors = sns.color_palette('muted')

        for row in range(matrix.shape[0]):
            label = labels[row]
            if label in DATA_FIGURE_COLORS:
                color = DATA_FIGURE_COLORS[label]
            else:
                color = XKCD_COLORS[XKCD_COLORS_IDX]
                XKCD_COLORS_IDX += 1
                DATA_FIGURE_COLORS[label] = color
            ax.bar(np.arange(len(times)), matrix[row,:], bottom=offset, color=color, label=label,
                width=1)
            offset = offset + matrix[row,:]

        # Set ylims
        if ix == 1:
            ax.set_ylim(bottom=0, top=5e10)  

        # Color in perturbations, only set label for the top
        pl.visualization.shade_in_perturbations(ax, perturbations=subjset.perturbations, textsize=10, 
            alpha=0)
        for perturbation in subjset.perturbations:
            ax.axvline(x=perturbation.start, color='black', linestyle='--', lw=2)
            ax.axvline(x=perturbation.end, color='black', linestyle='--', linewidth=2)

        # Set y ticks

        # Set X ticks
        l = np.arange(len(times), step=10)
        ax.set_xticks(ticks=l)
        ax.set_xticklabels(labels=times[l], fontsize=13, fontweight='bold')

        ax.set_ylabel('CFUs/g', size=13, fontweight='bold')

    # # Make breakpoints between top and middle axes
    # d = .005  # how big to make the diagonal lines in axes coordinates
    # # arguments to pass to plot, just so we don't keep repeating them
    # kwargs = dict(transform=axtop.transAxes, color='k', clip_on=False)
    # axtop.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    # axtop.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    # Make lines for perturbations
    axwhole.set_xlim(axleft.get_xlim())

    if figlabelleft is not None:
        axleft.text(x = 0, y= 1.05,
            s=figlabelleft, fontsize=20, fontweight='bold',
            transform=axleft.transAxes)
    if figlabelright is not None:
        axright.text(x = 0, y= 1.05,
            s=figlabelright, fontsize=20, fontweight='bold',
            transform=axright.transAxes)
    if figlabelinoculum is not None:
        axinoculum.text(x = 0, y= 1.05,
            s=figlabelinoculum, fontsize=20, fontweight='bold',
            transform=axinoculum.transAxes)
    # Set titles
    if DTYPE == 'abs':
        label = 'CFUs/g'
    elif DTYPE == 'rel':
        label = 'Relative Abundance'
    else:
        label = 'Counts'
    axleft.set_xlabel('Time (d)', size=15, fontweight='bold', va='center', labelpad=-0.1, 
        transform=axleft.transAxes)
    axright.set_xlabel('Time (d)', size=15, fontweight='bold', va='center', labelpad=-0.1, 
        transform=axleft.transAxes)
    
    if HEALTHY:
        title = 'Healthy'
    else:
        title = 'Ulcerative Colitis'

    axwhole.text(x = 0.5, y= 1.15,
        s=title, fontsize=25, fontweight='bold',
        transform=axwhole.transAxes)

    # axwhole.set_title(title, fontsize=25, fontweight='bold', pad=5, 
    #     transform=axwhole.transAxes)

    mark_inset(parent_axes=axleft, inset_axes=axright, loc1a=2, loc1b=1, 
        loc2a=3, loc2b=4, fc='none', ec='crimson')
    axright.spines['top'].set_color('crimson')
    axright.spines['bottom'].set_color('crimson')
    axright.spines['left'].set_color('crimson')
    axright.spines['right'].set_color('crimson')
    axright.tick_params(axis='both', color='crimson')

    # Make the innoculum
    inoculum_subjset = loadinoculum()
    if HEALTHY:
        inoculum = inoculum_subjset['healthy']
    else:
        inoculum = inoculum_subjset['ulcerative colitis']

    # print(inoculum.df()['raw'].head())

    df = inoculum.cluster_by_taxlevel(dtype='raw', lca=False, taxlevel=TAXLEVEL,
        index_formatter='%({})s %({})s'.format(upper_tax, lower_tax), smart_unspec=False)

    # print(df.head())
    # print('cutoff abund', CUTOFF_FRAC_ABUNDANCE)

    df = _get_top(df, cutoff_frac_abundance=CUTOFF_FRAC_ABUNDANCE, 
        taxlevel=TAXLEVEL)

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
    axinoculum.set_ylabel('Relative Abundance', size=13, fontweight='bold')
    axinoculum.xaxis.set_major_locator(plt.NullLocator())
    axinoculum.xaxis.set_minor_locator(plt.NullLocator())

    # plt.savefig(BASEPATH + 'datafigure_healthy{}_{}.pdf'.format(HEALTHY, TAXLEVEL))

    # plt.show()
    if make_legend:
        # Last one is the one that aggregates the rest of the taxonomies
        # axwhole = fig.add_subplot(111, facecolor='none')

        labels = list(DATA_FIGURE_COLORS.keys())
        labels.sort()

        print(labels)

        # Put aggregate last
        last_label = None
        for label in labels:
            if len(label.split(' ')) > 2:
                last_label = label
                break
        if last_label is None:
            raise ValueError('mer sauce')
        labels.remove(last_label)
        labels.append(last_label)

        ims = []
        for label in labels:
            im, = axwhole.bar([0],[0],color=DATA_FIGURE_COLORS[label], label=label)
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
                l1 = 'Uncultured'
            legend_labels = legend_labels + [l1.capitalize()]
        legend_labels = legend_labels + ['Other < {}%'.format(CUTOFF_FRAC_ABUNDANCE*100)]
        legend_labels = legend_labels + ['$\\bf{' + lower_tax.capitalize() + '}$']
        for label in labels[:-1]:
            _,l2 = label.split(' ')
            if l2 == 'nan':
                l2 = ''
            legend_labels = legend_labels + [l2.capitalize()]

        axwhole.legend(legend_handle, legend_labels, ncol=3, loc='upper left', 
            bbox_to_anchor=(1.02, 1), fontsize=12)
    axwhole.spines['top'].set_visible(False)
    axwhole.spines['bottom'].set_visible(False)
    axwhole.spines['left'].set_visible(False)
    axwhole.spines['right'].set_visible(False)
    axwhole.xaxis.set_major_locator(plt.NullLocator())
    axwhole.xaxis.set_minor_locator(plt.NullLocator())
    axwhole.yaxis.set_major_locator(plt.NullLocator())
    axwhole.yaxis.set_minor_locator(plt.NullLocator())

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
        dfnew = subj.cluster_by_taxlevel(dtype='abs', lca=False, taxlevel=TAXLEVEL, 
            index_formatter='%({})s %({})s'.format(upper_tax, lower_tax), smart_unspec=False)
        df, times_cnts = _add_unequal_col_dataframes(df=df, dfother=dfnew, times=times, 
            times_cnts=times_cnts, t2idx=t2idx)
        print('name', subj.name)
    df = df / df.sum(axis=0)

    # Only plot the OTUs that have a totol percent abundance over a threshold
    if CUTOFF_FRAC_ABUNDANCE is not None:
        df = _get_top(df, cutoff_frac_abundance=CUTOFF_FRAC_ABUNDANCE, taxlevel=TAXLEVEL)

    return df

def data_figure_rel_and_qpcr():
    '''Plot the relative abundance and the qpcr plot above it
    '''
    TAXLEVEL = 'family'
    CUTOFF_FRAC_ABUNDANCE = 0.01
    global DATA_FIGURE_COLORS
    global XKCD_COLORS_IDX

    # Make the dataframs for healthy and unhealthy
    df_healthy = _make_full_df(True, TAXLEVEL, CUTOFF_FRAC_ABUNDANCE)
    df_unhealthy = _make_full_df(False, TAXLEVEL, CUTOFF_FRAC_ABUNDANCE)

    # Set the colors from most to least abundant - only consider healthy
    M = df_healthy.to_numpy()
    a = np.sum(M, axis=1)
    idxs = np.argsort(a)[::-1] # reverse the indexes so it goes from largest to smallest

    for idx in idxs:
        label = df_healthy.index[idx]
        color = XKCD_COLORS[XKCD_COLORS_IDX]
        XKCD_COLORS_IDX += 1
        DATA_FIGURE_COLORS[label] = color

    fig = plt.figure(figsize=(20,10))
    gs = fig.add_gridspec(7,33)

    axqpcr1 = fig.add_subplot(gs[:1,3:15])
    axrel = fig.add_subplot(gs[1:4,3:15])
    axpert = fig.add_subplot(gs[:4,3:15], facecolor='none')
    axinoculum = fig.add_subplot(gs[1:4,:2])

    max_qpcr_value1 = _data_figure_rel_and_qpcr(True, TAXLEVEL=TAXLEVEL,
        df=df_healthy, CUTOFF_FRAC_ABUNDANCE=CUTOFF_FRAC_ABUNDANCE,
        axqpcr=axqpcr1, axrel=axrel, axpert=axpert,
        axinoculum=axinoculum, make_ylabels=True,
        figlabelinoculum='A', figlabelqpcr='B', figlabelrel='C',
        make_legend=False)

    axqpcr2 = fig.add_subplot(gs[:1,21:])
    axrel = fig.add_subplot(gs[1:4,21:])
    axpert = fig.add_subplot(gs[:4,21:], facecolor='none')
    axinoculum = fig.add_subplot(gs[1:4,18:20])

    max_qpcr_value2 = _data_figure_rel_and_qpcr(False, TAXLEVEL=TAXLEVEL, 
        df=df_unhealthy, CUTOFF_FRAC_ABUNDANCE=CUTOFF_FRAC_ABUNDANCE,
        axqpcr=axqpcr2, axrel=axrel, axpert=axpert,
        axinoculum=axinoculum, make_ylabels=True,
        figlabelinoculum='C', figlabelqpcr='D',
        make_legend=False)

    # Set the same max and min value for the qpcr measurements
    max_qpcr_value = np.max([max_qpcr_value1, max_qpcr_value2])
    axqpcr1.set_ylim(bottom=1e9, top=max_qpcr_value*(1.05))
    axqpcr2.set_ylim(bottom=1e9, top=max_qpcr_value*(1.05))
    axqpcr1.set_yscale('log')
    axqpcr2.set_yscale('log')


    # Make the legend
    axlegend1 = fig.add_subplot(gs[4:6, :], facecolor='none')
    _data_figure_rel_and_qpcr_legend(axlegend=axlegend1, TAXLEVEL=TAXLEVEL,
        CUTOFF_FRAC_ABUNDANCE=CUTOFF_FRAC_ABUNDANCE)

    # Make the caption
    txt = '(A) Relative abundance of the inoculum sample of the healthy consortium. ' \
    '(B) Top is the qPCR measurements over time for the healthy consortium. \nBottom ' \
    'are the relative abundances over time for the healthy consortium. ' \
    '(C) Relative abundance of the inoculum sample of the Ulcerative Colitis (UC) consortium. \n' \
    '(D) Top is the qPCR measurements over time for the UC consortium. Bottom \n' \
    'are the relative abundances over time for the UC consortium.'
    # axtext = fig.add_subplot(gs[6:, 10:20])
    fig.text(0.5, 0.03, txt, ha='center', fontsize=12, wrap=True)


    fig.subplots_adjust( wspace=0.58, left=0.1, right=0.97, top=0.94)
    plt.savefig(BASEPATH + 'datafigure_rel.pdf')
    plt.close()

def _data_figure_rel_and_qpcr_legend(axlegend, TAXLEVEL, CUTOFF_FRAC_ABUNDANCE):
    taxidx = TAXLEVEL_REV_IDX[TAXLEVEL]
    upper_tax = TAXLEVEL_INTS[taxidx+1]
    lower_tax = TAXLEVEL_INTS[taxidx]

    # Last one is the one that aggregates the rest of the taxonomies
    labels = list(DATA_FIGURE_COLORS.keys())
    labels.sort()
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
        fontsize=10)

    axlegend.spines['top'].set_visible(False)
    axlegend.spines['bottom'].set_visible(False)
    axlegend.spines['left'].set_visible(False)
    axlegend.spines['right'].set_visible(False)
    axlegend.xaxis.set_major_locator(plt.NullLocator())
    axlegend.xaxis.set_minor_locator(plt.NullLocator())
    axlegend.yaxis.set_major_locator(plt.NullLocator())
    axlegend.yaxis.set_minor_locator(plt.NullLocator())

def _data_figure_rel_and_qpcr(HEALTHY, TAXLEVEL, df, CUTOFF_FRAC_ABUNDANCE, axqpcr, 
    axrel, axpert, axinoculum, figlabelinoculum=None, figlabelqpcr=None,
    figlabelrel=None, make_legend=False, make_ylabels=True):
    '''Dada summarization figure for the paper
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
    # plt.xticks(ticks=locs, labels=ticklabels)


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
    df = inoculum.cluster_by_taxlevel(dtype='raw', lca=False, taxlevel=TAXLEVEL,
        index_formatter='%({})s %({})s'.format(upper_tax, lower_tax), smart_unspec=False)
    df = _get_top(df, cutoff_frac_abundance=CUTOFF_FRAC_ABUNDANCE, 
        taxlevel=TAXLEVEL)

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


    # fig.subplots_adjust(left=0.07, right=0.82, hspace=0.1)
    axqpcr.set_ylabel('CFUs/g', size=20, fontweight='bold')
    # axqpcr.yaxis.set_label_coords(-0.06, 0.5)
    axinoculum.set_ylabel('Relative Abundance', size=20, fontweight='bold')
    axrel.set_xlabel('Time (d)', size=20, fontweight='bold')
    # axrel.xaxis.set_label_coords(0.5,-0.1, transform=axrel.transAxes)
    axrel.set_ylim(bottom=0,top=1)
    
    
    if HEALTHY:
        title = 'Healthy Microbes'
    else:
        title = 'Ulcerative Colitis Microbes'
    axqpcr.set_title(title, fontsize=21, fontweight='bold', y=1.15)
        # transform=axqpcr.transAxes)


    axpert.set_xlim(axrel.get_xlim())
    pl.visualization.shade_in_perturbations(axpert, perturbations=subjset.perturbations, textsize=10, 
        alpha=0)
    for perturbation in subjset.perturbations:
        axpert.axvline(x=perturbation.start, color='black', linestyle='--', lw=2)
        axpert.axvline(x=perturbation.end, color='black', linestyle='--', linewidth=2)

    axpert.spines['top'].set_visible(False)
    axpert.spines['bottom'].set_visible(False)
    axpert.spines['left'].set_visible(False)
    axpert.spines['right'].set_visible(False)
    axpert.xaxis.set_major_locator(plt.NullLocator())
    axpert.xaxis.set_minor_locator(plt.NullLocator())
    axpert.yaxis.set_major_locator(plt.NullLocator())
    axpert.yaxis.set_minor_locator(plt.NullLocator())

    if figlabelinoculum is not None:
        axinoculum.text(x = 0, y= 1.05,
            s=figlabelinoculum, fontsize=20, fontweight='bold',
            transform=axinoculum.transAxes)
    if figlabelqpcr is not None:
        axpert.text(x = 0, y= 1.05,
            s=figlabelqpcr, fontsize=20, fontweight='bold',
            transform=axpert.transAxes)
    # if figlabelrel is not None:
    #     axrel.text(x = -0.08, y= 1.05,
    #         s=figlabelrel, fontsize=20, fontweight='bold',
    #         transform=axrel.transAxes)

    return max_qpcr_value

def figure1():
    '''Figure 1

    data1
    data2
    alpha diversity
    beta diversity
    '''
    fig = plt.figure(figsize=(40,20))
    gs = fig.add_gridspec(9,5) # This is zero indexed

    # Top right - data healthy
    print('Plotting healthy data')
    axtop = fig.add_subplot(gs[0, :2])
    axmiddle = fig.add_subplot(gs[1:3, :2])
    axbottom = fig.add_subplot(gs[3, :2])
    axpert = fig.add_subplot(gs[:4, :2], facecolor='none')
    data_figure(True, axtop, axmiddle, axbottom, axpert, figlabel='A')

    # Top left - data unhealthy
    print('plotting unhealthy data')
    axtop = fig.add_subplot(gs[0, 3:])
    axmiddle = fig.add_subplot(gs[1:3, 3:])
    axbottom = fig.add_subplot(gs[3, 3:])
    axpert = fig.add_subplot(gs[:4, 3:], facecolor='none')
    data_figure(False, axtop, axmiddle, axbottom, axpert, figlabel='B')

    # Bottom left - Alpha diversity
    print('alpha diversity')
    ax = fig.add_subplot(gs[-4:, :2])
    alpha_diversity_figure(ax=ax, figlabel='C')

    # Bottom right - beta diversity
    axleft = fig.add_subplot(gs[-4:, -2])
    axright = fig.add_subplot(gs[-4:, -1])
    axcenter = fig.add_subplot(gs[-4:, -2:], facecolor='none')
    print('beta diversity')
    beta_diversity_figure(axleft, axright, axcenter, figlabel='D')

    

    plt.savefig(BASEPATH + 'figure1_order.pdf')
    # plt.show()


# Species Heatmap
# ---------------
def species_heatmap():
    N_CONSEC = 5
    MIN_REL_ABUND = 0 #0.001

    fig = plt.figure()
    fig.set_size_inches(11.5, 8)
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
        label += asv.name.replace('OTU', 'ASV')
        labels.append(label)

    d = {}
    for i,idx in enumerate(df_uc.index):
        d[idx] = labels[i]
    df_uc = df_uc.rename(index=d)
    df_healthy = df_healthy.rename(index=d)


    return df_healthy, df_uc


# Species Heatmap
# ---------------
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
 
# alpha_diversity_figure()
# alpha_diversity_figure_boxplot()
# alpha_diversity_mean_std()

# beta_diversity_figure()
# beta_diversity_figure_no_zoom_2_colors()


# data_figure_side_by_side(HEALTHY=True)
# data_figure_both()
# data_figure_rel_and_qpcr()

# species_heatmap()

# colorplot()

# data_figure1_cutoff()

# temp()

phylogenetic_heatmap(False)


# figure1()
# plt.show()




