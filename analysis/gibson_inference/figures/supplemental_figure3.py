import matplotlib.pyplot as plt
import seaborn as sns
import mdsine2 as md2
from mdsine2.names import STRNAMES
import numpy as np
import sys
import os
import pandas as pd
import copy
import argparse
from IPython.display import display
from pandas.plotting import table
from scipy.stats import mannwhitneyu

import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
from matplotlib.colors import LogNorm
import matplotlib.colors as colors
from matplotlib.lines import Line2D
from statsmodels.stats.multitest import multipletests, fdrcorrection
from statannot import add_stat_annotation
from matplotlib.colors import ListedColormap

import ete3
from Bio import Phylo

HEATMAP_XLABEL_SIZE = 30
BOXPLOT_TICK_SIZE = 35
BOXPLOT_AXES_SIZE = 40
TITLE_FONTSIZE = 50

def detection_info(taxa_li, abund_data):

    data = np.ravel(abund_data)
    detected = []
    undetected = []

    for i in range(len(taxa_li)):
        otu = taxa_li[i].name
        if abund_data[i] >= 0.0001:
            detected.append(otu)
        else:
            undetected.append(otu)

    return set(detected), set(undetected)

def _make_phylogenetic_tree(tree_fname, names, taxa, ax, fig, figlabel=None, figlabelax=None):

    tree = ete3.Tree(tree_fname)
    tree.prune(names, True)
    tree.write(outfile='tmp/temp.nhx')
    fontsize=18

    taxonomies = ['family', 'order', 'class', 'phylum', 'kingdom']
    suffix_taxa = {'genus': '*',
        'family': '**', 'order': '***', 'class': '****', 'phylum': '*****', 'kingdom': '******'}
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

        n=120
        if len(ax.texts) > 50:
            n = 120
        taxonname += ' ' + taxon.name
        taxonname = ' ' + suffix + taxonname
        text._text = taxonname
        #if "OTU_5" in text._text:
         #   text._text = text._text + " - " + '- ' * (65 -len(text._text))
        text._text = text._text + " - " + '- ' * (n -len(text._text))
        text.set_fontsize(fontsize)

    return ax, taxa_order

def get_gram_list(taxa, names):

    neg_taxanames = []
    g_pos = ["Actinobacteria", "Deinococcus-Thermus"]
    pos_taxanames = []
    for name in names:
        taxon = taxa[name]
        try:
            if md2.is_gram_negative(taxon=taxon):
                neg_taxanames.append(name)
            else:
                pos_taxanames.append(name)
        except:
            if taxon.taxonomy["phylum"] in g_pos:
                pos_taxanames.append(name)
            else:
                neg_taxanames.append(name)

    return pos_taxanames, neg_taxanames

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
    max_rel = np.max(rel_abund)
    min_rel = np.min(np.where(rel_abund==0, np.inf, rel_abund))

    return max_rel, min_rel

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
    inoc_abundance_copy = copy.deepcopy(np.ravel(inoc_abundance_order))

    inoc_abundance_order += vmin - 1e-10,
    inoc_abundance_order = np.where(inoc_abundance_order>=vmax, vmax-1e-6, inoc_abundance_order)

    df = pd.DataFrame(inoc_abundance_order, index=order, columns=["inoc"])
    kwargs = {'norm': LogNorm(vmin=vmin, vmax=vmax)}

    cmap = sns.cubehelix_palette(n_colors=100, as_cmap=True, start=2, rot=0,
        dark=0, light=0.5)
    cmap.set_bad(color='silver')
    cmap.set_under(color='white')

    heatmap = sns.heatmap(df, yticklabels = False, cmap = cmap,
    ax = ax, norm = LogNorm(vmin=vmin, vmax=vmax),
    linewidth = 0.1, linecolor = "indianred", cbar=False)

    ax.set_xticklabels(ax.get_xticklabels(), fontsize = 30,
    rotation = 90, color="indianred")

    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color("indianred")

    return ax

def is_zero_abundance(order, studyset, dict_):
    zero_abundant = {}
    for subj in studyset:
        M = subj.matrix()['abs']
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


def _make_abundance_heatmap(chainname, name, studyset, order, ax, vmax, vmin,
    fig, make_colorbar, filter):

    mcmc = md2.BaseMCMC.load(chainname)
    taxas = [otu.name for otu in filter.taxa]
    clusters = mcmc.graph[STRNAMES.CLUSTERING_OBJ].tolistoflists()
    all_taxa = studyset.taxa
    all_taxa_idx_dict = {all_taxa[i].name : i for i in range(len(all_taxa))}
    zero_abundant_count = is_zero_abundance(order, studyset, all_taxa_idx_dict)

    filter_taxa = [otu.name for otu in filter.taxa]

    rel_abund = np.zeros(len(filter.taxa))
    min_data = []
    for subj in filter:
        M = subj.matrix()['rel']
        start_idx = np.searchsorted(subj.times, 14)
        end_idx = np.searchsorted(subj.times, 21.5)
        rel_abund += np.mean(M[:,start_idx:end_idx], axis=1)
    rel_abund = rel_abund/len(studyset)

    taxa_index_dict = {taxas[i] : i for i in range(len(taxas))}
    rel_abund_order = []

    for otu in order:
        if otu.strip() in taxa_index_dict:
            rel_abund_order.append(rel_abund[taxa_index_dict[otu.strip()]])
        else:
            rel_abund_order.append(np.nan)

    rel_abund_order = np.asarray(rel_abund_order )
    rel_abund_order = np.where(rel_abund_order >= vmax, vmax-1e-6, rel_abund_order)
    #rel_abund_order = np.where(rel_abund_order==0, vmin, rel_abund_order)
    rel_abund_order[rel_abund_order==0] = vmax + 1
    rel_abund_order += vmin-1e-10
    #print("rel:", rel_abund.shape)

    df = pd.DataFrame(rel_abund_order, index=order, columns=[name])

    cmap = sns.cubehelix_palette(n_colors=100, as_cmap=True, start=2, rot=0, dark=0, light=0.5)
    cmap.set_bad(color='gray')
    cmap.set_under(color='white')
    cmap.set_over(color="yellowgreen")

    for x in range(rel_abund_order.shape[0]):
        otu = order[x].strip()
        if otu in zero_abundant_count:
            ax.plot(0.5, x+0.5, marker="o", markersize=5, color="black")

    if not make_colorbar:
        heatmap = sns.heatmap(df, yticklabels = False, cmap = cmap,
        ax = ax, norm = LogNorm(vmin=vmin, vmax=vmax),
        linewidth = 0.1, linecolor = "black", cbar=False)

    else:
        cbaxes = fig.add_axes([0.53, 0.59, 0.02, 0.08]) # left, bottom, width, height

        heatmap = sns.heatmap(df, yticklabels = False, cmap = cmap,
        ax = ax, norm = LogNorm(vmin=vmin, vmax=vmax),
        linewidth = 0.1, linecolor = "black", cbar=True, cbar_ax=cbaxes)

        legend_elements = [Line2D([0], [0], marker= "s", color="white",
        label= "Undetected in days 14 - 21 but consistently\n detected in the experiment",
        markerfacecolor="yellowgreen",
        markersize=35), Line2D([0], [0], marker= "s", color="white",
        label= "Not detected consistently (not detected for 3 \nconsecutive "\
        "time-points in at least 1 mice)",
         markerfacecolor="gray",
        markersize=35), Line2D([0], [0], marker= "o", color="white",
        label= "Undetected throughout the experiment", markerfacecolor="black",
        markersize=25), Line2D([0], [0], marker= "^", color="white",
        label= "Positive Log2 fold change", markerfacecolor="black",
        markersize=25), Line2D([0], [0], marker= "v", color="white",
        label= "Negative Log2 fold change", markerfacecolor="black",
        markersize=25)]

        lgd_ax = fig.add_axes([0.71, 0.605, 0.06, 0.06])
        lgd_ax.legend(handles=legend_elements, loc="center", fontsize=35,
            frameon=False, labelspacing=2)
        lgd_ax = _remove_border(lgd_ax)

        cbar = heatmap.collections[0].colorbar
        cbar.ax.set_title('Relative\nAbundance\n', fontsize=40, fontweight='bold')
        cbar.ax.tick_params(labelsize=40, length=10, which="major")
        cbar.ax.tick_params(length=0, which="minor")

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=HEATMAP_XLABEL_SIZE,
    rotation = 90)

    for _, spine in ax.spines.items():
        spine.set_visible(True)

    return ax

def _show_deseq_results(order, deseq_data, ax):

    data_df = []
    for x in range(len(order)):
        otu = order[x].strip()
        data_df.append(np.nan)
    cmap = sns.cubehelix_palette(n_colors=100, as_cmap=True, start=2,
        rot=0, dark=0, light=0.5)
    cmap.set_bad("White")

    df = pd.DataFrame(data_df, index=order, columns=["Fold\nChange"])
    heatmap = sns.heatmap(df, yticklabels=False, xticklabels=True, cmap=cmap,
    ax=ax, cbar=False, linewidth=0.1, linecolor="black")
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=HEATMAP_XLABEL_SIZE,
        rotation = 90)


    for x in range(len(order)):
        otu = order[x].strip()
        if otu in deseq_data:
            if deseq_data[otu] > 0:
                ax.plot(0.5, x+0.5, marker="^", markersize=10, color="black")
            else:
                ax.plot(0.5, x+0.5, marker="v", markersize=10, color="black")
    #ax.patch.set_alpha(0)
    return ax

def consistency_filtering(subjset, dtype, threshold,min_num_consecutive,
    min_num_subjects,colonization_time=None, union_other_consortia=None):

    if union_other_consortia is not None:
        taxa_to_keep = set()
        for subjset_temp in [subjset, union_other_consortia]:
            subjset_temp = consistency_filtering(subjset_temp, dtype=dtype,
                threshold=threshold, min_num_consecutive=min_num_consecutive,
                colonization_time=colonization_time, min_num_subjects=min_num_subjects,
                union_other_consortia=None)
            for taxon_name in subjset_temp.taxa.names:
                taxa_to_keep.add(taxon_name)
        to_delete = []
        for aname in subjset.taxa.names:
            if aname not in taxa_to_keep:
                to_delete.append(aname)
    else:
        # Everything is fine, now we can do the filtering
        talley = np.zeros(len(subjset.taxa), dtype=int)
        for i, subj in enumerate(subjset):
            matrix = subj.matrix()[dtype]
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

    return talley

def _show_filter_count(subjset, order, ax, N):

    dtype = "rel"
    thresh = 0.0001
    num_consecutive = 3
    num_subjects = 1
    colonization_time = 0
    taxas = subjset.taxa
    all_taxa = {taxas[i].name: i for i in range(len(taxas))}
    subjset_consistency_filtering = consistency_filtering(subjset,
        dtype, thresh, num_consecutive, num_subjects, colonization_time)

    order_data = [subjset_consistency_filtering[all_taxa[taxa.strip()]] for taxa
       in order]
    col_names = ["# mice"]
    df = pd.DataFrame(order_data, index=order, columns=col_names)

    heatmap = sns.heatmap(df, yticklabels=False, ax=ax,
       cmap=ListedColormap(['white']), cbar=False, annot=True, linecolor="black",
        annot_kws={"fontsize":18}, linewidths=0.1)
    for t in ax.texts:
        t.set_text(t.get_text() + " / {}".format(N))
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=HEATMAP_XLABEL_SIZE ,
        rotation = 90)

    return ax

def _show_deseq_results(order, deseq_data, ax):

    data_df = []
    for x in range(len(order)):
        otu = order[x].strip()
        data_df.append(np.nan)
    cmap = sns.cubehelix_palette(n_colors=100, as_cmap=True, start=2,
        rot=0, dark=0, light=0.5)
    cmap.set_bad("White")

    df = pd.DataFrame(data_df, index=order, columns=["Fold\nChange"])
    heatmap = sns.heatmap(df, yticklabels=False, xticklabels=True, cmap=cmap,
    ax=ax, cbar=False, linewidth=0.1, linecolor="black")

    ax.set_xticklabels(ax.get_xticklabels(), fontsize = 25,
        rotation = 90)


    for x in range(len(order)):
        otu = order[x].strip()
        if otu in deseq_data:
            if deseq_data[otu] > 0:
                ax.plot(0.5, x+0.5, marker="^", markersize=10, color="black")
            else:
                ax.plot(0.5, x+0.5, marker="v", markersize=10, color="black")
    #ax.patch.set_alpha(0)
    return ax

def make_table(inoc_det_taxa, inoc_undet_taxa, filter_taxa, all_taxa):

    not_inoc_taxa = all_taxa - inoc_det_taxa
    not_filter_taxa = all_taxa - filter_taxa

    green_to_green = inoc_det_taxa.intersection(filter_taxa)
    green_to_grey = inoc_det_taxa.intersection(not_filter_taxa)

    grey_to_green = not_inoc_taxa.intersection(filter_taxa)
    grey_to_grey = not_inoc_taxa.intersection(not_filter_taxa)

    cols = ["Detected in Inoculum", "Consistently Detected", "# OTUs", "Percentage"]
    criteria1 = ["Yes", "Yes", "No",  "No"]
    criteria2 = ["Yes", "No", "Yes", "No"]
    quantity = [len(green_to_green), len(green_to_grey), len(grey_to_green),
        len(grey_to_grey)]
    percentage = [len(green_to_green)/len(inoc_det_taxa)*100, len(green_to_grey)/
        len(inoc_det_taxa)*100, len(grey_to_green)/len(not_inoc_taxa)*100,
        len(grey_to_grey)/len(not_inoc_taxa)*100]
    #remarks = ["Green to Green", "Green to Grey", "White to Green", "White to Grey"]
    percentage=["{:.3f}".format(i) for i in percentage]

    table_li = [criteria1, criteria2, quantity, percentage]
    table_df = pd.DataFrame(table_li, index=None).transpose()
    table_df.columns = cols
    table_df.index=[1, 2, 3, 4]
    table_df.to_csv("table.csv", sep=",")

    return table_df

def make_combined_table_inoc(healthy_inoc_taxa, uc_inoc_taxa, healthy_filter_taxa,
    uc_filter_taxa):

    healthy_green_to_green = healthy_inoc_taxa.intersection(healthy_filter_taxa)
    uc_green_to_green = uc_inoc_taxa.intersection(uc_filter_taxa)
    healthy_green_to_white = healthy_inoc_taxa - healthy_green_to_green
    uc_green_to_white = uc_inoc_taxa - uc_green_to_green

    cols = ["Donor", "Consistently Detected", "#OTUs", "Percentage"]
    donor_li = ["Healthy", "Healthy", "UC", "UC"]
    criteria_li = ["Yes", "No", "Yes", "No"]
    quantity = [len(healthy_green_to_green), len(healthy_green_to_white),
        len(uc_green_to_green), len(uc_green_to_white)]
    percentage = [len(healthy_green_to_green) / len(healthy_inoc_taxa)*100,
        len(healthy_green_to_white) / len(healthy_inoc_taxa)*100, len(uc_green_to_green)/
         len(uc_inoc_taxa)*100, len(uc_green_to_white)/len(uc_inoc_taxa)*100]

    table_li = [donor_li, criteria_li, quantity, percentage]
    table_df = pd.DataFrame(table_li, index=None).transpose()
    table_df.columns = cols
    table_df.index=[1, 2, 3, 4]
    #table_df.to_csv("table.csv", sep=",")

    return table_df

def make_combined_table_colonization(healthy_inoc_taxa, uc_inoc_taxa,
    healthy_filter_taxa, uc_filter_taxa):

    healthy_green_to_green = healthy_inoc_taxa.intersection(healthy_filter_taxa)
    uc_green_to_green = uc_inoc_taxa.intersection(uc_filter_taxa)
    healthy_white_to_green = healthy_filter_taxa - healthy_green_to_green
    uc_white_to_green = uc_filter_taxa - uc_green_to_green

    cols = ["Donor", "Detected in Inoculum", "#OTUs", "Percentage"]
    donor_li = ["Healthy", "Healthy", "UC", "UC"]
    criteria_li = ["Yes", "No", "Yes", "No"]
    quantity = [len(healthy_green_to_green), len(healthy_white_to_green),
        len(uc_green_to_green), len(uc_white_to_green)]
    percentage = [len(healthy_green_to_green) / len(healthy_filter_taxa)*100,
        len(healthy_white_to_green) / len(healthy_filter_taxa)*100, len(uc_green_to_green)/
         len(uc_filter_taxa)*100, len(uc_white_to_green)/len(uc_filter_taxa)*100]

    table_li = [donor_li, criteria_li, quantity, percentage]
    table_df = pd.DataFrame(table_li, index=None).transpose()
    table_df.columns = cols
    table_df.index=[1, 2, 3, 4]
    #table_df.to_csv("table.csv", sep=",")

    return table_df

def _make_bar_graph(data_df, ax, title, hue):

    percentage = data_df["Percentage"].to_numpy()
    percentage = [float(i) for i in percentage]
    percentage = [percentage[0],percentage[2], percentage[1], percentage[3]]
    bar_graph = sns.barplot(y="#OTUs", x="Donor",
        hue=hue, data=data_df, ax=ax)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=30, title_fontsize=30,
        title=hue, bbox_to_anchor=(1.02, 1), loc=2)
    ax.tick_params(axis='both', which='major', labelsize=BOXPLOT_TICK_SIZE)
    ax.set_xlabel("Donor", fontsize=BOXPLOT_AXES_SIZE, labelpad=6)
    ax.set_ylabel("# OTUs", fontsize=BOXPLOT_AXES_SIZE, labelpad=6)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold", loc="left")
    #ax.set_ylim(0, np.max(data_df["# OTUs"].to_numpy()))

    i = 0
    for p in bar_graph.patches:
        per = float(percentage[i])
        bar_graph.annotate("{:.3f}%".format(per), (p.get_x() + p.get_width() / 2.,
            p.get_height()/2+2), ha="center", va="center", size=30,
            textcoords = 'offset points', xytext = (0, -12))
        i += 1
    return ax

def get_experiment_data(subjset, all_data=True):

    rel_abund = np.zeros(len(subjset.taxa))

    for subj in subjset:
        M = subj.matrix()["rel"]
        if all_data:
            rel_abund += np.mean(M, axis=1)
        else:
            start_idx = np.searchsorted(subj.times, 14)
            end_idx = np.searchsorted(subj.times, 21.5)
            rel_abund += np.mean(M[:,start_idx:end_idx], axis=1)

    rel_abund /= len(subjset)
    return rel_abund

def phylogenetic_heatmap(inoc_pkl, healthy_pkl_all, uc_pkl_all, chain_healthy,
    chain_uc, outfile, tree_fname, taxa, healthy_filter_pkl, uc_filter_pkl,
    deseq):

    healthy_inoc_det, healthy_inoc_undet = detection_info(inoc_pkl.taxa,
        inoc_pkl["Healthy"].matrix()["rel"])
    uc_inoc_det, uc_inoc_undet = detection_info(inoc_pkl.taxa,
        inoc_pkl["Ulcerative Colitis"].matrix()["rel"])

    inoc_det_union = healthy_inoc_det.union(uc_inoc_det)
    print(len(healthy_inoc_det.union(uc_inoc_det)))
    print("inoc:", len(healthy_inoc_det), len(uc_inoc_det))

    healthy_filter_taxa = [otu.name for otu in healthy_filter_pkl.taxa]
    uc_filter_taxa = [otu.name for otu in uc_filter_pkl.taxa]
    expt_union = set(healthy_filter_taxa).union(set(uc_filter_taxa))
    print("len taxa:", len(healthy_filter_taxa), len(uc_filter_taxa))
    print("expt union:", len(expt_union))
    print("inoc det union:", len(inoc_det_union))
    all_taxa = inoc_det_union.union(expt_union)
    print("len all taxa:", len(all_taxa))
    healthy_summary = make_table(healthy_inoc_det, healthy_inoc_undet,
        set(healthy_filter_taxa), all_taxa)
    uc_summary = make_table(uc_inoc_det, uc_inoc_undet,
        set(uc_filter_taxa), all_taxa)
    combined_inoc_summary = make_combined_table_inoc(healthy_inoc_det, uc_inoc_det,
        set(healthy_filter_taxa), set(uc_filter_taxa))
    combined_colonization_summary = make_combined_table_colonization(healthy_inoc_det,
        uc_inoc_det, set(healthy_filter_taxa), set(uc_filter_taxa))
    grampos_taxanames, gramneg_taxanames = get_gram_list(inoc_pkl.taxa, all_taxa)


    fig = plt.figure(figsize=(60, 90))
    gs = fig.add_gridspec(104,104)

    pert_nrows = 0
    grampos_nrows = 104 - pert_nrows
    gramneg_nrows = int(grampos_nrows * len(gramneg_taxanames)/len(grampos_taxanames))
    ntwk_buffer = 0
    ntwk_row_start = 10 + gramneg_nrows + ntwk_buffer
    ntwk_nrows = int((100 - ntwk_row_start)/2)

    tree_ncols = 27
    tree_branch_ncols = 5
    heatmap_width = 50 - tree_ncols

    mcmc = md2.BaseMCMC.load(chain_healthy)
    healthy_nclusters = len(mcmc.graph[STRNAMES.CLUSTERING_OBJ])
    mcmc = md2.BaseMCMC.load(chain_uc)
    uc_nclusters = len(mcmc.graph[STRNAMES.CLUSTERING_OBJ])

    uc_ncols = int(heatmap_width*uc_nclusters/(uc_nclusters + healthy_nclusters))
    healthy_ncols = heatmap_width - uc_ncols -2

    ax_grampos_tree = fig.add_subplot(gs[ pert_nrows:pert_nrows+grampos_nrows,
        0:tree_branch_ncols])
    ax_grampos_tree_full = fig.add_subplot(gs[pert_nrows:pert_nrows+grampos_nrows,
        0:tree_branch_ncols], facecolor='none')

    ax_grampos_healthy_inoc = fig.add_subplot(gs[pert_nrows:pert_nrows+grampos_nrows,
        tree_ncols+3: tree_ncols + 5])
    ax_grampos_uc_inoc = fig.add_subplot(gs[ pert_nrows:pert_nrows+grampos_nrows,
        tree_ncols+healthy_ncols+3:tree_ncols+healthy_ncols + 5])

    ax_grampos_healthy_abund = fig.add_subplot(gs[pert_nrows:
        pert_nrows+grampos_nrows, tree_ncols+5:tree_ncols+7])
    ax_grampos_healthy_count = fig.add_subplot(gs[pert_nrows:pert_nrows+grampos_nrows,
        7+tree_ncols :9+tree_ncols])
    ax_grampos_uc_abund = fig.add_subplot(gs[pert_nrows:pert_nrows+grampos_nrows,
        tree_ncols+healthy_ncols + 5: tree_ncols+healthy_ncols + 7])
    ax_grampos_uc_count = fig.add_subplot(gs[pert_nrows:pert_nrows+grampos_nrows,
        7+tree_ncols+healthy_ncols:9+tree_ncols+healthy_ncols])

    ax_deseq_pos = fig.add_subplot(gs[pert_nrows:pert_nrows+grampos_nrows,
        tree_ncols+5 + healthy_ncols//2: tree_ncols + 5 +healthy_ncols//2 + 2])

    # gram negative
    ax_gramneg_tree = fig.add_subplot(gs[pert_nrows:pert_nrows+gramneg_nrows,
        53:53+tree_branch_ncols])
    ax_gramneg_tree_full = fig.add_subplot(gs[pert_nrows :pert_nrows+grampos_nrows,
        53:53+tree_branch_ncols], facecolor='none')

    ax_gramneg_healthy_inoc = fig.add_subplot(gs[pert_nrows:pert_nrows+gramneg_nrows,
        57+tree_ncols:59+tree_ncols])
    ax_gramneg_uc_inoc = fig.add_subplot(gs[pert_nrows:pert_nrows+gramneg_nrows,
        57+tree_ncols+healthy_ncols:59+tree_ncols+healthy_ncols])

    ax_gramneg_healthy_abund = fig.add_subplot(gs[pert_nrows :
       pert_nrows+gramneg_nrows, 59+tree_ncols:61+tree_ncols])
    ax_gramneg_healthy_count = fig.add_subplot(gs[pert_nrows:pert_nrows+gramneg_nrows,
        61+tree_ncols:63+tree_ncols])
    ax_gramneg_uc_abund = fig.add_subplot(gs[pert_nrows:pert_nrows+gramneg_nrows,
        59+tree_ncols+healthy_ncols:61+tree_ncols+healthy_ncols])
    ax_gramneg_uc_count = fig.add_subplot(gs[pert_nrows:pert_nrows+gramneg_nrows,
        61+tree_ncols+healthy_ncols:63+tree_ncols+healthy_ncols])

    ax_deseq_neg = fig.add_subplot(gs[pert_nrows:pert_nrows+gramneg_nrows,
        59+tree_ncols+healthy_ncols//2 :59+tree_ncols+healthy_ncols//2+2])

    ax_healthy_table = fig.add_subplot(gs[47:57 ,55:90])
    ax_uc_table = fig.add_subplot(gs[60:70,55 :90])
    ax_box_filter = fig.add_subplot(gs[73:87, 55:90])
    ax_box_inoc = fig.add_subplot(gs[90:104, 55:90])


    max_uc, min_uc = get_scale(chain_uc)
    max_healthy, min_healthy = get_scale(chain_healthy)
    max_= max(max_uc, max_healthy)
    min_= max(min_uc, min_healthy)
    #print("max min:", max_, min_)
    lim = 1e-6
    if min_ < lim:
        min_=lim
    if max_ > 1e-1:
        max_=1e-1

    ax_grampos_tree, grampos_taxaname_order = _make_phylogenetic_tree(
        tree_fname=tree_fname, names=grampos_taxanames, taxa=taxa, fig=fig,
        ax=ax_grampos_tree, figlabel='E', figlabelax=ax_grampos_tree_full)
    ax_gramneg_tree, gramneg_taxaname_order = _make_phylogenetic_tree(
        tree_fname=tree_fname, names=gramneg_taxanames, taxa=taxa, fig=fig,
        ax=ax_gramneg_tree, figlabel='F', figlabelax=ax_gramneg_tree_full)

    print("Gram Positive Inoculum")
    ax_grampos_healthy_inoc = _make_inoc_heatmap(pkl=inoc_pkl, subject_name="Healthy",
        order=grampos_taxaname_order, ax=ax_grampos_healthy_inoc, fig=fig,
        vmax=max_, vmin=min_, make_colorbar=False)
    ax_grampos_uc_inoc = _make_inoc_heatmap(pkl=inoc_pkl, subject_name="Ulcerative Colitis",
        order=grampos_taxaname_order, ax=ax_grampos_uc_inoc, fig=fig,
        vmax=max_, vmin=min_, make_colorbar=False)

    print("Gram Negative Inoculum")
    ax_gramneg_healthy_inoc = _make_inoc_heatmap(pkl=inoc_pkl, subject_name="Healthy",
        order=gramneg_taxaname_order, ax=ax_gramneg_healthy_inoc, fig=fig,
        vmax=max_, vmin=min_, make_colorbar=False)
    ax_gramneg_uc_inoc = _make_inoc_heatmap(pkl=inoc_pkl, subject_name="Ulcerative Colitis",
        order=gramneg_taxaname_order, ax=ax_gramneg_uc_inoc, fig=fig,
        vmax=max_, vmin=min_, make_colorbar=True)

    print("Gram Positive Post Colonization")
    ax_grampos_healthy_abund = _make_abundance_heatmap(chain_healthy, "Healthy",
        studyset=healthy_pkl_all, order=grampos_taxaname_order,
        ax=ax_grampos_healthy_abund,vmax=max_, vmin=min_, fig=fig, make_colorbar=False,
        filter=healthy_filter_pkl)
    ax_grampos_uc_abund = _make_abundance_heatmap(chain_uc, "UC", studyset=uc_pkl_all,
        order=grampos_taxaname_order, ax=ax_grampos_uc_abund,vmax=max_, vmin=min_,
        fig=fig, make_colorbar=False, filter=uc_filter_pkl)

    print("Gram Negative Post Colonization")
    ax_gramneg_healthy_abund = _make_abundance_heatmap(chain_healthy, "Healthy",
        studyset=healthy_pkl_all, order=gramneg_taxaname_order,
        ax=ax_gramneg_healthy_abund,vmax=max_, vmin=min_, fig=fig, make_colorbar=False,
        filter=healthy_filter_pkl)
    ax_gramneg_uc_abund = _make_abundance_heatmap(chain_uc, "UC", studyset=uc_pkl_all,
        order=gramneg_taxaname_order, ax=ax_gramneg_uc_abund,vmax=max_, vmin=min_,
        fig=fig, make_colorbar=True, filter=uc_filter_pkl)

    ax_deseq_pos = _show_deseq_results(grampos_taxaname_order, deseq,
        ax_deseq_pos)
    ax_deseq_neg = _show_deseq_results(gramneg_taxaname_order, deseq,
        ax_deseq_neg)

    ax_grampos_healthy_count = _show_filter_count(healthy_pkl_all,
        grampos_taxaname_order, ax_grampos_healthy_count, len(healthy_pkl_all))
    ax_grampos_uc_count = _show_filter_count(uc_pkl_all, grampos_taxaname_order,
        ax_grampos_uc_count, len(uc_pkl_all))
    ax_gramneg_healthy_count = _show_filter_count(healthy_pkl_all,
        gramneg_taxaname_order, ax_gramneg_healthy_count, len(healthy_pkl_all))
    ax_gramneg_uc_count = _show_filter_count(uc_pkl_all, gramneg_taxaname_order,
        ax_gramneg_uc_count, len(uc_pkl_all))


    #ax_healthy_table = _make_bar_graph(healthy_summary, ax_healthy_table, "C")
    #ax_uc_table = _make_bar_graph(uc_summary, ax_uc_table, "D")

    ax_healthy_table = _make_bar_graph(combined_inoc_summary, ax_healthy_table, "C",
        "Consistently Detected")
    ax_uc_table = _make_bar_graph(combined_colonization_summary, ax_uc_table, "D",
        "Detected in Inoculum")
    print(combined_inoc_summary)
    print(combined_colonization_summary)

    healthy_abund_data = get_experiment_data(healthy_filter_pkl, all_data=True)
    healthy_index = {healthy_filter_taxa[i] : i for i in range(len(healthy_filter_taxa))}

    uc_abund_data = get_experiment_data(uc_filter_pkl, all_data=True)
    uc_index = {uc_filter_taxa[i] : i for i in range(len(uc_filter_taxa))}

    column_names1 = ["OTUS", "Detected in Inoculum", "Donor", "Relative Abundance"]
    column_names2 = ["OTUS", "Consistently Detected", "Donor", "Relative Abundance"]
    healthy_colonized_df = format_box_plot_data(healthy_abund_data,
        healthy_filter_taxa, healthy_inoc_det, healthy_index, column_names1, "Healthy")
    uc_colonized_df = format_box_plot_data(uc_abund_data,
        uc_filter_taxa, uc_inoc_det, uc_index, column_names1, "UC")
    print("df colonization:", healthy_colonized_df.shape, uc_colonized_df.shape)
    all_colonized_df = pd.concat([healthy_colonized_df, uc_colonized_df])
    colonized_test = perform_significance_test(healthy_colonized_df,
        uc_colonized_df, "Detected in Inoculum")

    healthy_inoc_data = np.ravel(inoc_pkl["Healthy"].matrix()["rel"])
    uc_inoc_data = np.ravel(inoc_pkl["Ulcerative Colitis"].matrix()["rel"])

    inoc_taxa = [otu.name for otu in inoc_pkl.taxa]
    inoc_index_dict = {inoc_taxa[i]:i for i in range(len(inoc_taxa))}

    healthy_inoc_df = format_box_plot_data(healthy_inoc_data, healthy_inoc_det,
        healthy_filter_taxa, inoc_index_dict, column_names2, "Healthy")
    uc_inoc_df = format_box_plot_data(uc_inoc_data, uc_inoc_det,
        uc_filter_taxa, inoc_index_dict, column_names2, "UC")
    inoc_test = perform_significance_test(healthy_inoc_df,
        uc_inoc_df, "Consistently Detected")
    print("df inoculum:", healthy_inoc_df.shape, uc_inoc_df.shape)
    all_filter_df = pd.concat([healthy_inoc_df, uc_inoc_df])

    ax_gramneg_tree = _remove_border(ax_gramneg_tree)
    ax_grampos_tree = _remove_border(ax_grampos_tree)
    ax_gramneg_tree_full = _remove_border(ax_gramneg_tree_full)
    ax_grampos_tree_full = _remove_border(ax_grampos_tree_full)

    box_plot(ax_box_filter, all_colonized_df, "E", "Detected in Inoculum",
        "Rel Abundance (post colonization)", colonized_test[1])
    box_plot(ax_box_inoc, all_filter_df, "F", "Consistently Detected",
        "Rel Abunance (inoculum)", inoc_test[1])


    fig.text(x=0.22, y=0.89, s="Gram+", fontsize=40, fontweight="bold")
    fig.text(x=0.62, y=0.89, s="Gram-", fontsize=40, fontweight="bold")

    fig.text(x=0.15, y=0.90, s="A", fontsize=50, fontweight="bold")
    fig.text(x=0.55, y=0.90, s="B", fontsize=50, fontweight="bold")


    suffix_taxon = {'genus': '*',
        'family': '**', 'order': '***', 'class': '****', 'phylum': '*****', 'kingdom': '******'}
    text = '$\\bf{Taxonomy}\, \\bf{ Key}$\n'
    for taxon in suffix_taxon:
        text += ' {} : {},'.format(suffix_taxon[taxon], taxon)
    fig.text(x=0.515, y=0.55, s=text[:-1], fontsize=40)


    fig.savefig(outfile, bbox_inches="tight")

def perform_significance_test(healthy_df, uc_df, criteria):

    healthy_green_data = healthy_df.loc[healthy_df[criteria] ==
        "Yes"]["Relative Abundance"].to_numpy()
    healthy_white_data = healthy_df.loc[healthy_df[criteria] ==
        "No"]["Relative Abundance"].to_numpy()
    uc_green_data = uc_df.loc[uc_df[criteria] == "Yes"]["Relative Abundance"].to_numpy()
    uc_white_data = uc_df.loc[uc_df[criteria] == "No"]["Relative Abundance"].to_numpy()

    healthy_mann_test = mannwhitneyu(healthy_green_data, healthy_white_data)
    uc_mann_test = mannwhitneyu(uc_green_data, uc_white_data)

    p_mann = [healthy_mann_test[1], uc_mann_test[1]]
    hypothesis_test = multipletests(p_mann, alpha=0.05, method="fdr_bh", is_sorted=False)

    return hypothesis_test

def format_box_plot_data(data, otu_li, otu_other_li, otu_index, col_names,
    donor_name, all_data=True):

    detected_info = []
    abundance = []
    donor = []
    otus = []

    for otu in otu_li:
        index = otu_index[otu]
        abundance.append(data[index])
        otus.append(otu)
        donor.append(donor_name)
        if otu in otu_other_li:
            detected_info.append("Yes")
        else:
            detected_info.append("No")

    df = pd.DataFrame(list(zip(otus, detected_info, donor, abundance)),
        columns=col_names)
    return df

def box_plot(axes, df, title, hue, ylab, pvalues):

    axes.xaxis.grid(False)
    axes.yaxis.grid(True)

    #axes.set_title(title, loc="left", fontweight="bold", fontsize=20)

    axes.set_yscale("log")

    sns.boxplot(y="Relative Abundance", x="Donor",hue=hue,
        data=df, whis=[2.5, 97.5], width=.75,
        showfliers=False, ax=axes, palette="Set3", linewidth=4)
    sns.stripplot(y="Relative Abundance", x="Donor", hue=hue,
        data=df, size=4,
        linewidth=2, alpha=0.5, ax=axes, palette="Set3", dodge=True) #, color=".3"
    handles, labels = axes.get_legend_handles_labels()
    axes.legend(handles[0:2], labels[0:2], bbox_to_anchor=(1.02, 1), loc=2,
        borderaxespad=0., fontsize=30, title=hue,
        title_fontsize=30)

    axes.set_ylabel(ylab, fontsize=BOXPLOT_AXES_SIZE, labelpad=6)
    axes.set_xlabel("Donor", fontsize=BOXPLOT_AXES_SIZE, labelpad=6)
    axes.tick_params(axis='both', labelsize=BOXPLOT_TICK_SIZE)
    axes.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold", loc="left")

    add_stat_annotation(axes, data=df, x="Donor", y="Relative Abundance", hue=hue,
        box_pairs=[(("Healthy", "Yes"), ("Healthy", "No")), (("UC", "Yes"), ("UC", "No"))],
        perform_stat_test=False, text_format="star", loc="inside", verbose=2,
        pvalues=pvalues, fontsize=40, linewidth=3)

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
    parser.add_argument("-deseq", "--deseq_data", required = "True",
        help = "path to the deseq_results")

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    chain_loc = "/data/cctm/darpa_perturbation_mouse_study/MDSINE2_Paper/output/"\
        "mdsine2/fixed_clustering_mixed_prior/"
    chain_healthy = chain_loc + 'healthy-seed0-mixed/mcmc.pkl'
    chain_uc = chain_loc + 'uc-seed0-mixed/mcmc.pkl'
    output_loc = "gibson_inference/figures/output_figures/"
    os.makedirs(output_loc, exist_ok=True)
    outfile = output_loc + 'supplementary_figure3.pdf'
    tree_fname = '/data/cctm/darpa_perturbation_mouse_study/MDSINE2_Paper/'\
    'gibson_dataset/files/phylogenetic_placement_OTUs/phylogenetic_tree_only_query.nhx'

    inoc_study = md2.Study.load(args.study_inoc)
    healthy_study_all = md2.Study.load(args.study_healthy)
    uc_study_all = md2.Study.load(args.study_uc)

    deseq_df = pd.read_csv(args.deseq_data, index_col=0)

    healthy_filtered_study = md2.Study.load(args.detected_study_healthy)
    uc_filtered_study = md2.Study.load(args.detected_study_uc)

    taxa = inoc_study.taxa

    print(len(healthy_filtered_study.taxa), len(uc_filtered_study.taxa))
    detected_taxa = [otu.name for otu in healthy_filtered_study.taxa]
    colonized_taxa = [otu.name for otu in md2.BaseMCMC.load(chain_healthy).graph.data.taxa]

    print(len(detected_taxa), len(colonized_taxa))
    print(len(set(detected_taxa).intersection(set(colonized_taxa))))

    deseq_values = {deseq_df.index[i]:deseq_df["log2FoldChange"][i] for i in range(
        len(deseq_df.index))}

    phylogenetic_heatmap(inoc_pkl=inoc_study, healthy_pkl_all=healthy_study_all,
        uc_pkl_all=uc_study_all, chain_healthy=args.chain_healthy, chain_uc=args.chain_uc,
        outfile=outfile, tree_fname=tree_fname, taxa=taxa, healthy_filter_pkl=
        healthy_filtered_study, uc_filter_pkl=uc_filtered_study, deseq=deseq_values)
