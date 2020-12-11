'''
#plots the raw data
to : run
python supplemental_figure2.py -filter1 "../files/figures/healthy_7.txt"\
        -filter2 "../files/figures/uc_7.txt"\
        -file1 "../../processed_data/gibson_healthy_agg_taxa.pkl"\
        -file2 "../../processed_data/gibson_uc_agg_taxa.pkl"

'''

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import pylab as pl
import mdsine2 as md2
import argparse

PERTURBATION_COLOR = 'orange'

def parse_args():

    parser = argparse.ArgumentParser(description = "files needed for making"\
    "supplemental figure 2")
    parser.add_argument("-filter1", "--healthy_filter", required = "True",
    help = ".txt file containing names of OTUs that pass filtering in Healthy")
    parser.add_argument("-filter2", "--uc_filter", required = "True",
    help = ".txt file containing names of OTUs that pass filtering in UC")
    parser.add_argument("-file1", "--healthy_pkl", required = "True",
        help = "pickled md2.base.Study file for healthy subjects")
    parser.add_argument("-file2", "--uc_pkl", required = "True",
        help = "pickled md2.base.Study file for UC subjects")

    return parser.parse_args()

def count_times(df, times, times_cnts, t2idx):
    """
    count the number of times data is collected at a given sample time point
    """

    for col in df.columns :
        if col in times:
            times_cnts[t2idx[col]] += 1

    return times_cnts

def add_unequal_col_dataframes(df, dfother, times, times_cnts, t2idx):

    times_cnts = count_times(dfother, times, times_cnts, t2idx)
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

def combine_dfs(subjset, dtype):

    times = []
    for subj in subjset:
        times = np.append(times, subj.times)
    times = np.sort(np.unique(times))
    t2idx = {}
    for i, t in enumerate(times):
        t2idx[t] = i
    times_cnts = np.zeros(len(times))

    df = None
    for subj in subjset:
        dfnew = subj.df()[dtype]
        df, times_cnts = add_unequal_col_dataframes(df = df, dfother = dfnew,
        times = times, times_cnts = times_cnts, t2idx = t2idx)
    df = df / times_cnts
    return df

def get_top_abundance_data(subjset_he, subjset_uc):
    """
       obtains the name of the top 400 most abundant species

       @parameters
       ------------------------------------------------------------------------
       subjset1 : mdsine2.base.SubjectSet

       @returns
       ------------------------------------------------------------------------
       [str] : List containing the names of 400 most abundance species
    """
    subjset = []
    for subj in subjset_he:
        subjset.append(subj)

    for subj in subjset_uc:
        subjset.append(subj)

    top_ = 210
    combined_df = combine_dfs(subjset, "abs")
    df_np = combined_df.to_numpy()
    total_abund = df_np.sum(axis = 1)
    idxs = np.argsort(total_abund)[::-1]
    top_idxs = idxs[0 : top_]
    sorted_idxs = np.sort(top_idxs)
    final_li = ["OTU_" + str(i + 1) for i in sorted_idxs]

    np_healthy_200 = combine_dfs(subjset_he, "abs").to_numpy()[sorted_idxs]
    np_uc_200 = combine_dfs(subjset_uc, "abs").to_numpy()[sorted_idxs]

    df_uc = pd.DataFrame(np_uc_200, index = final_li,
            columns = combined_df.columns)
    df_healthy = pd.DataFrame(np_healthy_200, index = final_li,
            columns = combined_df.columns)

    return df_healthy, df_uc

def get_filtered_otu_names(filename):
    """
    reads the .txt (filename) file containing the names of OTUs that passed
    the filter and returns the names as a list
    """

    pass_filter = {}
    file = open(filename, "r")
    rank = 0
    final_li = []
    for line in file:
        if len(line) != 0:
            pass_filter[line.strip()] = rank
            rank += 1
            final_li.append(line.strip())

    return final_li

def get_max(data1, data2):
    """
       returns the maximum of data1 and data2
    """

    max_ = 0
    if np.amax(data1) > np.amax(data2):
        max_ = np.amax(data1)
    else:
        max_ = np.amax(data2)

    return max_

def get_mask1(df, filter_):
    """
       obtains the mask (boolean array) for asvs that pass the filter

       @parameters
       ------------------------------------------------------------------------
       df : (pandas DataFrame)
       filter_ : ([str]) a list containing names of ASVs that pass the filter

       @returns
       ------------------------------------------------------------------------
       (pd.DataFrame)
    """
    mask_array = []
    np_df = df.to_numpy()
    index = df.index
    shape = np_df.shape
    for i in range(len(index)):
        otu_name = index[i].split(",")[0].replace(" ", "_")
        if otu_name in filter_:
            mask_array.append([True] * shape[1])
        else:
            mask_array.append([False] * shape[1])

    return np.asarray(mask_array)

def get_mask2(df):
    """
       obtains the mask (boolean array) for asvs for which abundance is 0

       @parameters
       ------------------------------------------------------------------------
       df : (pandas DataFrame)

       @returns
       ------------------------------------------------------------------------
       (pd.DataFrame)
    """
    mask_array = []
    np_df = df.to_numpy()
    row_sum = np.sum(np_df, axis = 1)
    shape = np_df.shape
    format_array = []
    for i in range(len(row_sum)):
        if row_sum[i] == 0:
            mask_array.append([False] * shape[1])
        else:
            mask_array.append([True] * shape[1])
        format_array.append(["x"] * shape[1])

    return np.asarray(mask_array), np.asarray(format_array)

def add_perturbation_label(ax, perturbations, subj, times, textcolor='black',
     textsize=None, alpha=0.25, label=True):

    if md2.issubject(subj):
        subj = subj.name
    if not md2.isstr(subj):
        raise ValueError('`Cannot recognize {}'.format(subj))
    if perturbations is None or len(perturbations) == 0:
        return ax

    pert_locs = []
    pert_names = []
    for pidx, perturbation in enumerate(perturbations):

        if subj not in perturbation.starts or subj not in perturbation.ends:
            continue
        ax.axvspan(
            xmin = perturbation.starts[subj],
            xmax = perturbation.ends[subj],
            facecolor = PERTURBATION_COLOR,
            alpha=alpha, zorder=-10000)

        pert_locs.append((times.index(perturbation.ends[subj]) +
        times.index(perturbation.starts[subj])) / 2)
        name = perturbation.name
        if name is None:
            name = 'pert{}'.format(pidx)
        pert_names.append(name)

    if label:
        # Set the names on the top x-axis
        ax2 = ax.twiny()

        left,right = ax.get_xlim()
        ax2.set_xlim(ax.get_xlim())
        pl = []
        pn = []
        for idx, loc in enumerate(pert_locs):
            if loc > left and loc < right:
                pl.append(loc)
                pn.append(pert_names[idx])
        ax2.set_xticks(pl)
        ax2.set_xticklabels(pn)
        ax2.tick_params('x', which='both', length=0, colors=textcolor,
            labelsize=textsize)

    return ax

def draw_heatmap(data_healthy, data_uc, uc_filter, healthy_filter, max_, min_,
    name, subjset, times, leg_text, title_healthy, title_uc):

    print("Making Heatmap")
    subj = ""
    for subj_ in subjset:
        subj = subj_

    log_norm1 = LogNorm(vmax = max_, vmin = min_)
    times_li = list(times)

    fig = plt.figure(figsize = (55, 70))
    spec = gridspec.GridSpec(ncols = 4, nrows = 6, figure = fig,
        width_ratios = [11, 11, 1, 1])
    axes1 = fig.add_subplot(spec[0 : 6, 0])
    axes2 = fig.add_subplot(spec[0 : 6, 1])
    cax1 = fig.add_subplot(spec[1 : 5, 2])
    cax2 = fig.add_subplot(spec[1 : 5, 3])

    cmap1 = sns.color_palette("Blues", as_cmap=True)

    map1 = sns.heatmap(data_healthy, cmap = cmap1, ax = axes1, yticklabels = True,
    xticklabels = True, cbar_ax = cax1, vmax = max_, vmin = min_, norm = log_norm1,
    linewidth = 0.1, cbar_kws = {"shrink" : 0.75, "fraction" : 0.075})

    map2 = sns.heatmap(data_uc, cmap = cmap1, ax = axes2, yticklabels = True,
    xticklabels = True, vmax = max_, vmin = min_, norm = log_norm1,
    linewidth = 0.1, cbar = False)

    cbar1 = map1.collections[0].colorbar
    cax1.set_title("CFUs/g \n (pass filter) \n", fontweight = "bold",
         fontsize = "30")
    cax1.tick_params(labelsize = 35, length = 10, which = "major")
    cax1.tick_params(length = 5, which = "minor")

    axes1 = add_perturbation_label(axes1, subjset.perturbations,subj, times_li,
            alpha=0, textsize = 35)
    for perturbation in subjset.perturbations:
        axes1.axvline(x = times_li.index(perturbation.starts[subj.name]),
        color = "black", linestyle = '-', lw=2.5)
        axes1.axvline(x = times_li.index(perturbation.ends[subj.name]) + 1,
        color = "black", linestyle = '-', lw=2.5)

    axes2 = add_perturbation_label(axes2, subjset.perturbations, subj, times_li,
            alpha=0, textsize = 35)
    for perturbation in subjset.perturbations:
        axes2.axvline(x = times_li.index(perturbation.starts[subj.name]),
        color = "black", linestyle = '-', lw=2.5)
        axes2.axvline(x = times_li.index(perturbation.ends[subj.name]) + 1,
        color = "black", linestyle = '-', lw=2.5)

    uc_mask1 = get_mask1(data_uc, uc_filter)
    healthy_mask1 = get_mask1(data_healthy, healthy_filter)
    cmap2 = sns.color_palette("Greys", as_cmap=True)

    map3 = sns.heatmap(data_healthy, cmap = cmap2, ax = axes1, yticklabels = True,
    xticklabels = True, cbar_ax = cax2, vmax = max_, vmin = min_, norm = log_norm1,
    mask = healthy_mask1, linewidth = 0.1, cbar_kws = {"shrink" : 0.75,
    "fraction" : 0.075})

    map4 = sns.heatmap(data_uc, cmap = cmap2, ax = axes2, yticklabels = True,
    xticklabels = True,  vmax = max_, vmin = min_, norm = log_norm1,
    mask = uc_mask1, linewidth = 0.1, cbar = False)

    cbar2 = map3.collections[0].colorbar #colorbar for p-values
    cax2.set_title("CFUs/g \n (not pass filter)\n", fontweight = "bold",
    fontsize = "30")
    cax2.tick_params(labelsize = 35, length = 10, which = "major")
    cax2.tick_params(length = 5, which = "minor")

    uc_mask2, uc_format = get_mask2(data_uc)
    healthy_mask2, healthy_format = get_mask2(data_healthy)

    cmap3 = sns.color_palette("seismic", as_cmap=True)

    map1 = sns.heatmap(data_healthy, annot = healthy_format, fmt = "", cmap = cmap3,
    xticklabels = True, cbar = False, vmax = 1, vmin = -1, ax = axes1, yticklabels = True,
    mask = healthy_mask2, linewidth = 0.1, annot_kws = {"fontsize": 20})

    map2 = sns.heatmap(data_uc, annot = uc_format, fmt = "", cmap = cmap3,
    xticklabels = True, cbar = False,vmax = 1, vmin = -1, ax = axes2, yticklabels = True,
    mask = uc_mask2, linewidth = 0.1, annot_kws = {"fontsize": 22.5})

    axes1.set_xlabel("Days", fontweight = "bold", fontsize = 27.5)
    axes1.set_xticklabels(axes1.get_xticklabels(), fontsize = 20)
    axes1.set_yticklabels(axes1.get_yticklabels(), fontsize = 17)

    axes2.set_xlabel("Days", fontweight = "bold", fontsize = 27.5)
    axes2.set_xticklabels(axes2.get_xticklabels(), fontsize = 20)
    axes2.set_yticklabels(axes2.get_yticklabels(), fontsize = 17)

    axes1.set_title(title_healthy, fontsize = 45, fontweight = "bold", loc = "left")
    axes2.set_title(title_uc, fontsize = 45, fontweight = "bold", loc = "left")


    fig.text(0, -0.05, leg_text, fontsize = 40, fontweight = "bold", transform =
    axes1.transAxes)

    plt.savefig(name + ".pdf", bbox_inches = "tight", dpi = 100)
    plt.savefig(name + ".png", bbox_inches = "tight", dpi = 100)


def get_hierarchy(taxo):
    """
       returns the lowest defined hierarchy
       @parameters
       taxo : (pl.Taxa)
    """

    name = ""
    if taxo["species"] != "NA":
        return taxo["genus"].split("_")[0] + " " + taxo["species"]
    elif taxo["genus"] != "NA":
        return "* " +  taxo["genus"]
    elif taxo["family"] != "NA":
        return "** " + taxo["family"]
    elif taxo["order"] != "NA":
        return "*** " + taxo["order"]
    elif taxo["class"] != "NA":
        return "**** " + taxo["class"]
    elif taxo["phylum"] != "NA":
        return "***** " + taxo["phylum"]
    else:
        return "****** " + taxo["kingdom"]

def get_names(df, subjset):
    """
    get the names of OTU in the df

    @return
    -------------------------------------------------------------------------
    (dict) : (str) otu_id -> (str) otu name

    """

    taxas = subjset.taxas
    index_old = df.index
    names_dict = {}
    for otu in index_old:
        taxonomy = taxas[otu]
        hierarchy = get_hierarchy(taxonomy)
        names_dict[otu] = otu.replace("_", " ") + ", " + hierarchy

    return names_dict


def main():

    args = parse_args()

    save_path = "output_figures/"
    subjset_healthy = md2.Study.load(args.healthy_pkl)
    subjset_uc = md2.Study.load(args.uc_pkl)

    times = []
    for subj in subjset_uc:
        times = np.append(times, subj.times)
    times = np.sort(np.unique(times))

    df_healthy, df_uc = get_top_abundance_data(subjset_healthy, subjset_uc)

    names_uc = get_names(df_uc, subjset_uc)
    names_healthy = get_names(df_healthy, subjset_healthy)

    uc_pass_filtering = get_filtered_otu_names(args.uc_filter)
    healthy_pass_filtering = get_filtered_otu_names(args.healthy_filter)

    max_ = get_max(df_uc.to_numpy(), df_healthy.to_numpy())
    min_ = 1e4
    legend = "Taxonomy Key \n* Genus, ** : Family, *** : Order, **** : Class,"\
              " ***** : Phylum, ****** : Kingdom"

    draw_heatmap(df_healthy, df_uc, uc_pass_filtering,
                 healthy_pass_filtering, max_, min_, save_path + "supplemental_figure2",
                 subjset_uc, times, legend, "A", "B")

if __name__ == "__main__":
    main()
