"""
python supplemental_figure_10_and_11.py -file1 "../files/figures/healthy_CV_DATASET_PATHs.tsv"\
      -file2 "../files/figures/uc_coclusters.tsv"\
      -file3 "../files/figures/healthy_clusters.tsv"\
      -file4 "../files/figures/uc_clusters.tsv"\
      -file5 "../../processed_data/gibson_healthy_agg_taxa.pkl"\
      -file6 "../../processed_data/gibson_uc_agg_taxa.pkl"\
      -file7 "../files/figures/healthy_interactions.tsv"\
      -file8 "../files/figures/uc_interactions.tsv"\
      -opt True
plots supplemental figure 6 and 7 i.e. the posterior co-clustering probabilities

"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import figure_helper
import argparse
import mdsine2 as md2
from matplotlib.colors import SymLogNorm

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.cluster import hierarchy
from scipy.spatial import distance as dist
import os 

def parse_args():

    parser = argparse.ArgumentParser(description = "files needed for making"\
    "supplemental figure 6 and 7")
    parser.add_argument("-file1", "--healthy_prob", required = "True",
    help = ".tsv file containing co-clustering probailities for healthy cohort")
    parser.add_argument("-file2", "--uc_prob", required = "True",
    help = ".tsv file containing co-clustering probailities for UC cohort")
    parser.add_argument("-file3", "--healthy_cluster", required = "True",
    help = ".tsv file containing consensus cluster assignment for healthy cohort")
    parser.add_argument("-file4", "--uc_cluster", required = "True",
    help = ".tsv file containing consensus cluster assignment for UC cohort")
    parser.add_argument("-file5", "--healthy_pkl", required = "True",
        help = "pickled md2.base.Study file for healthy subjects")
    parser.add_argument("-file6", "--uc_pkl", required = "True",
        help = "pickled md2.base.Study file for UC subjects")
    parser.add_argument("-file7", "--healthy_interactions", required = "True",
    help = ".tsv file containing interaction matrix for healthy cohort")
    parser.add_argument("-file8", "--uc_interactions", required = "True",
    help = ".tsv file containing interaction matrix for UC cohort")
    parser.add_argument("-opt", "--enable_opt", required = "True",
    help = ".tsv file containing consensus cluster assignment for UC cohort")

    return parser.parse_args()

def plot_interactions_heatmap(data, x_names, y_names, figsize_x, figsize_y,
     fontsize_x, fontsize_y, figname):

    max_ = 1e-9
    min_ = 1e-14
    n = int(np.log10(max_ // min_))
    df = pd.DataFrame(data, index = y_names, columns = x_names)

    fig = plt.figure(figsize = (figsize_x, figsize_y))
    axes = fig.add_subplot(1, 1, 1)

    major_ticks = [1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14 ,0, -1e-9, -1e-10,
    -1e-11, -1e-12, -1e-13, -1e-14]
    heatmap = sns.heatmap(df, xticklabels = 2, yticklabels = True, cmap = "RdBu",
    ax = axes, cbar_kws = {"shrink" : 0.75, "ticks" : major_ticks},
    norm = SymLogNorm(1e-14, vmax = max_, vmin = -max_),
    linewidth = 0.05, vmin = -max_, vmax = max_, linecolor = "black")

    axes.set_xticklabels(axes.get_xticklabels(), fontsize = fontsize_x,
    rotation = 90)
    axes.set_yticklabels(axes.get_yticklabels(), fontsize = fontsize_y,
    rotation = 0)
    axes.tick_params(length = 7.5, width = 1 , left = True, bottom = True,
    color = "black")

    cbar = heatmap.collections[0].colorbar
    cbar = heatmap.collections[0].colorbar #colorbar for p-values
    cbar.ax.tick_params(labelsize = 20, length = 15, width = 1, which = "major")
    cbar.ax.tick_params(length = 5, width = 1, which = "minor")

    cbar.ax.set_title( "Interaction \n Strength \n", fontweight = "bold",
                  fontsize = 25)

    for _, spine in axes.spines.items():
        spine.set_visible(True)

    legend = "Taxonomy Key \n* : Genus, ** : Family, *** : Order, **** : Class,"\
              "***** : Phylum, ****** : Kingdom"
    pos = axes.get_position()
    fig.text(pos.x0, pos.y0 - 0.05, legend, fontsize = 25, fontweight = "bold")
    #fig.text(0, 0.85, code,fontsize = 100, fontweight = "bold")

    loc = "output_figures/"
    if not os.path.exists(loc):
        os.makedirs(loc, exist_ok = True)

    plt.savefig(loc + figname + ".pdf", dpi = 100, bbox_inches = "tight")
    print("done")


def interaction_map(interaction_mat, x_names, y_names, otus_order, index_d,
     name):
    print("Interaction Heatmap")
    matrix_reordered = helper.reorder(interaction_mat, otus_order, index_d)
    plot_interactions_heatmap(matrix_reordered, x_names, y_names,
              24, 27, 17, 14, name + "_interaction_map")

def get_leaves_order(data, otu_li, cluster_d, enable_opt = True):
    """
       get the optimal order for the heatmap; if enable_opt is False, then
       return the OTU order based on consensus cluster dict

       @parameters
        data : (numpy) a square matrix
        otu_li : a list containing the list of OTU ids in serial order

        @returns
        [str] : a list consisting of OTU ids
    """
    order = otu_li
    linkage_ = ""
    leaves = ""
    dist_mat = dist.squareform(1 - data)

    if enable_opt:
        linkage_ = linkage(dist_mat, "average", optimal_ordering = True)
        leaves = dendrogram(linkage_)["leaves"]
        order = [otu_li[i] for i in leaves]
        return order
    else:
        return [otu for id in cluster_d for otu in cluster_d[id]]

def main():

    args = parse_args()
    enable_opt = True
    if args.enable_opt == "False":
        enable_opt = False

    uc_savename = ""
    healthy_savename = ""
    if enable_opt:
        uc_savename = "uc_opt"
        healthy_savename = "healthy_opt"
    else:
        uc_savename = "uc_no_opt"
        healthy_savename = "healthy_no_opt"

    subjset_healthy = md2.Study.load(args.healthy_pkl)
    subjset_uc = md2.Study.load(args.uc_pkl)

    cluster_healthy = helper.parse_cluster(args.healthy_cluster)
    healthy_cocluster_prob = pd.read_csv(args.healthy_prob, sep = "\t",
                        index_col = 0)
    healthy_order_li = list(healthy_cocluster_prob.columns)
    healthy_order_d = {healthy_order_li[i] : i for i in range(len(healthy_order_li))}
    healthy_opt_order = get_leaves_order(healthy_cocluster_prob.to_numpy(),
                        healthy_order_li, cluster_healthy, enable_opt)
    x_healthy, y_healthy = helper.get_axes_names(healthy_opt_order,
        cluster_healthy, subjset_healthy)
    healthy_interaction = pd.read_csv(args.healthy_interactions, sep = "\t",
                        index_col = 0)
    healthy_order_li = list(healthy_interaction.columns)
    healthy_order_d = {healthy_order_li[i] : i for i in range(len(healthy_order_li))}
    print("Making Figure 10")
    interaction_map(healthy_interaction.to_numpy(), x_healthy, y_healthy,
        healthy_opt_order, healthy_order_d, healthy_savename)

    cluster_uc = helper.parse_cluster(args.uc_cluster)
    uc_cocluster_prob = pd.read_csv(args.uc_prob, sep = "\t",
                        index_col = 0)
    uc_order_li = list(uc_cocluster_prob.columns)
    uc_order_d = {uc_order_li[i] : i for i in range(len(uc_order_li))}
    uc_opt_order = get_leaves_order(uc_cocluster_prob.to_numpy(),
                        uc_order_li, cluster_uc, enable_opt)
    x_uc, y_uc = helper.get_axes_names(uc_opt_order,
        cluster_uc, subjset_uc)
    uc_interaction = pd.read_csv(args.uc_interactions, sep = "\t",
                        index_col = 0)
    uc_order_li = list(uc_interaction.columns)
    uc_order_d = {uc_order_li[i] : i for i in range(len(uc_order_li))}
    print("Making Figure 11")
    interaction_map(uc_interaction.to_numpy(), x_uc, y_uc,
        uc_opt_order, uc_order_d, uc_savename)

if __name__ == "__main__":
    main()