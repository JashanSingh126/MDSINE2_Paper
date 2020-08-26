#This program makes plots of interests
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib.colors as colors
from matplotlib.colors import LogNorm
import os
import pickle

def get_max(data):

    max_ = max(data.max())
    min_ = np.abs(min(data.min()))
    bound = max_
    if min_ > max_:
        bound = min_
    if bound > 5:
        bound = 5

    return bound

def produce_plot1(df, xlab, ylab, title, f_name):
    """plots heatmap of enriched pathway/modules/cazymes

       @Parameters
       df : (dataframe) data to plotted
       xlab : (str) x-axis label
       ylab : (str) y-axis label
       title : (str) title of the plot
       f_name : (str) name for the saved plot
    """

    #fig = plt.figure(figsize = (15, 15))
    cmap = sns.color_palette("Blues_r", n_colors = 4)

    cm = sns.heatmap(data = np.log10(df / 5),xticklabels = True,
    yticklabels = True, vmax = -1, vmin = -5, cmap = cmap,
    linewidths = 0.01, cbar_kws = {"ticks" : [-4.5, -3.5, -2.5, -1.5],
    "label" : "adjusted p-values"})
    cbar = cm.collections[0].colorbar #colorbar for p-values
    cbar.set_ticklabels(["p < 0.0005", "p < 0.005", "p<0.05", "p > 0.05"])


    plt.savefig(f_name + ".jpg", dpi = 1200, bbox_inches = "tight")

def produce_plot2(df, perturb_data_file, abund_data_file, level, xlab, ylab,
                  title, type, n_cluster, f_name, class_count):

    """ plots the heatmap of p-values along with pertubation and composition
        @Parameters
        df : (dataframe) data to be plotted
        perturb_data_file : (pkl) pertubation data
        abund_data_file : (pkl, dictionary) composition based on phylum, order
        xlab : (str) x-axis label
        ylab : (str) y-axis label
        title : (str) title of the plot
        type : (type) type of enrichment analysis
        n_cluster : (int) number of clusters
        f_name: name of saved file

    """
    #complete the missing data
    columns = df.columns
    shape = df.shape
    missing =[1e0] * shape[0]
    for i in range(1, n_cluster + 1):
        if "Cluster " + str(i) not in columns:
            #print("Missing", "Cluster" , i)
            df["Cluster " + str(i)] = missing
    sorted_cols = ["Cluster " + str(i) for i in range(1, n_cluster + 1)]
    new_df = df[sorted_cols]

    taxo_df = pd.read_pickle(abund_data_file)[level]
    taxo_df_total = sum(taxo_df.sum())
    rel_abundance = taxo_df / taxo_df_total #computed the abundance
    #rel_abundance.replace(0.000, np.nan)
    rel_max = max(rel_abundance.max())
    rel_min = min(rel_abundance.min())

    perturb_df = pd.read_pickle(perturb_data_file)
    perturb_max = get_max(perturb_df)


    fig = plt.figure(figsize = (28, 18))#, constrained_layout = True)
    spec = gridspec.GridSpec(ncols = 1 , nrows=10, figure = fig)
    ax1 = fig.add_subplot(spec[0:4, 0])
    ax2 = fig.add_subplot(spec[4:6, 0])
    ax3 = fig.add_subplot(spec[6:10, 0])

    kwargs = {'norm': LogNorm(vmin=rel_min, vmax=rel_max)}

    cmap1 = sns.color_palette("Blues_r", n_colors = 4)
    cmap2 = sns.cubehelix_palette(n_colors=100, as_cmap=True, start = 2, rot=0,
                                  dark=0, light=0.5)
    map1 = sns.heatmap(np.log10(new_df / 5), ax = ax1, cbar = True, cmap = cmap1,
    linewidths = 0.5, yticklabels = True, vmax = -1, vmin = -5,
    cbar_kws = {"ticks" : [-4.5, -3.5, -2.5, -1.5]})
    map2 = sns.heatmap(perturb_df, linewidths = 0.5, ax = ax2, cmap = "RdBu",
    vmax = perturb_max, vmin = -perturb_max)
    map3 = sns.heatmap(rel_abundance, linewidths = 0.5, ax = ax3, cmap = "Greys", vmax = rel_max,
    vmin = 1e-8, cbar_kws = {"ticks" : [1e-1,1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]}, **kwargs)

    cbar = map1.collections[0].colorbar #colorbar for p-values
    cbar.set_ticklabels(["p < 0.0005", "p < 0.005", "p<0.05", "p > 0.05"])
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label(label = "adjusted p-values", size = 15)

    ax1.set_xlabel("")
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation = 0, fontsize = 15)
    ax1.set_ylabel(ylab, fontweight = "bold", rotation = 90, fontsize = 15)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation = 90, fontsize = 15)
    ax1.tick_params(length=0)
    #ax1.yaxis.set_label_coords(-0.5,1.00)
    loc = ax1.yaxis.get_ticklabels()[0]

    ax2.set_yticklabels(ax2.get_yticklabels(), rotation = 0, fontsize = 15)
    ax2.set_ylabel("Perturbation", fontweight = "bold", rotation = 90, fontsize = 15)
    #ax2.yaxis.set_label_coords(-0.5,1.00)
    ax2.tick_params(length=0)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation = 90, fontsize = 15)
    #ax2.set_xticklabels([])
    ax2.set_xlabel("")

    labs = []
    #full_label = list(ax1.get_xticklabels())
    for i in range(len(ax2.get_xticklabels())):
        new_tick =  str(i + 1 )+ "\np:" + str(class_count[i+1]["gp"]) + \
        "\nn:" + str(class_count[i + 1]["gn"]) + "\nna:" + str(class_count[i + 1]["nan"])
        #print(new_tick)
        labs.append(new_tick)

    ax3.set_xlabel("Cluster ID", fontweight = "bold")
    #print(list(ax1.get_xticklabels()))
    ax3.set_ylabel(level, rotation = 90, fontweight = "bold", fontsize = 15)
    #ax3.yaxis.set_label_coords(-0.5,1.00)
    ax3.set_yticklabels(ax3.get_yticklabels(), rotation = 0, fontsize = 15)
    ax3.set_xticklabels(labs, rotation = 0, fontsize = 15)
    ax3.tick_params(length=0)

    cbar3 = map3.collections[0].colorbar
    cbar3.set_ticklabels(["1e-1", "1e-2", "1e-3", "1e-4", "1e-5", "1e-6", "1e-7",0])
    cbar3.ax.tick_params(labelsize=15)
    cbar3.set_label(label = "Relative Abundance", size = 15)


    cbar2 = map2.collections[0].colorbar
    cbar2.ax.tick_params(labelsize=15)
    cbar2.set_label(label = "perturbation values", size = 15)

    fig.text(0.5, -0.02, "Fig : " + title, ha = "center", fontweight='bold')
    plt.tight_layout()
    #plt.show()
    plt.savefig(f_name + ".jpg", dpi = 1200, bbox_inches = "tight")
