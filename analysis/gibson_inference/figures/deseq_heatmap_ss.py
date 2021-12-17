#makes the heatmap showing the results of deseq analysis at steady state.
#heatmap in supplementary figure 2

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import os
from matplotlib import rcParams
from matplotlib import font_manager

rcParams['pdf.fonttype'] = 42

font_dirs = ['gibson_inference/figures/arial_fonts']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
#print(font_files)

for font_file in font_files:
    #print(font_file)
    ff = font_file.split("/")[-1]
    if "._" not in ff:
        font_manager.fontManager.addfont(font_file)

# change font
rcParams['font.family'] = 'Arial'

def parse_args():

    parser = argparse.ArgumentParser(description = "files needed for making"\
    "main figure 2")
    parser.add_argument("-loc", "--deseq_loc", required = "True",
        help = "location of the folder containing the results of deseq")
    parser.add_argument("-abund", "--abundance", required='True',
        help="High/Low abundance")
    parser.add_argument("-txt", "--txt_file", required = "True",
        help = "name of the text file")
    parser.add_argument("-taxo", "--taxonomy", required = "True",
        help = "name of the taxonomy")
    parser.add_argument("-o", "--output_name", required = "True",
        help = "Name of the output file")
    parser.add_argument("-o_loc", "--output_loc", required="True",
        help = "location of the output")


    return parser.parse_args()

def get_deseq_info_phylum(loc, donor):

    df = pd.read_csv("{}/{}.csv".format(loc, donor), index_col=0)
    raw_names = df.index
    np_df = df.to_numpy()

    cleaned_names = []
    names_dict = {}
    i = 0

    for name in raw_names:
        p_adj = np_df[i, -1]
        if not np.isnan(p_adj):
            cleaned_names.append(name)
        names_dict[cleaned_names[-1]] = name
        i += 1

    return set(cleaned_names), names_dict

def get_deseq_info_order(loc, donor):

    df = pd.read_csv("{}/{}.csv".format(loc, donor), index_col=0)
    raw_names = df.index
    np_df = df.to_numpy()

    cleaned_names = []
    names_dict = {}
    i = 0

    for name in raw_names:
        name_split = name.split()
        p_adj = np_df[i, -1]
        order_family = " ".join(name_split[2:])
        if not np.isnan(p_adj) and p_adj < 0.05:
            if "unknown" in order_family:
                if order_family !="unknown unknown":
                    cleaned_names.append(order_family.replace("unknown", "unknown"))
                else:
                    cleaned_names.append("{},phylum".format(name_split[0]) + " " +
                        "Phylum")
            else:
                cleaned_names.append(order_family)
            names_dict[cleaned_names[-1]] = name
        i += 1

    return set(cleaned_names), names_dict

def select_rows(names, names_dict, df):

    values = []
    for short_name in names:
        name = ""
        if short_name in names_dict:
            name = names_dict[short_name]

        if name in df.index:
            row = df.loc[name]
            p = row["padj"]
            log_change = row["log2FoldChange"]
            #print(name, p)
            if p <=0.05:
                if log_change < 0:
                    values.append(-1)
                else:
                    values.append(1)
            else:
                values.append(np.nan)
        else:
            #print("not present ", name)
            values.append(np.nan)

    return values

def make_df(loc, names, names_dict, donor):

    df = pd.read_csv("{}/{}.csv".format(loc, donor), index_col=0)
    dict_df = {}
    dict_df["p_ss"] = select_rows(names, names_dict, df)

    combined_df = pd.DataFrame.from_dict(dict_df)
    combined_df.index = names
    if combined_df.empty:
        dict_df["p_ss"] = [np.nan]
        combined_df = pd.DataFrame.from_dict(dict_df)

    return combined_df

def make_plot(df, abundance, taxonomy, name, loc):

    #3.5, 11
    fig = "" #plt.figure(figsize=(3.5, 11))
    gs = "" #fig.add_gridspec(23, 6)
    axes1 = ""

    if taxonomy == "order":
        fig = plt.figure(figsize=(3.5, 11))
        gs = fig.add_gridspec(23, 6)
        if abundance == "high":
            axes1 = fig.add_subplot(gs[0:18,  0:3])
        elif abundance == "low":
            axes1 = fig.add_subplot(gs[19:23, 0:3])

    if taxonomy == "phylum":
        fig = plt.figure()
        gs = fig.add_gridspec(5, 1)
        if abundance == "high":
            axes1 = fig.add_subplot(gs[0:4,  0])
        elif abundance == "low":
            axes1 = fig.add_subplot(gs[4:5, 0])


    make_heatmap(df, axes1, False, "Healthy", False)
    os.makedirs(loc, exist_ok=True)
    fig.savefig("{}/{}.pdf".format(loc, name), bbox_inches="tight")
    plt.close()

def make_heatmap(df, axes, label_x=False, title=None, label_y=False):

    colors = {"red":-1, "blue":1}
    l_colors = sorted(colors, key=colors.get)
    #cmap = mpl.colors.ListedColormap(l_colors)
    cmap = "RdBu"
    axes.set_aspect("equal")
    if label_x:
        axes = sns.heatmap(df, cmap=cmap, cbar=False, yticklabels=label_y,
            linecolor="black", linewidths=1, ax=axes, vmin=-2, vmax=2,
            xticklabels=label_x)
        axes.set_xticklabels(axes.get_xticklabels(), rotation=75, fontsize=12)
    else:
        axes = sns.heatmap(df, cmap=cmap, cbar=False, yticklabels=label_y,
            linecolor="black", linewidths=1, ax=axes, xticklabels=False, vmin=-2,
            vmax=2)

    data = df.to_numpy()
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            n = data[y, x]
            text = ""
            if not np.isnan(n):
                if n > 0:
                    axes.text(x + 0.5, y+0.5, "$+$", horizontalalignment='center',
                        verticalalignment='center', fontsize=12)
                else:
                    axes.text(x + 0.5, y+0.5, "$-$", horizontalalignment='center',
                        verticalalignment='center', fontsize=12)
    for _, spine in axes.spines.items():
         spine.set_visible(True)

if __name__ =="__main__":

    args = parse_args()
    deseq_loc = args.deseq_loc
    file = open("{}/{}.txt".format(deseq_loc, args.txt_file))
    labels = file.read().split(", ")
    new_labels = []
    for lab in labels:
        new_lab = lab
        if args.taxonomy == "phylum":
            new_lab = lab.split()[1]
        if "NA" in lab:
            new_lab = lab.replace("NA", "unknown")
        new_labels.append(new_lab)
    names = ""
    names_dict = ""
    if args.taxonomy == "order":
        names, names_dict = get_deseq_info_order(deseq_loc, "deseq_ss")
    elif args.taxonomy == "phylum":
        names, names_dict = get_deseq_info_phylum(deseq_loc, "deseq_ss")

    names_not_abundant = sorted(list(set(names) - set(new_labels)))
    df_abundant = make_df(deseq_loc, new_labels, names_dict, "deseq_ss")

    df_non_abundant = make_df(deseq_loc, names_not_abundant, names_dict,
        "deseq_ss")


    if args.abundance == "high":
        print("Making Heatmap for species with abundance > 0.5%")
        make_plot(df_abundant, args.abundance,
            args.taxonomy, args.output_name, args.output_loc)
    elif args.abundance == "low":
        print("Making Heatmap for species with abundance < 0.5%")
        make_plot(df_non_abundant, args.abundance,
            args.taxonomy, args.output_name, args.output_loc)
    print("Done making heatmaps")
