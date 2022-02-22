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

def get_deseq_info(loc, donor):

    hfd_names = set(pd.read_csv("{}/{}1.csv".format(loc, donor), index_col=0).index)
    vanc_names = set(pd.read_csv("{}/{}2.csv".format(loc, donor), index_col=0).index)
    gent_names = set(pd.read_csv("{}/{}3.csv".format(loc, donor), index_col=0).index)

    names_union = hfd_names.union(vanc_names.union(gent_names))
    #print(names_union)
    #print(len(names_union))
    cleaned_names = []
    for name in names_union:

        if "unknown" in name:
            if name !="unknown unknown":
                cleaned_names.append(name)
        else:
            cleaned_names.append(name)
    return set(cleaned_names)

def select_rows(names, df):

    values = []
    for name in names:
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

def make_df(loc, names, donor):

    hfd_df = pd.read_csv("{}/{}1.csv".format(loc, donor), index_col=0)
    vanc_df = pd.read_csv("{}/{}2.csv".format(loc, donor), index_col=0)
    gent_df = pd.read_csv("{}/{}3.csv".format(loc, donor), index_col=0)

    dict_df = {"High Fat Diet": [], "Vancomycin":[], "Gentamicin":[]}

    dict_df["High Fat Diet"] = select_rows(names, hfd_df)
    dict_df["Vancomycin"] = select_rows(names, vanc_df)
    dict_df["Gentamicin"] = select_rows(names, gent_df)

    combined_df = pd.DataFrame.from_dict(dict_df)
    combined_df.index = names

    return combined_df


def make_plot(df1, df2, abundance, taxonomy, name, loc):

    #3.5, 11
    fig = plt.figure(figsize=(3.5, 11))
    gs = fig.add_gridspec(28, 6)
    axes1 = ""
    axes2 = ""

    if taxonomy == "order":
        if abundance == "high":
            axes1 = fig.add_subplot(gs[0:18,  0:3])
            axes2 = fig.add_subplot(gs[0:18, 3:6])
        elif abundance == "low":
            axes1 = fig.add_subplot(gs[19:28, 0:3])
            axes2 = fig.add_subplot(gs[19:28, 3:6])

    if taxonomy == "phylum":
        gs = fig.add_gridspec(28, 7)
        if abundance == "high":
            axes1 = fig.add_subplot(gs[0:18,  0:3])
            axes2 = fig.add_subplot(gs[0:18, 4:7])
        elif abundance == "low":
            axes1 = fig.add_subplot(gs[19:28, 0:3])
            axes2 = fig.add_subplot(gs[19:28, 4:7])


    make_heatmap(df1, axes1, False, "Healthy", False)
    make_heatmap(df2, axes2, False, "UC", False)

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

    #if title is not None:
    #    axes.set_title(title, fontweight="bold", fontsize=16)

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

    uc_names = get_deseq_info(deseq_loc, "uc")
    healthy_names = get_deseq_info(deseq_loc, "healthy")
    names_union = uc_names.union(healthy_names)

    names_not_abundant = sorted(list(set(names_union) - set(new_labels)))

    healthy_df_abundant = make_df(deseq_loc, new_labels, "healthy")
    uc_df_abundant = make_df(deseq_loc, new_labels, "uc")

    healthy_df_non_abundant = make_df(deseq_loc, names_not_abundant, "healthy")
    uc_df_non_abundant = make_df(deseq_loc, names_not_abundant, "uc")



    if args.abundance == "high":
        make_plot(healthy_df_abundant, uc_df_abundant, args.abundance,
            args.taxonomy, args.output_name, args.output_loc)
    elif args.abundance == "low":
        make_plot(healthy_df_non_abundant, uc_df_non_abundant, args.abundance,
            args.taxonomy, args.output_name, args.output_loc)
