#makes a box plot illusrating the MDSINE2 error. The figure is equivalent to
#figure3. The difference being that it shows the results for MDSINE2 only.

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
import pickle as pkl
import pandas as pd
import seaborn as sns
import matplotlib.gridspec as gridspec
from statsmodels.stats import multitest
import matplotlib.lines as mlines
import argparse
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

REL_ORDER = ["MDSINE2"]
ABS_ORDER = ["MDSINE2"]

HEX_REL = sns.color_palette("tab10").as_hex()
HEX_ABS = sns.color_palette("tab10").as_hex()

PAL_REL = {"MDSINE2":HEX_REL[0]}
PAL_ABS = {"MDSINE2":HEX_REL[0]}

TITLE_FONTSIZE = 18
TICK_FONTSIZE = 12
AXES_FONTSIZE = 16

def parse_args():

    parser = argparse.ArgumentParser(description = "files needed for making"\
    "figure 3")
    parser.add_argument("-md2_path", "--mdsine_path", required = True,
        help = "path lo mdsine2 forward sims")
    parser.add_argument("-o_path", "--output_path", required=True,
        help = "path to the folder where the output is saved")

    return parser.parse_args()

def compute_rms_error(pred, truth):
    """computes the root mean square error"""

    error = np.sqrt(np.mean(np.square(pred - truth), axis=1))

    return error

def combine_into_df(data_dict):
    """creates a dataframe contaning data from all subjects and methods"""

    all_data_y = {}
    for key1 in data_dict:
        for key2 in data_dict[key1]:
            if key2 not in all_data_y:
                all_data_y[key2] = []
            all_data_y[key2] += list(data_dict[key1][key2])

    methods= []
    errors = []
    for keys in all_data_y:
        n = len(all_data_y[keys])
        methods += [keys] * n
        errors += all_data_y[keys]

    df = pd.DataFrame(list(zip(methods, errors)), columns=["Method", "Error"])
    return df

def format_box_plot_data(subjs, donor, loc_md2, dict_, type_, limit, use_log):

    all_data = {}
    epsilon=0

    for subj in subjs:
        cv_name = "{0}-cv{1}".format(donor, subj)
        prefix = dict_[cv_name]

        true_abund = np.load(loc_md2 + "{}-cv{}-validate-{}-full-truth"\
        ".npy".format(donor, subj, subj))#[:, 1:]
        #add limit of detection
        true_abund = np.where(true_abund<1e5, 1e5, true_abund)

        pred_abund = np.load(loc_md2  + "{}-cv{}-validate-{}-full"\
        ".npy".format(donor, subj, subj))#[:, :, 1:]
        pred_abund = np.where(pred_abund < 1e5, 1e5, pred_abund)

        times = np.load(loc_md2  + "{}-cv{}-validate-{}-full-times"\
        ".npy".format(donor, subj, subj))#[1:]

        pred_abund_median = np.nanmedian(pred_abund, axis=0)

        if type_ =="abs":
            if use_log:
                pred_abund_median = np.log10(pred_abund_median)
                true_abund = np.log10(true_abund)
                pred_abund = np.log10(pred_abund)

            md2_error = compute_rms_error(pred_abund_median, true_abund)

            pred_errors = {"MDSINE2":  md2_error}
            all_data[prefix] = pred_errors

        elif type_ =="rel":
            rel_true_abund = true_abund / np.sum(true_abund, axis=0,
                keepdims=True)
            rel_pred_abund_median = pred_abund_median / np.nansum(pred_abund_median,
                axis=0, keepdims=True)
            rel_true_abund = np.where(rel_true_abund<1e-6, 1e-6, rel_true_abund)
            rel_pred_abund_median = np.where(rel_pred_abund_median<1e-6, 1e-6,
                rel_pred_abund_median)

            if use_log:
                rel_true_abund = np.log10(rel_true_abund)
                rel_pred_abund_median = np.log10(rel_pred_abund_median)

            md2_error = compute_rms_error(rel_pred_abund_median, rel_true_abund)

            pred_errors = {"MDSINE2": md2_error}
            all_data[prefix] = pred_errors


    combined_df = combine_into_df(all_data)

    return combined_df

def box_plot(data_df, axes, title, use_log, type_, ylab):
    """make a box plot"""

    axes.xaxis.grid(False)
    axes.yaxis.grid(True)

    axes.set_title(title, loc="left", fontweight="bold", fontsize=TITLE_FONTSIZE)
    if not use_log:
        axes.set_yscale("log")
    palette = PAL_REL
    order = REL_ORDER
    if type_ =="abs":
        palette = PAL_ABS
        order = ABS_ORDER

    sns.boxplot(y="Error", x="Method", data=data_df, whis=[2.5, 97.5], width=.75,
        showfliers=False, ax=axes, palette=palette, order=order)
    sns.stripplot(y="Error", x="Method", data=data_df, size=2,
        linewidth=0.5, alpha=0.5, ax=axes, palette=palette, order=order) #, color=".3"


    axes.set_ylabel(ylab, fontsize=AXES_FONTSIZE, fontweight="bold")
    axes.set_xlabel("Model", fontsize=AXES_FONTSIZE, labelpad=3, fontweight="bold")
    axes.set_xticklabels(axes.get_xticklabels(), rotation=0, fontsize=TICK_FONTSIZE)
    axes.tick_params(axis='y', labelsize=TICK_FONTSIZE)


def export_df(order, p_values, p_values_adjusted, name):

    loc = "gibson_inference/figures/output_figures/p_fig3"
    os.makedirs(loc, exist_ok=True)
    order_li = ["{}-{}".format(order[0], order[i]) for i in range(1, len(order))]
    df = pd.DataFrame(np.vstack((p_values, p_values_adjusted)).T)
    df.columns = ["Raw p-value", "BH adjusted p-value"]
    df.index = order_li
    df.to_csv("{}/{}.csv".format(loc, name), sep=",")

def main():

    healthy_subjs = ["2", "3", "4", "5"]
    uc_subjs = ["6", "7", "8", "9", "10"]
    prior = "mixed"
    use_log=True
    rel_lim=1e-6
    abs_lim=1e5
    ep = 6

    print("Making Figure 3")
    fig = plt.figure(figsize=(12, 4.5))
    spec = gridspec.GridSpec(ncols=22, nrows=1, figure=fig)

    ax_he_abs_box = fig.add_subplot(spec[0, 0:4])
    ax_uc_abs_box = fig.add_subplot(spec[0, 6:10])

    ax_he_rel_box = fig.add_subplot(spec[0, 12:16])
    ax_uc_rel_box = fig.add_subplot(spec[0, 18:22])

    args = parse_args()
    mdsine_path = args.mdsine_path

    dict_cv = {"healthy-cv5":"gibson_healthy_predictions-3-",
    "healthy-cv4":"gibson_healthy_predictions-2-",
    "healthy-cv3":"gibson_healthy_predictions-1-",
    "healthy-cv2":"gibson_healthy_predictions-0-",
     "uc-cv6":"gibson_uc_predictions-1-",
    "uc-cv7":"gibson_uc_predictions-2-", "uc-cv8":"gibson_uc_predictions-3-",
    "uc-cv9":"gibson_uc_predictions-4-", "uc-cv10":"gibson_uc_predictions-0-"}

    healthy_abs_box_df = format_box_plot_data(healthy_subjs, "healthy",
       mdsine_path, dict_cv, "abs", abs_lim, True)
    uc_abs_box_df = format_box_plot_data(uc_subjs, "uc", mdsine_path, dict_cv,
         "abs", abs_lim, True)

    healthy_rel_box_df = format_box_plot_data(healthy_subjs, "healthy",
       mdsine_path, dict_cv, "rel", rel_lim, True)
    uc_rel_box_df  = format_box_plot_data(uc_subjs, "uc", mdsine_path,
        dict_cv, "rel", rel_lim, True)

    box_plot(healthy_abs_box_df, ax_he_abs_box, "A. Healthy", use_log, "abs",
        "RMSE (log Abs Abundance)")
    box_plot(uc_abs_box_df, ax_uc_abs_box, "B. UC", use_log, "abs",
        "RMSE (log Abs Abundance)")
    box_plot(healthy_rel_box_df, ax_he_rel_box, "C. Healthy", use_log, "rel",
        "RMSE (log Rel Abundance)")
    box_plot(uc_rel_box_df, ax_uc_rel_box, "D. UC", use_log, "rel",
        "RMSE (log Rel Abundance)")

    os.makedirs(args.output_path, exist_ok=True)
    fig.savefig(args.output_path + "md2_error.pdf",
        dpi=200)

    print("Done Making Figure")

main()
