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

REL_ORDER = ["MDSINE2", "cLV", "LRA", "gLV-RA", "gLV-ridge", "gLV-elastic\n net"]
ABS_ORDER = ["MDSINE2", "gLV-ridge", "gLV-elastic\n net"]

HEX_REL = sns.color_palette("tab10").as_hex()
HEX_ABS = sns.color_palette("tab10").as_hex()

PAL_REL = {"MDSINE2":HEX_REL[0], "cLV":HEX_REL[3], "LRA":HEX_REL[4],
   "gLV-RA":HEX_REL[5], "gLV-ridge":HEX_REL[1], "gLV-elastic\n net":HEX_REL[2]}
PAL_ABS = {"MDSINE2":HEX_REL[0], "gLV-ridge":HEX_REL[1],
    "gLV-elastic\n net":HEX_REL[2]}

TITLE_FONTSIZE = 16
TICK_FONTSIZE = 12
AXES_FONTSIZE = 15


def compute_rms_error(pred, truth):
    """computes the root mean square error"""

    error = np.sqrt(np.mean(np.square(pred - truth), axis=1))

    return error

def parse_args():

    parser = argparse.ArgumentParser(description = "files needed for making"\
    "figure 4")
    parser.add_argument("-md2_path", "--mdsine_path", required = True,
        help = "path lo mdsine2 forward sims")
    parser.add_argument("-clv_path1", "--clv_elas_path", required = True,
        help = "path to compostional model sims trained using elastic net")
    parser.add_argument("-clv_path2", "--clv_ridge_path", required = True,
        help = "path to compostional model sims trained using ridge net")
    parser.add_argument("-glv_path1", "--glv_elas_path", required = True,
        help = "path to aboslute model sims trained using elastic net")
    parser.add_argument("-glv_path2", "--glv_ridge_path", required = True,
        help = "path to absolute model sims trained using ridge")
    parser.add_argument("-o_path", "--output_path", required=True,
        help = "path to the folder where the output is saved")

    return parser.parse_args()

def combine_data_alternative(pred_dict, mean_dict, binned_dict):

    pred_method_dict = {}
    mean_method_dict = {}
    bin_method_dict = {}

    for key1 in pred_dict:
        for key2 in pred_dict[key1]:
            if key2 not in pred_method_dict:
                pred_method_dict[key2] = []
            if key2 not in mean_method_dict:
                mean_method_dict[key2] = []
            if key2 not in bin_method_dict:
                bin_method_dict[key2] = []
            pred_method_dict[key2] += list(pred_dict[key1][key2])
            mean_method_dict[key2] += list(mean_dict[key1])
            bin_method_dict[key2] += list(binned_dict[key1])

    bins = []
    methods = []
    errors = []
    mean_abund = []

    for keys in pred_method_dict:
        n = len(pred_method_dict[key2])
        methods += [keys] * n
        errors += pred_method_dict[keys]
        mean_abund += mean_method_dict[keys]
        bins += bin_method_dict[keys]

    df = pd.DataFrame(list(zip(methods, errors, bins, mean_abund)),
        columns=["Method", "Error", "Floor", "Mean Abundance"])

    return df

def format_alternative_plot_data(subjs, donor, loc_md2, loc_elas, loc_ridge,
    dict_, type_, limit, use_log):

    all_data = {}
    true_data = {}
    binned_data = {}
    abund_all = []
    epsilon=0
    print("Formatting {} {} abundance data for alternative-plot".format(donor,
        type_))
    for subj in subjs:
        cv_name = "{0}-cv{1}".format(donor, subj)
        prefix = dict_[cv_name]
        true_abund = np.load(loc_md2 + "{}-cv{}-validate-{}-full-truth"\
            ".npy".format(donor, subj, subj))
        true_abund = np.where(true_abund<1e5, 1e5, true_abund)

        pred_abund = np.load(loc_md2 + "{}-cv{}-validate-{}-full"\
            ".npy".format(donor, subj, subj))
        pred_abund = np.where(pred_abund<1e5, 1e5, pred_abund)

        times = np.load(loc_md2 + "{}-cv{}-validate-{}-full-times"\
            ".npy".format(donor, subj, subj))#[1:]

        pred_abund_median = np.median(pred_abund, axis=0)

        if type_ =="abs":
            true_abund_mean = np.mean(true_abund, axis=1)
            pred_glv_elastic = pkl.load(open(loc_elas+ prefix +
                "glv-abs", "rb"))[0].T
            pred_glv_ridge = pkl.load(open(loc_ridge + prefix +
                "glv-abs", "rb"))[0].T

            pred_glv_elastic = np.where(pred_glv_elastic<1e5, 1e5,
                pred_glv_elastic)
            pred_glv_ridge = np.where(pred_glv_ridge<1e5, 1e5,
                pred_glv_ridge)

            if use_log:
                pred_abund_median = np.log10(pred_abund_median)
                true_abund = np.log10(true_abund)
                true_abund_mean = np.log10(true_abund_mean)
                pred_glv_elastic = np.log10(pred_glv_elastic)
                pred_glv_ridge = np.log10(pred_glv_ridge)

            glv_elas_error = compute_rms_error(pred_glv_elastic, true_abund)
            glv_ridge_error = compute_rms_error(pred_glv_ridge, true_abund)
            md2_error = compute_rms_error(pred_abund_median, true_abund)
            pred_errors = {"MDSINE2":  md2_error, "gLV-elastic\n net": glv_elas_error,
                "gLV-ridge": glv_ridge_error}

            all_data[prefix] = pred_errors
            true_data[prefix] = true_abund_mean
            binned_data[prefix] = np.floor(true_abund_mean)
            abund_all += list(true_abund_mean)

        elif type_ =="rel":
            rel_true_abund = true_abund / np.sum(true_abund, axis=0,
                keepdims=True)
            rel_true_abund_mean = np.mean(rel_true_abund, axis=1)
            rel_pred_abund_median = pred_abund_median / np.nansum(pred_abund_median,
                axis=0, keepdims=True)

            pred_clv = pkl.load(open(loc_elas + prefix + "clv", "rb"))[0].T#[:, 1:]
            pred_glv = pkl.load(open(loc_elas + prefix + "glv", "rb"))[0].T#[:, 1:]
            pred_lra = pkl.load(open(loc_elas + prefix + "lra", "rb"))[0].T#[:, 1:]
            pred_glv_ra = pkl.load(open(loc_elas + prefix + "glv-ra", "rb"))[0].T#[:, 1:]
            pred_glv_ridge = pkl.load(open(loc_ridge + prefix + "glv", "rb"))[0].T#[:, 1:]

            pred_clv = np.where(pred_clv<1e-6, 1e-6, pred_clv)
            pred_glv = np.where(pred_glv<1e-6, 1e-6, pred_glv)
            pred_lra = np.where(pred_lra<1e-6, 1e-6, pred_lra)
            pred_glv_ra = np.where(pred_glv_ra<1e-6, 1e-6, pred_glv_ra)
            pred_glv_ridge = np.where(pred_glv_ridge<1e-6, 1e-6, pred_glv_ridge)

            rel_true_abund_mean = np.where(rel_true_abund_mean<1e-6, 1e-6,
                rel_true_abund_mean)
            rel_true_abund = np.where(rel_true_abund<1e-6, 1e-6, rel_true_abund)
            rel_pred_abund_median = np.where(rel_pred_abund_median<1e-6, 1e-6,
                rel_pred_abund_median)

            if use_log:
                rel_pred_abund_median = np.log10(rel_pred_abund_median)
                rel_true_abund = np.log10(rel_true_abund)
                rel_true_abund_mean = np.log10(rel_true_abund_mean)

                pred_clv = np.log10(pred_clv+epsilon)
                pred_glv = np.log10(pred_glv+epsilon)
                pred_lra = np.log10(pred_lra+epsilon)
                pred_glv_ra = np.log10(pred_glv_ra+epsilon)
                pred_glv_ridge = np.log10(pred_glv_ridge+epsilon)
            md2_error = compute_rms_error(rel_pred_abund_median, rel_true_abund)
            clv_error = compute_rms_error(pred_clv, rel_true_abund)
            glv_error = compute_rms_error(pred_glv, rel_true_abund)
            lra_error = compute_rms_error(pred_lra, rel_true_abund)
            glv_ra_error = compute_rms_error(pred_glv_ra, rel_true_abund)
            glv_ridge_error = compute_rms_error(pred_glv_ridge, rel_true_abund)

            pred_errors = {"MDSINE2":  md2_error, "cLV": clv_error,
            "gLV-elastic\n net": glv_error, "LRA": lra_error, "gLV-RA": glv_ra_error,
            "gLV-ridge": glv_ridge_error}

            all_data[prefix] = pred_errors
            true_data[prefix] = rel_true_abund_mean
            binned_data[prefix] = np.floor(rel_true_abund_mean)
            abund_all += list(rel_true_abund_mean)

    bar_df = pd.DataFrame(abund_all, columns=["Mean Abundance"])
    combined_df = combine_data_alternative(all_data, true_data, binned_data)

    return combined_df, bar_df

def get_bin_category(bins, df):

    df_np = df["Mean Abundance"].to_numpy()
    bins_np = np.asarray(bins)

    bins_ = np.digitize(df_np, bins_np)
    final_bins = []
    for i in bins_:
        #if i==5:
            #print("yes")
        if i == len(bins):
            final_bins.append(len(bins)-1)
        else:
            final_bins.append(i)

    return np.asarray(final_bins)

def alternative_plot(data_df, title1, title2, use_log, bar_df, type_, axes1,
   axes2, y_lab, donor, add_legend=False):

    def test_stars(p, dtype):
        star = ""
        if p < 0.05:
            star= "x"#"$\\times$"
        else:
            star= "o"#"$\\circ$"
        #print(p, star)
        return star

    quantile_10 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    quantile_25 = [0, 0.25, 0.5, 0.75, 1.00]
    quantile_20 = [0, 0.2, 0.4, 0.6, 0.8, 1.00]
    quantile_1 = [0, 1.00]

    xtick_labels = []

    bins = bar_df["Mean Abundance"].quantile(quantile_10).to_list()
    #print("bins:", bins)
    bin_category = get_bin_category(bins, data_df)
    data_df["Bins"] = bin_category
    order, p_values = signed_rank_test_alt(data_df, donor, type_)

    xtick_labels = ["[{:.1f},\n {:.1f}]".format(bins[i-1], bins[i])
        for i in range(1, len(bins))]
    if type_ =="rel":
        xtick_labels = ["[{:.1f},\n {:.1f}]".format(bins[i-1], bins[i])
            for i in range(1, len(bins))]
    palette = PAL_REL
    order = REL_ORDER
    if type_ =="abs":
        palette = PAL_ABS
        order = ABS_ORDER
    sns.boxplot(y="Error", x="Bins", hue="Method", data=data_df, whis=[2.5, 97.5],
        showfliers=False, ax=axes2, palette=palette, hue_order=order)#, palette=my_pal
    sns.stripplot(y="Error", x="Bins", hue="Method", data=data_df, size=2,
         alpha=0.5, ax=axes2, dodge=True, palette=palette,
        linewidth=0.5, hue_order=order) #color=".8",

    y_data = data_df.groupby(["Bins", "Method"])["Error"].max().values
    star = [test_stars(p, type_) for p in p_values]
    y_data_round = ["{:.1f}".format(p) for p in y_data]

    handles, labels = axes2.get_legend_handles_labels()
    l = ""
    histplot = sns.histplot(data=bar_df.to_numpy(), ax=axes1, bins=bins,
        color="lime", cbar=False, legend=False, stat="density")

    i_star = 0
    i_box = 0

    for tick in range(len(axes2.get_xticklabels())):
        if type_ == "abs":
            y = max(y_data[i_box+1:i_box+3]) + 0.1
            axes2.text(tick+0.145, y, "  ".join(star[i_star:i_star+2]),
                horizontalalignment="center", color="black", fontsize=10)
        else:
            y = max(y_data[i_box+1:i_box+6])+0.1
            if y > 5.5:
                y = y-0.5

            axes2.text(tick+0.065, y, "".join(star[i_star:i_star+5]),
                horizontalalignment="center", color="black", fontsize=10)

        if type_ =="abs":
            i_star += 2
            i_box += 3
        elif type_=="rel":
            i_star += 5
            i_box += 6

    if type_ =="rel":
        axes1.set_xlabel("log (Mean Rel Abundance)", fontsize=AXES_FONTSIZE,
           labelpad=3, fontweight="bold")
        axes2.set_xlabel("log (Mean Rel Abundance)", fontsize=AXES_FONTSIZE,
            fontweight="bold")
        axes1.set_ylim(0, 0.65)
        axes2.set_ylim([0, 5])

    else:
        #print("abs setting x label")
        axes1.set_xlabel("log (Mean Abs Abundance)", fontsize=AXES_FONTSIZE
            , labelpad=3, fontweight="bold")
        axes2.set_xlabel("log (Mean Abs Abundance)", fontsize=AXES_FONTSIZE,
            fontweight="bold")
        axes1.set_ylim(0, 0.6)
        axes2.set_ylim([0, 6.2])


    axes1.set_title(title1, loc="left", fontweight="bold", fontsize=TITLE_FONTSIZE)
    axes1.set_ylabel("Frequency", fontsize=AXES_FONTSIZE,
        fontweight="bold")
    axes2.set_ylabel(y_lab, fontsize=AXES_FONTSIZE, fontweight="bold")

    axes1.tick_params(axis='y', labelsize=TICK_FONTSIZE)
    axes1.tick_params(axis='x', labelsize=TICK_FONTSIZE)
    axes2.tick_params(axis='y', labelsize=TICK_FONTSIZE)

    axes2.yaxis.grid(True)
    axes2.set_title(title2, loc="left", fontweight="bold", fontsize=TITLE_FONTSIZE)
    axes2.set_xticklabels(xtick_labels, fontsize=TICK_FONTSIZE)

    if not use_log:
        axes2.set_yscale("log")

    if add_legend:
        n = 3
        if type_=="rel":
            n=6
        lgd2=axes2.legend(handles[0:n], labels[0:n], bbox_to_anchor=(1.01, 1),
            loc=2, borderaxespad=0., fontsize=TICK_FONTSIZE, title_fontsize=
            AXES_FONTSIZE, title="$\\bf{Model}$")
        axes2.add_artist(lgd2)

        handles = []
        l = mlines.Line2D([],[], color="black", linestyle='none',
            marker='x', label='p < 0.05', markerfacecolor='none')
        handles.append(l)
        l = mlines.Line2D([],[], color="black",linestyle='none',
            marker ='o', label='p > 0.05', markerfacecolor='none')
        handles.append(l)

        lgnd3 = axes2.legend(handles = handles, title='$\\bf{P-values}$',
            loc='lower left', borderaxespad=0., bbox_to_anchor=(1.01, 0),
            title_fontsize=TICK_FONTSIZE, fontsize = TICK_FONTSIZE)
        axes2.add_artist(lgnd3)
    else:
        axes2.get_legend().remove()

def signed_rank_test_alt(data_df, donor, type_):

    bins_dict = {}
    otu_order = {}
    for row in data_df.to_numpy():
        method = row[0]
        bin_id = row[-1]
        mean_abundance = row[3]
        error = row[1]
        if bin_id not in bins_dict:
            bins_dict[bin_id] = {}
        if bin_id not in otu_order:
            otu_order[bin_id] = {}
        if method not in bins_dict[bin_id]:
            bins_dict[bin_id][method] = []
        if method not in otu_order[bin_id]:
            otu_order[bin_id][method] = []

        bins_dict[bin_id][method].append(error)
        otu_order[bin_id][method].append(mean_abundance)

    ref_key = "MDSINE2"
    p_values = []
    order  = []
    for b in sorted(bins_dict.keys()):
        ref_data = bins_dict[b][ref_key]
        #print("bin:", b)
        for m in bins_dict[b]:
            if m != ref_key:
                other_data = bins_dict[b][m]
                order.append("{} and {}".format(ref_key, m))
                s, p = stats.wilcoxon(ref_data, other_data, alternative='less')
                p_values.append(p)
    test = multitest.multipletests(p_values, alpha=0.05, method="fdr_bh")

    return order, test[1]

def main():

    healthy_subjs = ["2", "3", "4", "5"]
    uc_subjs = ["6", "7", "8", "9", "10"]
    prior = "mixed"
    use_log=True
    rel_lim=1e-6
    abs_lim=1e5
    ep = 6

    fig = plt.figure(figsize=(20, 14))
    spec = gridspec.GridSpec(ncols=40, nrows=26, figure=fig)

    ax_he_abs_hist = fig.add_subplot(spec[0:3, 0:19])
    ax_uc_abs_hist = fig.add_subplot(spec[0:3, 21:40])

    ax_he_abs_bin = fig.add_subplot(spec[5:12, 0:19])
    ax_uc_abs_bin = fig.add_subplot(spec[5:12, 21:40])

    ax_he_rel_hist = fig.add_subplot(spec[14:17, 0:19])
    ax_uc_rel_hist = fig.add_subplot(spec[14:17, 21:40])

    ax_he_rel_bin = fig.add_subplot(spec[19:26, 0:19])
    ax_uc_rel_bin = fig.add_subplot(spec[19:26, 21:40])

    args = parse_args()
    mdsine_path = args.mdsine_path
    clv_elas_path = args.clv_elas_path
    clv_ridge_path = args.clv_ridge_path
    glv_elas_path = args.glv_elas_path
    glv_ridge_path = args.glv_ridge_path

    dict_cv = {"healthy-cv5":"gibson_healthy_predictions-3-",
    "healthy-cv4":"gibson_healthy_predictions-2-",
    "healthy-cv3":"gibson_healthy_predictions-1-",
    "healthy-cv2":"gibson_healthy_predictions-0-",
     "uc-cv6":"gibson_uc_predictions-1-",
    "uc-cv7":"gibson_uc_predictions-2-", "uc-cv8":"gibson_uc_predictions-3-",
    "uc-cv9":"gibson_uc_predictions-4-", "uc-cv10":"gibson_uc_predictions-0-"}

    healthy_abs_alt_df, healthy_abs_abund_df = format_alternative_plot_data(
        healthy_subjs, "healthy", mdsine_path, glv_elas_path, glv_ridge_path,
        dict_cv, "abs", 1e5, use_log)
    uc_abs_alt_df, uc_abs_abund_df = format_alternative_plot_data(
        uc_subjs, "uc", mdsine_path, glv_elas_path, glv_ridge_path,
        dict_cv, "abs", 1e5, use_log)

    healthy_rel_alt_df, healthy_rel_abund_df = format_alternative_plot_data(
        healthy_subjs, "healthy", mdsine_path, clv_elas_path, clv_ridge_path,
        dict_cv, "rel", 1e5, use_log)
    uc_rel_alt_df, uc_rel_abund_df = format_alternative_plot_data(
        uc_subjs, "uc", mdsine_path, clv_elas_path, clv_ridge_path,
        dict_cv, "rel", 1e5, use_log)

    alternative_plot(healthy_abs_alt_df, "A", "C", use_log, healthy_abs_abund_df,
       "abs", ax_he_abs_hist, ax_he_abs_bin, "RMSE (log Abs Abundance)",
       "healthy")
    alternative_plot(uc_abs_alt_df, "B","D", use_log, uc_abs_abund_df,
        "abs", ax_uc_abs_hist, ax_uc_abs_bin, "RMSE (log Abs Abundance)",
        "uc", add_legend=True)

    alternative_plot(healthy_rel_alt_df, "E", "G", use_log, healthy_rel_abund_df,
       "rel", ax_he_rel_hist, ax_he_rel_bin, "RMSE (log Rel Abundance)",
        "healthy")
    alternative_plot(uc_rel_alt_df, "F", "H", use_log, uc_rel_abund_df,
        "rel", ax_uc_rel_hist, ax_uc_rel_bin, "RMSE (log Rel Abundance)",
        "uc", add_legend=True)

    os.makedirs(args.output_path, exist_ok=True)
    fig.savefig(args.output_path+"supplemental_figure4.pdf", bbox_inches="tight",
        dpi=800)

main()
