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

REL_ORDER = ["MDSINE2", "cLV", "LRA", "gLV-RA", "gLV-ridge", "gLV-elastic\n net"]
ABS_ORDER = ["MDSINE2", "gLV-ridge", "gLV-elastic\n net"]

PAL_REL = {"MDSINE2":"red", "cLV":"orange", "LRA":"brown",
   "gLV-RA":"yellow", "gLV-ridge":"blue", "gLV-elastic\n net":"green"}
PAL_ABS = {"MDSINE2":"red", "gLV-ridge":"blue",
    "gLV-elastic\n net":"green"}

TITLE_FONTSIZE = 18
TICK_FONTSIZE = 10.5
AXES_FONTSIZE = 14

def parse_args():

    parser = argparse.ArgumentParser(description = "files needed for making"\
    "figure 3")
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

def format_box_plot_data(subjs, donor, loc_md2, loc_elas, loc_ridge, dict_,
    type_, limit, use_log):

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
            pred_glv_elastic = pkl.load(open(loc_elas + prefix + "glv-abs",
                "rb"))[0].T
            pred_glv_ridge = pkl.load(open(loc_ridge+ prefix + "glv-abs",
                "rb"))[0].T
            pred_glv_elastic = np.where(pred_glv_elastic < 1e5, 1e5,
                pred_glv_elastic)
            pred_glv_ridge = np.where(pred_glv_ridge < 1e5, 1e5,
                pred_glv_ridge)

            if use_log:
                pred_abund_median = np.log10(pred_abund_median)
                true_abund = np.log10(true_abund)
                pred_abund = np.log10(pred_abund)
                pred_glv_elastic = np.log10(pred_glv_elastic)
                pred_glv_ridge = np.log10(pred_glv_ridge)

            md2_error = compute_rms_error(pred_abund_median, true_abund)
            glv_elas_error = compute_rms_error(pred_glv_elastic, true_abund)
            glv_ridge_error = compute_rms_error(pred_glv_ridge, true_abund)

            pred_errors = {"MDSINE2":  md2_error, "gLV-elastic\n net": glv_elas_error,
                "gLV-ridge": glv_ridge_error}
            all_data[prefix] = pred_errors

        elif type_ =="rel":
            rel_true_abund = true_abund / np.sum(true_abund, axis=0,
                keepdims=True)
            rel_pred_abund_median = pred_abund_median / np.nansum(pred_abund_median,
                axis=0, keepdims=True)
            rel_true_abund = np.where(rel_true_abund<1e-6, 1e-6, rel_true_abund)
            rel_pred_abund_median = np.where(rel_pred_abund_median<1e-6, 1e-6,
                rel_pred_abund_median)

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

            if use_log:
                rel_true_abund = np.log10(rel_true_abund)
                rel_pred_abund_median = np.log10(rel_pred_abund_median)

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

            pred_errors = {"MDSINE2": md2_error, "cLV": clv_error,
                "gLV-elastic\n net": glv_error, "LRA" : lra_error, "gLV-RA" : glv_ra_error,
                "gLV-ridge": glv_ridge_error}
            all_data[prefix] = pred_errors


    combined_df = combine_into_df(all_data)

    test_results = signed_rank_test_box(all_data, "MDSINE2", donor, type_)
    return combined_df, test_results

def box_plot(data_df, axes, title, use_log, type_, ylab, pvalues):
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


    axes.set_ylabel(ylab, fontsize=AXES_FONTSIZE)
    axes.set_xlabel("Model", fontsize=AXES_FONTSIZE, labelpad=3)
    axes.set_xticklabels(axes.get_xticklabels(), rotation=0, fontsize=TICK_FONTSIZE)
    axes.tick_params(axis='y', labelsize=TICK_FONTSIZE)

    if type_=="abs":
        add_annotation(axes, data_df, "Method", "Error",
        [("MDSINE2", "gLV-ridge"), ("MDSINE2", "gLV-elastic\n net")],pvalues,
        ABS_ORDER)
        axes.set_ylim([0, 6.25])
    else:
        add_annotation(axes, data_df, "Method", "Error",
        [("MDSINE2","cLV"),("MDSINE2","LRA"), ("MDSINE2","gLV-RA"),
            ("MDSINE2", "gLV-ridge"), ("MDSINE2", "gLV-elastic\n net")],pvalues,
        REL_ORDER)
        axes.set_ylim([0, 5.5])

def add_annotation(axes, data_df, x_var, y_var, box_pairs, p_values, order):

    def star_p(p):
        if p >=0.05:
            return "ns"
        elif 0.01 <= p < 0.05:
            return "*"
        elif 0.001 <= p < 0.01:
            return "**"
        elif 0.0001 <= p < 0.001:
            return "***"
        else:
            return "****"

    order_index = {order[i]:i for i in range(len(order))}
    y_prev = 0
    for i in range(len(box_pairs)):
        pair = box_pairs[i]
        y1_data = data_df.loc[data_df[x_var]==pair[0]]
        y1 = np.max(y1_data[y_var].to_numpy())
        y2_data = data_df.loc[data_df[x_var]==pair[1]]
        y2 = np.max(y2_data[y_var].to_numpy())
        y = max(y1, y2)+0.05
        if y_prev ==0:
            y_prev = y
        if y < y_prev:
            y = y_prev
        x1 = order_index[pair[0]]
        x2 = order_index[pair[1]]
        h = 0.1
        axes.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c="k")
        star = star_p(p_values[i])
        axes.text((x1+x2)/2, y + h, star, ha = "center", va="bottom", color="k")
        y_prev = y+ 3*h

def signed_rank_test_box(data_dict, ref_key, donor, type_):

    #print("running wilcoxon signed-rank test for {}".format(donor))
    combined_data = {}
    for key1 in data_dict:
        for key2 in data_dict[key1]:
            if key2 not in combined_data:
                combined_data[key2] = []
            #print(key1, key2)
            combined_data[key2] += list(data_dict[key1][key2])

    ref_data = combined_data[ref_key]
    #print(ref_data)
    p_vals = []
    order = []
    for key in combined_data:
        if key != ref_key:
            order.append("{} and {}".format(ref_key, key))
            s, p = stats.wilcoxon(ref_data, combined_data[key], alternative='less')
            #print(combined_data[key])
            p_vals.append(p)

    test = multitest.multipletests(p_vals, alpha=0.05, method="fdr_bh")

    return list(test[1])

def main():

    healthy_subjs = ["2", "3", "4", "5"]
    uc_subjs = ["6", "7", "8", "9", "10"]
    prior = "mixed"
    use_log=True
    rel_lim=1e-6
    abs_lim=1e5
    ep = 6

    fig = plt.figure(figsize=(21, 4.5))
    spec = gridspec.GridSpec(ncols=33, nrows=1, figure=fig)

    ax_he_abs_box = fig.add_subplot(spec[0, 0:5])
    ax_uc_abs_box = fig.add_subplot(spec[0, 6:11])

    ax_he_rel_box = fig.add_subplot(spec[0, 12:22])
    ax_uc_rel_box = fig.add_subplot(spec[0, 23:33])

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

    healthy_abs_box_df, test_healthy_abs = format_box_plot_data(healthy_subjs, "healthy",
       mdsine_path, glv_elas_path, glv_ridge_path, dict_cv, "abs", abs_lim, True)
    uc_abs_box_df, test_uc_abs = format_box_plot_data(uc_subjs, "uc", mdsine_path,
        glv_elas_path, glv_ridge_path, dict_cv, "abs", abs_lim, True)

    healthy_rel_box_df, test_healthy_rel = format_box_plot_data(healthy_subjs, "healthy",
       mdsine_path, clv_elas_path, clv_ridge_path, dict_cv, "rel", rel_lim, True)
    uc_rel_box_df, test_uc_rel  = format_box_plot_data(uc_subjs, "uc", mdsine_path,
        clv_elas_path, clv_ridge_path, dict_cv, "rel", rel_lim, True)


    box_plot(healthy_abs_box_df, ax_he_abs_box, "A", use_log, "abs",
        "RMSE ($\log_{10}$ Abs Abundance)", test_healthy_abs)
    box_plot(uc_abs_box_df, ax_uc_abs_box, "B", use_log, "abs",
        "RMSE ($\log_{10}$ Abs Abundance)", test_uc_abs)
    box_plot(healthy_rel_box_df, ax_he_rel_box, "C", use_log, "rel",
        "RMSE ($\log_{10}$ Rel Abundance)", test_healthy_rel)
    box_plot(uc_rel_box_df, ax_uc_rel_box, "D", use_log, "rel",
        "RMSE ($\log_{10}$ Rel Abundance)", test_uc_rel)

    os.makedirs(args.output_path, exist_ok=True)
    fig.savefig(args.output_path+"figure3.pdf", bbox_inches="tight",
        dpi=800)

main()
