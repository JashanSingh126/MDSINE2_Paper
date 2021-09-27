"""
python make_plots.py -file1 "seed0_mult_sparse_3/distance.csv" \
       -file2 "seed0_mult_sparse_3/arithmetic_mean_data.csv" \
       -file3 "seed0_mult_sparse_3/arithmetic_mean_null_all.csv"
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def parse_args():

    parser = argparse.ArgumentParser(description = "files needed to make main"\
    "figure 5 (Phylogenetic Neighborhood Analysis)")

    parser.add_argument("-file1", "--thresholds", required = "True",
        help = ".tsv file containing the distance thresholds")
    parser.add_argument("-file2", "--mdsine2_corr", required = "True",
        help = ".tsv Spearman Correlation between UC and Healthy agglomeration"\
        " vectors in the MDSINE2 model")
    parser.add_argument("-file3", "--null_corr", required = "True",
        help = ".tsv Spearman Correlation between UC and Healthy agglomeration"\
        " vectors in the null model")

    return parser.parse_args()


def plot(x, y1, y2, p5, p95, x_lab, y_lab, title, leg1, leg2, savename):
    """
      x, y1, y2 : (np.array, float) flattened arrays of same dimension
      p5, p95 : (np.array, float) the 5th and 95th percentiles
      x_lab, y_lab : (str) the names of x and y axes
      leg1, leg2 : (str) the plot legends
    """

    fig = plt.figure(figsize=(14, 5))
    xticks = 100 - x
    #space for the PNA animation
    axes1 = fig.add_subplot(1, 2, 1)
    axes1.set_title("A", fontweight="bold", fontsize=18, loc="left")
    axes1.axis("off")

    axes = fig.add_subplot(1, 2, 2)
    axes.set_xlabel(x_lab, fontweight = "bold", fontsize=12)
    axes.set_ylabel(y_lab, fontweight = "bold", fontsize=12)
    #axes.set_title("Healthy-UC Correspondence", fontweight = "bold")
    axes.set_title("B", loc="left", fontweight="bold", fontsize=18)
    axes.plot(x, y1, label = leg1, color = "red")
    axes.plot(x, y2, label = leg2, color = "blue")

    axes.set_xticklabels([100 - x1 for x1 in axes.get_xticks()])
    axes.fill_between(x, p5, p95, color = "blue", alpha = 0.2)
    axes.legend(loc=2)
    fig.subplots_adjust(left=0.015, right=0.985, top=.92, bottom=0.12)
    fig.savefig(savename + ".pdf", dpi = 400)
    print("plotted and saved", savename)

def main():

    args = parse_args()
    thresh_pd = pd.read_csv(args.thresholds, index_col = 0)
    mdsine2_pd = pd.read_csv(args.mdsine2_corr, index_col = 0)
    null_pd = pd.read_csv(args.null_corr, index_col = 0)

    thresh_np = thresh_pd.to_numpy().flatten()
    mdsine2_np = mdsine2_pd.to_numpy().flatten()
    null_np = null_pd.to_numpy()

    null_mean = np.mean(null_np, axis = 1)
    percentile_5 = np.percentile(null_np, 2.5, axis = 1)
    percentile_95 = np.percentile(null_np, 97.5, axis = 1)

    output_loc = "gibson_inference/figures/output_figures/"
    os.makedirs(output_loc, exist_ok=True)

    plot(thresh_np * 100, mdsine2_np, null_mean, percentile_5, percentile_95,
    "Percent Identity", "Mean Spearman Correlation", args.thresholds.split("/")[0],
    "Observed", "Null Distribution", output_loc+"figure5")


main()
