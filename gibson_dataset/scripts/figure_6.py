"""
Eigenvalues + Cycle histograms figure.
"""
import os
import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

import argparse
import pandas as pd
from collections import defaultdict

import logging
import mdsine2 as md2

import pickle


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Figure 6 of main paper."
    )

    # Input specification.
    parser.add_argument('-o', '--out_dir', required=True,
                        help='<Required> The output directory.')
    parser.add_argument('--healthy_cycles_taxa',
                        help='<Required> The path to the CSV output of cycle_count_otu.py run on Healthy.')
    parser.add_argument('--uc_cycles_taxa',
                        help='<Required> The path to the CSV output of cycle_count_otu.py run on UC.')
    parser.add_argument('--healthy_cycles_clusters',
                        help='<Required> The path to the CSV output of cycle_count_cluster.py run on Healthy.')
    parser.add_argument('--uc_cycles_clusters',
                        help='<Required> The path to the CSV output of cycle_count_otu.py run on UC.')
    parser.add_argument('--eig_path',
                        help='<Required> The path to the output of compute_eigenvalues.py.')
    parser.add_argument('-f', '--format', required=False, default="pdf",
                        help='The file format to save.')
    parser.add_argument('-dpi', '--dpi', required=False, default=500, type=int,
                        help='The DPI of the figure image rendering.')
    return parser.parse_args()


def plot_unsigned_counts(healthy, uc, ax, title="", do_log=False):
    """ Total (disregarding signs) """
    num_samples = len(healthy)
    lengths = [2, 3, 4]

    x = np.array(list(range(len(lengths))))
    width = 0.35

    data = []
    for k in lengths:
        data.append([
            np.sum([healthy[i][k + 1][sign] for sign in healthy[i][k + 1].keys()])
            for i in range(num_samples)
        ])

    means = np.array([np.mean(d) for d in data])
    medians = np.array([np.median(d) for d in data])
    lower_q = np.array([np.quantile(d, 0.25) for d in data])
    upper_q = np.array([np.quantile(d, 0.75) for d in data])

    ax.bar(
        x - (width / 2),
        means,
        width=width,
        color='b',
        alpha=0.5,
        label='Healthy'
    )
    ax.errorbar(
        x - (width / 2),
        y=medians,
        yerr=np.vstack([medians - lower_q, upper_q - medians]),
        elinewidth=1.0,
        capsize=1.0,
        fmt='.',
        color='b'
    )

    data = []
    for k in lengths:
        data.append([
            np.sum([uc[i][k+1][sign] for sign in uc[i][k+1].keys()])
            for i in range(num_samples)
        ])
    means = np.array([np.mean(d) for d in data])
    medians = np.array([np.median(d) for d in data])
    lower_q = np.array([np.quantile(d, 0.25) for d in data])
    upper_q = np.array([np.quantile(d, 0.75) for d in data])

    ax.set_ylabel("Count")
    ax.bar(
        x + (width / 2),
        means,
        width=width,
        color='r',
        alpha=0.5,
        label='UC'
    )
    ax.errorbar(
        x + (width / 2),
        y=medians,
        yerr=np.vstack([medians - lower_q, upper_q - medians]),
        elinewidth=1.0,
        capsize=1.0,
        fmt='.',
        color='r'
    )

    if do_log:
        ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(lengths, fontsize=8)
    ax.set_title(title, x=-0.05, y=1.0, pad=-14)
    ax.set_xlabel("Cycle Length")
    ax.legend(loc='upper left')
    # outpath = os.path.join(out_dir, filename)
    # plt.savefig(outpath, format='pdf')


def plot_signed_counts(n_samples, healthy, uc, ax, title="", do_log=False):
    # Per sign
    signs = ['(+ +)',
             '(- -)',
             '(+ -)',
             '(+ + +)',
             '(- - -)',
             '(+ + -)',
             '(- - +)']

    df = pd.DataFrame(columns=["Healthy", "UC"],
                      dtype=np.float,
                      index=pd.MultiIndex.from_product(
                          [signs, range(n_samples)]
                      ))

    data_healthy = [
        [healthy[i][3]['++'] for i in range(n_samples)],
        [healthy[i][3]['--'] for i in range(n_samples)],
        [healthy[i][3]['+-'] + healthy[i][3]['-+'] for i in range(n_samples)],
        [healthy[i][4]['+++'] for i in range(n_samples)],
        [healthy[i][4]['---'] for i in range(n_samples)],
        [healthy[i][4]['++-'] + healthy[i][4]['+-+'] + healthy[i][4]['-++'] for i in range(n_samples)],
        [healthy[i][4]['--+'] + healthy[i][4]['-+-'] + healthy[i][4]['+--'] for i in range(n_samples)]
    ]
    for sign, arr in zip(signs, data_healthy):
        df.loc[(sign, slice(None)), "Healthy"] = arr

    data_uc = [
        [uc[i][3]['++'] for i in range(n_samples)],
        [uc[i][3]['--'] for i in range(n_samples)],
        [uc[i][3]['+-'] + uc[i][3]['-+'] for i in range(n_samples)],
        [uc[i][4]['+++'] for i in range(n_samples)],
        [uc[i][4]['---'] for i in range(n_samples)],
        [uc[i][4]['++-'] + uc[i][4]['+-+'] + uc[i][4]['-++'] for i in range(n_samples)],
        [uc[i][4]['--+'] + uc[i][4]['-+-'] + uc[i][4]['+--'] for i in range(n_samples)]
    ]
    for sign, arr in zip(signs, data_uc):
        df.loc[(sign, slice(None)), "UC"] = arr

    width = 0.35
    width_offset = width / 2

    def autolabel(rects, ax, values):
        """
        Attach a text label above each bar displaying its height
        """
        for rect, value in zip(rects, values):
            height = value
            if height < 0.05 * ax.get_ylim()[1]:
                ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * rect.get_height(),
                        "{:.2f}".format(height),
                        ha='center', va='bottom', fontsize='x-small')

    df_means = df.groupby(level=0).mean()
    df_lowers = df.groupby(level=0).quantile(0.25)
    df_medians = df.groupby(level=0).quantile(0.5)
    df_uppers = df.groupby(level=0).quantile(0.75)
    for col, offset, color in [('Healthy', -width_offset, 'b'), ('UC', width_offset, 'r')]:
        means = [df_means.loc[sgn, col].item() for sgn in signs]
        medians = np.array([df_medians.loc[sgn, col].item() for sgn in signs])
        lower_q = np.array([df_lowers.loc[sgn, col].item() for sgn in signs])
        upper_q = np.array([df_uppers.loc[sgn, col].item() for sgn in signs])
        x = np.arange(0, len(means), 1)

        rects = ax.bar(
            x + offset,
            means,
            width=width,
            color=color,
            alpha=0.5,
            label=col
        )
        autolabel(rects, ax, means)
        ax.errorbar(
            x + offset,
            y=medians,
            yerr=np.vstack([medians - lower_q, upper_q - medians]),
            elinewidth=1.0,
            capsize=1.0,
            fmt='.',
            color=color
        )

    if do_log:
        ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([sgn for sgn in signs], fontsize=8, rotation=45, ha="center")
    ax.set_title(title, x=-0.05, y=1.0, pad=-14)
    ax.set_xlabel("Cycle Sign")
    ax.legend(loc='upper left')


def parse_csv_cycle_counts(filepath):
    with open(filepath, 'r') as f:
        for line in f:
            tokens = line.split(";")
            cycle = [int(i) for i in tokens[0].strip().split("->")]
            count = int(tokens[1].strip())
            bayes = float(tokens[2].strip())
            sampled_signs = tokens[3].split(",") # TODO parse signs and get consensus sign via majority vote.
            for idx, sgn in enumerate(sampled_signs):
                sgn = sgn.strip()
                if len(sgn) > 0:
                    yield idx, cycle, count, sgn

def dd2():
    # value for missing len: defaultdict
    return defaultdict(int)


def dd1():
    # value for missing idx: defaultdict
    return defaultdict(dd2)


def handle_dataset(cycles_path):
    cache_file = os.path.splitext(cycles_path)[0] + ".pkl"
    try:
        with open(cache_file, 'rb') as f:
            counts = pickle.load(f)
    except Exception:
        # sample idx -> LEN -> sign -> count
        counts = defaultdict(dd1)
        for idx, cycle, count, sign in parse_csv_cycle_counts(cycles_path):
            counts[idx][len(cycle)][sign] = counts[idx][len(cycle)][sign] + count
        with open(cache_file, 'wb') as f:
            pickle.dump(counts, f, pickle.HIGHEST_PROTOCOL)
    return counts


def generate_eigenvalue_figure(fig, ax1, ax2, cbar_ax, eig_path):
    """
    Trom Travis's script.
    """

    ####################################################
    # Eigenvalue heatmap
    ####################################################
    npzfile = np.load(eig_path)
    healthy_eigs = npzfile['healthy']
    uc_eigs = npzfile['uc']
    hexbins = []

    for d_idx, (eig, ax) in enumerate([
        (healthy_eigs, ax1),
        (uc_eigs, ax2)
    ]):
        X = eig.real.flatten()
        Y = eig.imag.flatten()
        hx = ax.hexbin(X, Y, cmap='inferno', mincnt=1, linewidths=0.2, norm=LogNorm())
        # hx = ax.hexbin(X, Y, cmap='inferno', mincnt=1, linewidths=0.2, norm=LogNorm(vmin=1.0, vmax=10**3)
        hexbins.append(hx)

        ax.ticklabel_format(useOffset=False)

    ax1.set_xlabel('Real')
    ax2.set_xlabel('Real')
    ax1.set_ylabel('Complex')
    ax2.set_yticklabels([])
    fig.colorbar(hexbins[0], cax=cbar_ax)


def main():
    args = parse_args()
    md2.config.LoggingConfig(level=logging.INFO)
    FORMAT = args.format
    DPI = args.dpi

    out_path = os.path.join(args.out_dir, "figure6.{}".format(FORMAT))
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 14),
                             gridspec_kw={'height_ratios': [1.5, 1, 1], 'hspace': 0.4})

    # ================= Eigenvalues ==================
    axes[0].axis('off')
    axes[0]._frameon = False
    ax1 = axes[0].inset_axes([0, 0, 0.5, 1.0])
    ax2 = axes[0].inset_axes([0.55, 0, 0.5, 1.0])
    cbar_ax = axes[0].inset_axes([1.07, 0., 0.01, 1.0])
    generate_eigenvalue_figure(fig, ax1, ax2, cbar_ax, args.eig_path)

    # Put both plots on same scale.
    ax1.get_shared_x_axes().join(ax1, ax2)
    ax1.get_shared_y_axes().join(ax1, ax2)
    ax2.autoscale()

    # # =================== ASV-ASV ====================
    with open(args.healthy_cycles_taxa, 'r') as f:
        line = f.readline()
        tokens = line.split(";")
        sampled_signs = tokens[3].split(",")
        n_samples = len(sampled_signs)

    # healthy_thresholded_otu_cycles = handle_dataset(
    #     cycles_path=args.healthy_cycles_taxa,
    # )
    # uc_thresholded_otu_cycles = handle_dataset(
    #     cycles_path=args.uc_cycles_taxa,
    # )
    #
    # axes[1].axis('off')
    # axes[1]._frameon = False
    # ax1 = axes[1].inset_axes([0, 0, 0.48, 1.0])
    # ax2 = axes[1].inset_axes([0.55, 0, 0.5, 1.0])
    # plot_unsigned_counts(healthy_thresholded_otu_cycles, uc_thresholded_otu_cycles,
    #                      ax=ax1,
    #                      do_log=True)
    # ax1.set_ylabel("ASV-ASV Count")
    #
    # plot_signed_counts(n_samples, healthy_thresholded_otu_cycles, uc_thresholded_otu_cycles,
    #                    ax=ax2,
    #                    do_log=True)
    # ax2.set_ylabel("ASV-ASV Count")
    #
    #
    # # ================== Cluster-Cluster ==================
    # healthy_clustered_cycles = handle_dataset(
    #     cycles_path=args.healthy_cycles_clusters,
    # )
    # uc_clustered_cycles = handle_dataset(
    #     cycles_path=args.uc_cycles_clusters,
    # )
    #
    # axes[2].axis('off')
    # axes[2]._frameon = False
    # ax1 = axes[2].inset_axes([0, 0, 0.48, 1.0])
    # ax2 = axes[2].inset_axes([0.55, 0, 0.5, 1.0])
    # plot_unsigned_counts(healthy_clustered_cycles, uc_clustered_cycles,
    #                      ax=ax1)
    # ax1.set_ylabel("Cluster-Cluster Count")
    # plot_signed_counts(n_samples, healthy_clustered_cycles, uc_clustered_cycles,
    #                    ax=ax2)
    # ax2.set_ylabel("Cluster-Cluster Count")


    # ================ RENDERING TO OUTPUT FILE =============
    fig.tight_layout()
    plt.savefig(out_path, format=FORMAT, dpi=DPI)
    logging.info("Figure saved to {}.".format(out_path))


if __name__ == "__main__":
    main()
