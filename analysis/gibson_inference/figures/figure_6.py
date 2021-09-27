"""
Eigenvalues + Cycle histograms figure.
"""
import itertools
import os
import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

import argparse
from collections import defaultdict

import pickle
from mdsine2.logger import logger

import pandas as pd
import seaborn as sns


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
    n_samples = 10000
    lengths = [2, 3]
    signs = [
        ['++', '+-', '-+', '--'],
        ['+++', '++-', "+-+", "-++", "+--", "-+-", "--+", "---"],
        [''.join(signs) for signs in itertools.product(['+', '-'], repeat=4)]
    ]

    x = np.array(list(range(len(lengths))))

    df = pd.DataFrame(columns=["Value"],
                      dtype=np.float,
                      index=pd.MultiIndex.from_product(
                          [lengths, range(n_samples), ["Healthy", "UC"]],
                          names=["Length", "Index", "Dataset"]
                      ))

    for k_idx, k in enumerate(lengths):
        df.loc[(k, slice(None), "Healthy")] = np.expand_dims(
            np.sum(np.array([
                [healthy[i][k + 1][sign] for sign in signs[k_idx]]
                for i in range(n_samples)
            ]), axis=1),
            axis=1
        )
        df.loc[(k, slice(None), "UC")] = np.expand_dims(
            np.sum(np.array([
                [uc[i][k + 1][sign] for sign in signs[k_idx]]
                for i in range(n_samples)
            ]), axis=1),
            axis=1
        )

    df = df.reset_index()

    if do_log:
        log_value = np.log(df["Value"].to_numpy())
        log_value[~np.isfinite(log_value)] = 0
        df["Value"] = log_value
        # ax.set_yscale('log')

    sns.boxplot(x="Length", y="Value", data=df, hue="Dataset", ax=ax)
    sns.swarmplot(x="Length", y="Value", data=df, color=".25", hue="Dataset", ax=ax)

    # sns.violinplot(x="Length", y="Value", hue="Dataset", data=df, ax=ax, cut=0, inner="quartile", bw=0.1)

    ax.set_xticks(x)
    ax.set_xticklabels(lengths, fontsize=8)
    # ax.set_title(title, x=-0.05, y=1.0, pad=-14)
    ax.set_xlabel("Cycle Length")
    ax.legend(loc='upper left')


def plot_signed_counts(healthy, uc, ax, title="", do_log=False):
    # Per sign
    signs = ['(+ +)',
             '(- -)',
             '(+ -)',
             '(+ + +)',
             '(- - -)',
             '(+ + -)',
             '(- - +)']

    n_samples = 10000

    df = pd.DataFrame(columns=["Value"],
                      dtype=np.float,
                      index=pd.MultiIndex.from_product(
                          [signs, range(n_samples), ["Healthy", "UC"]],
                          names=["Sign", "Index", "Dataset"]
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
        df.loc[(sign, slice(None), "Healthy")] = np.expand_dims(np.array(arr), axis=1)

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
        df.loc[(sign, slice(None), "UC")] = np.expand_dims(np.array(arr), axis=1)

    df = df.reset_index()
    if do_log:
        log_value = np.log(df["Value"].to_numpy())
        log_value[~np.isfinite(log_value)] = 0
        df["Value"] = log_value
        # ax.set_yscale('log')

    sns.boxplot(x="Sign", y="Value", data=df, hue="Dataset", ax=ax)
    sns.swarmplot(x="Sign", y="Value", data=df, color=".25", hue="Dataset", ax=ax)
    # sns.violinplot(x="Sign", y="Value", hue="Dataset", data=df, ax=ax, cut=0, inner="quartile", bw=0.1)


def parse_csv_cycle_counts(filepath):
    with open(filepath, 'r') as f:
        for line in f:
            tokens = line.split(";")
            cycle = [int(i) for i in tokens[0].strip().split("->")]
            total_count = int(tokens[1].strip())
            bayes = float(tokens[2].strip())
            sampled_signs = tokens[3].split(",") # TODO parse signs and get consensus sign via majority vote.
            for idx, sgn in enumerate(sampled_signs):
                sgn = sgn.strip()
                if len(sgn) > 0:
                    yield idx, cycle, total_count, sgn

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
            print("Loaded pickle.")
    except Exception:
        print("Failed to load pickle. Regenerating counts.")
        # sample idx -> LEN -> sign -> count
        counts = defaultdict(dd1)
        for idx, cycle, total_count, sign in parse_csv_cycle_counts(cycles_path):
            counts[idx][len(cycle)][sign] = counts[idx][len(cycle)][sign] + 1
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
    FORMAT = args.format
    DPI = args.dpi

    # out_path = os.path.join(args.out_dir, "figure6.{}".format(FORMAT))
    # fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 14),
    #                          gridspec_kw={'height_ratios': [1.5, 1, 1], 'hspace': 0.4})

    # ================= Eigenvalues ==================
    out_path = os.path.join(args.out_dir, "figure6-eig.{}".format(FORMAT))
    print("Plotting eigenvalues.")
    # axes[0].axis('off')
    # axes[0]._frameon = False
    # ax1 = axes[0].inset_axes([0, 0, 0.5, 1.0])
    # ax2 = axes[0].inset_axes([0.55, 0, 0.5, 1.0])
    # cbar_ax = axes[0].inset_axes([1.07, 0., 0.01, 1.0])
    fig, [ax1, ax2, cbar_ax] = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))
    generate_eigenvalue_figure(fig, ax1, ax2, cbar_ax, args.eig_path)

    # Put both plots on same scale.
    ax1.get_shared_x_axes().join(ax1, ax2)
    ax1.get_shared_y_axes().join(ax1, ax2)
    ax2.autoscale()

    # ================ RENDERING TO OUTPUT FILE =============
    fig.tight_layout()
    plt.savefig(out_path, format=FORMAT, dpi=DPI)
    logger.info("Eigenvalue figure saved to {}.".format(out_path))

    # ================= OTU-OTU interactions ==================
    out_path = os.path.join(args.out_dir, "figure6-otu.{}".format(FORMAT))
    print("Plotting OTU-OTU interactions.")
    fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2, figsize=(15, 10), gridspec_kw={'width_ratios': [1, 3]})

    healthy_thresholded_otu_cycles = handle_dataset(
        cycles_path=args.healthy_cycles_taxa,
    )
    uc_thresholded_otu_cycles = handle_dataset(
        cycles_path=args.uc_cycles_taxa,
    )

    # axes[1].axis('off')
    # axes[1]._frameon = False
    # ax1 = axes[1].inset_axes([0, 0, 0.48, 1.0])
    # ax2 = axes[1].inset_axes([0.55, 0, 0.5, 1.0])
    plot_unsigned_counts(healthy_thresholded_otu_cycles, uc_thresholded_otu_cycles,
                         ax=ax1,
                         do_log=True)
    ax1.set_ylabel("ASV-ASV Log-Count")

    plot_signed_counts(healthy_thresholded_otu_cycles, uc_thresholded_otu_cycles,
                       ax=ax2,
                       do_log=True)
    ax2.set_ylabel("ASV-ASV Log-Count")

    # ================ RENDERING TO OUTPUT FILE =============
    fig.tight_layout()
    plt.savefig(out_path, format=FORMAT, dpi=DPI)
    logger.info("Eigenvalue figure saved to {}.".format(out_path))

    # ================== Cluster-Cluster ==================
    out_path = os.path.join(args.out_dir, "figure6-cluster.{}".format(FORMAT))
    print("Plotting cluster-cluster interactions.")
    fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2, figsize=(15, 10), gridspec_kw={'width_ratios': [1, 3]})
    healthy_clustered_cycles = handle_dataset(
        cycles_path=args.healthy_cycles_clusters,
    )
    uc_clustered_cycles = handle_dataset(
        cycles_path=args.uc_cycles_clusters,
    )

    # axes[2].axis('off')
    # axes[2]._frameon = False
    # ax1 = axes[2].inset_axes([0, 0, 0.48, 1.0])
    # ax2 = axes[2].inset_axes([0.55, 0, 0.5, 1.0])
    plot_unsigned_counts(healthy_clustered_cycles, uc_clustered_cycles,
                         ax=ax1)
    ax1.set_ylabel("Cluster-Cluster Log-Count")
    plot_signed_counts(healthy_clustered_cycles, uc_clustered_cycles,
                       ax=ax2)
    ax2.set_ylabel("Cluster-Cluster Log-Count")

    # ================ RENDERING TO OUTPUT FILE =============
    fig.tight_layout()
    plt.savefig(out_path, format=FORMAT, dpi=DPI)
    logger.info("Figure saved to {}.".format(out_path))


if __name__ == "__main__":
    main()
