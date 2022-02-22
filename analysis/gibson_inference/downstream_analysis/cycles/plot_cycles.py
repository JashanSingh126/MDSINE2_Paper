import argparse
import numpy as np
import scipy
import scipy.stats
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import mdsine2 as md2
from mdsine2.names import STRNAMES


def parse_args():
    parser = argparse.ArgumentParser(description="Plot the interaction eigenvalues.")

    # Input specification.
    parser.add_argument('-h', '--healthy_mcmc_path', required=True, type=str,
                        help='<Required> The path to the Healthy cohort MCMC pickle.')
    parser.add_argument('-uc', '--uc_mcmc_path', required=True, type=str,
                        help='<Required> The path to the Dysbiotic (UC) cohort MCMC pickle.')
    parser.add_argument('-p', '--plot_path', required=True, type=str,
                        help='<Required> The target output path of the plot')

    parser.add_argument('--format', required=False, type=str,
                        default='pdf',
                        help='<Optional> The plot output format. Default: `pdf`')

    return parser.parse_args()


def main():
    args = parse_args()

    default_colors = sns.color_palette()
    healthy_color = default_colors[0]
    uc_color = default_colors[1]

    healthy = MdsineOutput("Healthy", args.healthy_mcmc_path)
    uc = MdsineOutput("Healthy", args.uc_mcmc_path)
    assert healthy.num_samples == uc.num_samples

    print("Computing cycles for Healthy dataset.")
    healthy_signed_cycles = compute_signed_statistics(
        healthy.get_clustered_interactions()
    )

    print("Computing cycles for Dysbiotic dataset.")
    uc_signed_cycles = compute_signed_statistics(
        uc.get_clustered_interactions()
    )

    plot(healthy.num_samples,
         healthy_signed_cycles, uc_signed_cycles,
         healthy_color, uc_color,
         args.plot_path, args.format)


class MdsineOutput(object):
    """
    A class to encode the data output by MDSINE.
    """

    def __init__(self, dataset_name, pkl_path):
        self.dataset_name = dataset_name
        self.mcmc = md2.BaseMCMC.load(pkl_path)
        self.taxa = self.mcmc.graph.data.taxa
        self.name_to_taxa = {otu.name: otu for otu in self.taxa}

        self.interactions = None
        self.clustering = None

        self.clusters_by_idx = {
            c_idx: [self.get_taxa(oidx) for oidx in cluster.members]
            for c_idx, cluster in enumerate(self.get_clustering())
        }

    @property
    def num_samples(self) -> int:
        return self.mcmc.n_samples

    def get_cluster_df(self):
        return pd.DataFrame([
            {
                "id": cluster.id,
                "idx": c_idx + 1,
                "otus": ",".join([self.get_taxa(otu_idx).name for otu_idx in cluster.members]),
                "size": len(cluster)
            }
            for c_idx, cluster in enumerate(self.clustering)
        ])

    def get_interactions(self):
        if self.interactions is None:
            self.interactions = self.mcmc.graph[STRNAMES.INTERACTIONS_OBJ].get_trace_from_disk(section='posterior')
        return self.interactions

    def get_taxa(self, idx):
        return self.taxa.index[idx]

    def get_taxa_by_name(self, name: str):
        return self.name_to_taxa[name]

    def get_taxa_str(self, idx):
        tax = self.taxa.index[idx].taxonomy
        family = tax["family"]
        genus = tax["genus"]
        species = tax["species"]

        if genus == "NA":
            return "{}**".format(family)
        elif species == "NA":
            return "{}, {}*".format(family, genus)
        else:
            return "{}, {} {}".format(family, genus, species)

    def get_taxa_str_long(self, idx):
        return "{}\n[{}]".format(self.get_taxa(idx).name, self.get_taxa_str(idx))

    def get_clustering(self):
        if self.clustering is None:
            self.clustering = self.mcmc.graph[STRNAMES.CLUSTERING_OBJ]
            for cidx, cluster in enumerate(self.clustering):
                cluster.idx = cidx
        return self.clustering

    def get_clustered_interactions(self):
        clusters = self.get_clustering()
        otu_interactions = self.get_interactions()
        cluster_interactions = np.zeros(
            shape=(
                otu_interactions.shape[0],
                len(clusters),
                len(clusters)
            ),
            dtype=np.float
        )
        cluster_reps = [
            next(iter(cluster.members)) for cluster in clusters
        ]
        for i in range(cluster_interactions.shape[0]):
            cluster_interactions[i] = otu_interactions[i][np.ix_(cluster_reps, cluster_reps)]
        return cluster_interactions


def compute_signed_statistics(interactions):
    """
    Loop through each gibbs sample. For each gibbs sample, compute the number of cycles, assorted by sign.
    (Does not tell us exactly which cycles appear frequently.)
    """
    N = interactions.shape[0]
    ans = {
        '++': np.zeros(N),
        '--': np.zeros(N),
        '+-': np.zeros(N),
        '+++': np.zeros(N),
        '---': np.zeros(N),
        '++-': np.zeros(N),
        '--+': np.zeros(N)
    }
    for idx, mat in enumerate(interactions):
        signed_cycle_counts = count_signed_cycles(mat)
        for sgn, counts in ans.items():
            counts[idx] = signed_cycle_counts[sgn]
    return ans


def count_signed_cycles(mat):
    '''
    Count length 2 and 3 cycles, given a particular interaction matrix (corresp. to a single gibbs sample).
    '''
    adj = np.copy(mat).T
    plus_adj = np.zeros(shape=adj.shape, dtype=np.int)
    plus_adj[adj > 0] = 1
    minus_adj = np.zeros(shape=adj.shape, dtype=np.int)
    minus_adj[adj < 0] = 1

    return {
        '++': count_cycles_with_sign(['++'], plus_adj, minus_adj) / 2,
        '--': count_cycles_with_sign(['--'], plus_adj, minus_adj) / 2,
        '+-': count_cycles_with_sign(['+-', '-+'], plus_adj, minus_adj) / 2,
        '+++': count_cycles_with_sign(['+++'], plus_adj, minus_adj) / 3,
        '---': count_cycles_with_sign(['---'], plus_adj, minus_adj) / 3,
        '++-': count_cycles_with_sign(['++-', '+-+', '-++'], plus_adj, minus_adj) / 3,
        '--+': count_cycles_with_sign(['--+', '-+-', '+--'], plus_adj, minus_adj) / 3,
    }


def count_cycles_with_sign(signs, plus, minus):
    """
    Multiply adjacency matrices to count the number of cycles. Example: #(+-+) = Trace[(M+) * (M-) * (M+)]
    """
    ans = 0
    for pattern in signs:
        M = np.eye(plus.shape[0])
        for sign in pattern:
            if sign == "+":
                M = M @ plus
            elif sign == "-":
                M = M @ minus
        ans = ans + np.sum(np.diag(M))
    return ans


def plot(n_samples, healthy_signed_cycles, uc_signed_cycles, healthy_color, uc_color,
         plot_path, plot_format):
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))

    signs = ['++', '--', '+-', '+++', '---', '++-', '--+']
    sign_order = {
        "({})".format(" ".join(sgn)): i for i, sgn in enumerate(signs)
    }
    df = pd.DataFrame(columns=["Count"],
                      dtype=np.float,
                      index=pd.MultiIndex.from_product(
                          [signs, range(n_samples), ["Healthy", "UC"]],
                          names=["Sign", "Index", "Dataset"]
                      ))

    for pattern in signs:
        df.loc[(pattern, slice(None), "Healthy")] = healthy_signed_cycles[pattern].reshape(-1, 1)
        df.loc[(pattern, slice(None), "UC")] = uc_signed_cycles[pattern].reshape(-1, 1)

    df = df.reset_index()
    df["Sign"] = df["Sign"].map({
        "++": "(+ +)",
        "--": "(- -)",
        "+-": "(+ -)",
        "+++": "(+ + +)",
        "---": "(- - -)",
        "++-": "(+ + -)",
        "--+": "(- - +)",
    })

    sns.violinplot(x="Sign",
                   y="Count",
                   hue="Dataset", data=df,
                   ax=ax,
                   scale="count",
                   cut=0,
                   #                    inner="quartile",
                   bw=0.5,
                   palette={"Healthy": healthy_color, "UC": uc_color})

    # =========== P-values + Benjamini-Hochberg correction
    df_healthy = df.loc[df["Dataset"] == "Healthy", ["Sign", "Index", "Count"]]
    df_uc = df.loc[df["Dataset"] == "UC", ["Sign", "Index", "Count"]]
    df_merged = df_healthy.merge(
        df_uc,
        left_on=["Sign", "Index"],
        right_on=["Sign", "Index"],
        how="inner",
        suffixes=["Healthy", "UC"]
    )

    # Compute statistic (raw p-values)
    def fn(tbl):
        u = scipy.stats.mannwhitneyu(
            tbl["CountHealthy"], tbl["CountUC"],
            alternative="less"
        )
        return u

    pvalues = df_merged.groupby(
        "Sign"
    ).apply(fn)

    pvalues_df = pd.DataFrame(
        {"pvalue": pvalues.sort_values()}
    )

    # Apply BH correction
    p_adjusted = []
    p_adj_prev = 0.0
    for i, (index, row) in enumerate(pvalues_df.iterrows()):
        p_adj = row["pvalue"].pvalue * pvalues_df.shape[0] / (i + 1)
        p_adj = min(max(p_adj, p_adj_prev), 1)
        p_adjusted.append(p_adj)
        p_adj_prev = p_adj

    pvalues_df["pvalue_adj"] = p_adjusted

    pvalues_df = pvalues_df.reset_index()
    sig_signs = pvalues_df.loc[pvalues_df["pvalue_adj"] <= 1e-3, "Sign"]
    print(pvalues_df)

    for sgn in sig_signs:
        idx = sign_order[sgn]
        pval = pvalues_df.loc[pvalues_df['Sign'] == sgn, 'pvalue_adj'].item()
        if pval <= 1e-4:
            indicator = "****"
        elif pval <= 1e-3:
            indicator = "***"
        else:
            indicator = "ERR"
        y = df.loc[df['Sign'] == sgn, "Count"].max()
        stat_annotate(idx - 0.5, idx + 0.5, y=y, h=0.5, ax=ax, color='black', desc=indicator)
    ax.set_ylabel("Number of cycles per sample")

    plt.savefig(plot_path, format=plot_format)


def stat_annotate(x1, x2, y, h, color, ax, lw=1.0, desc='*'):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=lw, c=color)
    ax.text((x1+x2)*.5, y+h, desc, ha='center', va='bottom', color=color)


if __name__ == "__main__":
    main()
