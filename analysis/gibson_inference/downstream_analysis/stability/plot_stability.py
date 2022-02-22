from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import scipy.stats

import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser("Forward simulate by excluding a cluster from the day-20 levels.")
    parser.add_argument('--healthy-dir', '-h', dest='healthy_dir', type=str, required=True,
                        help='<Required> The directory containing the outputs of '
                             'evaluate_stability_simulated.py script for the Healthy dataset.')
    parser.add_argument('--uc-dir', '-h', dest='uc_dir', type=str, required=True,
                        help='<Required> The directory containing the outputs of '
                             'evaluate_stability_simulated.py script for the Dysbiotic (UC) dataset.')

    parser.add_argument('-p', '--plot-path', dest='plot_path', required=True, type=str,
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

    pert_fig = PerturbationSimFigure(
        args.healthy_dir,
        args.uc_dir,
        healthy_color,
        uc_color
    )

    fig, axes = plt.figure(2, 1, figsize=(10, 16))
    pert_fig.plot_deviations(axes[0])
    pert_fig.plot_deviations(axes[1])

    plt.savefig(args.plot_path, format=args.format)


class PerturbationSimFigure(object):
    """Render figures for perturbation simulations. (Figures 6A,6B)"""

    def __init__(self, healthy_dir: Path, uc_dir: Path, healthy_color, uc_color):
        self.healthy_color = healthy_color
        self.uc_color = uc_color

        # ============ Preprocessing
        print("Loading dataframes from disk.")
        healthy_random_pert_metadata_df: pd.DataFrame = pd.read_hdf(healthy_dir / 'metadata.h5', key="df", mode="r")
        healthy_random_pert_fwsim_df: pd.DataFrame = pd.read_hdf(healthy_dir / 'fwsim.h5', key="df", mode="r")

        uc_random_pert_metadata_df: pd.DataFrame = pd.read_hdf(uc_dir / 'metadata.h5', key="df", mode="r")
        uc_random_pert_fwsim_df: pd.DataFrame = pd.read_hdf(uc_dir / 'fwsim.h5', key="df", mode="r")

        print("Merging dataframes.")
        healthy_random_pert_merged_df = self.posthoc_random_pert_helper(healthy_random_pert_fwsim_df,
                                                                        healthy_random_pert_metadata_df)
        uc_random_pert_merged_df = self.posthoc_random_pert_helper(uc_random_pert_fwsim_df, uc_random_pert_metadata_df)

        print("Computing difference levels.")
        self.random_pert_concat_df = self.random_pert_diff_figure_df(healthy_random_pert_merged_df,
                                                                     uc_random_pert_merged_df)

        print("Computing diversities.")
        self.random_pert_diversity_df, self.healthy_random_pert_baseline_diversity, self.uc_random_pert_baseline_diversity = self.precompute_diversities(
            healthy_random_pert_fwsim_df, uc_random_pert_fwsim_df
        )

        print("Finished initialization.")

    @staticmethod
    def posthoc_random_pert_helper(fwsim: pd.DataFrame, metadata: pd.DataFrame):
        merged_df = fwsim.loc[(fwsim["PerturbedFrac"] != 0.0), :].merge(
            fwsim.loc[(fwsim["PerturbedFrac"] == 0.0), ["OTU", "SteadyState", "Died", "SampleIdx"]],
            left_on=["OTU", "SampleIdx"],
            right_on=["OTU", "SampleIdx"],
            how="inner",
            suffixes=["", "Base"]
        ).merge(
            metadata.loc[:, ["OTU", "PerturbedFrac", "Perturbation", "Trial", "IsPerturbed"]],
            left_on=["OTU", "PerturbedFrac", "Perturbation", "Trial"],
            right_on=["OTU", "PerturbedFrac", "Perturbation", "Trial"]
        ).set_index(["OTU", "PerturbedFrac", "Perturbation", "Trial", "SampleIdx"])

        merged_df["SteadyStateDiff"] = np.abs(
            np.log10(merged_df["SteadyState"] + 1e5) - np.log10(merged_df["SteadyStateBase"] + 1e5))

        return merged_df

    @staticmethod
    def random_pert_diff_figure_df(healthy_merged_df: pd.DataFrame, uc_merged_df: pd.DataFrame, perturbation=-2.0):
        #   OTU 	PerturbedFrac 	Perturbation 	Trial 	SampleIdx

        healthy_agg_df = healthy_merged_df.loc[
            (slice(None), slice(None), perturbation, slice(None), slice(None)),
            ["SteadyStateDiff"]
        ].groupby(
            level=[1, 2, 3, 4]  # "PerturbedFrac", "Perturbation", "Trial", "SampleIdx"
        ).mean().groupby(
            level=[0, 1, 2]  # "PerturbedFrac", "Perturbation", "Trial"
        ).mean()

        uc_agg_df = uc_merged_df.loc[
            (slice(None), slice(None), perturbation, slice(None), slice(None)),
            ["SteadyStateDiff"]
        ].groupby(
            level=[1, 2, 3, 4]  # "PerturbedFrac", "Perturbation", "Trial", "SampleIdx"
        ).mean().groupby(
            level=[0, 1, 2]  # "PerturbedFrac", "Perturbation", "Trial"
        ).mean()

        healthy_agg_df["Dataset"] = "Healthy"
        uc_agg_df["Dataset"] = "UC"
        concat_df = pd.concat([
            healthy_agg_df.reset_index(),
            uc_agg_df.reset_index()
        ])

        concat_df["key"] = r'$\alpha$:' + concat_df["PerturbedFrac"].astype(str) + "\nPert:" + concat_df[
            "Perturbation"].astype(str)
        return concat_df

    @staticmethod
    def precompute_diversities(healthy_fwsim_df, uc_fwsim_df, perturbation=-2.0):
        def agg(x):
            p = x["SteadyState"].to_numpy()
            u = np.ones(p.shape[0])
            return scipy.stats.entropy(p) / scipy.stats.entropy(u)

        # ======= Altered
        healthy_diversity = healthy_fwsim_df.loc[
            (healthy_fwsim_df["PerturbedFrac"] != 0.0) & (healthy_fwsim_df["Perturbation"] == perturbation),
            ["Trial", "SampleIdx", "SteadyState", "PerturbedFrac"]
        ].groupby(["PerturbedFrac", "Trial", "SampleIdx"]).apply(
            agg
        ).groupby(level=[0, 1]).mean()
        healthy_diversity = pd.DataFrame({"Diversity": healthy_diversity})
        healthy_diversity["Dataset"] = "Healthy"

        uc_diversity = uc_fwsim_df.loc[
            (uc_fwsim_df["PerturbedFrac"] != 0.0) & (uc_fwsim_df["Perturbation"] == perturbation),
            ["Trial", "SampleIdx", "SteadyState", "PerturbedFrac"]
        ].groupby(["PerturbedFrac", "Trial", "SampleIdx"]).apply(
            agg
        ).groupby(level=[0, 1]).mean()
        uc_diversity = pd.DataFrame({"Diversity": uc_diversity})
        uc_diversity["Dataset"] = "UC"

        diversities = pd.concat([healthy_diversity.reset_index(), uc_diversity.reset_index()]).reset_index()

        # ======= Baselines
        healthy_baseline_diversity = healthy_fwsim_df.loc[
            healthy_fwsim_df["PerturbedFrac"] == 0.0,
            ["SampleIdx", "SteadyState"]
        ].groupby("SampleIdx").apply(
            agg
        ).mean()

        uc_baseline_diversity = uc_fwsim_df.loc[
            uc_fwsim_df["PerturbedFrac"] == 0.0,
            ["SampleIdx", "SteadyState"]
        ].groupby("SampleIdx").apply(
            agg
        ).mean()

        return diversities, healthy_baseline_diversity, uc_baseline_diversity

    def plot_deviations(
            self,
            ax,
            ymin=0.0, ymax=0.35
    ):
        df = self.random_pert_concat_df
        sns.swarmplot(x="PerturbedFrac",
                      y="SteadyStateDiff",
                      hue="Dataset",
                      ax=ax,
                      data=df,
                      palette={"Healthy": self.healthy_color, "UC": self.uc_color},
                      size=3,
                      dodge=True)

        sns.boxplot(
            data=df,
            x="PerturbedFrac", y="SteadyStateDiff",
            hue="Dataset",
            whis=[2.5, 97.5],
            ax=ax,
            showfliers=False,
            palette={"Healthy": self.healthy_color, "UC": self.uc_color},
            boxprops=dict(alpha=.4)
        )

        ax.set_ylim([ymin, ymax])

        # Axis labels
        ax.set_xlabel("Fraction of OTUs perturbed")
        ax.set_ylabel("Difference from Baseline Steady State")

        # =========== P-values + Benjamini-Hochberg correction
        df_healthy = df.loc[df["Dataset"] == "Healthy", ["PerturbedFrac", "SteadyStateDiff"]]
        df_uc = df.loc[df["Dataset"] == "UC", ["PerturbedFrac", "SteadyStateDiff"]]
        df_merged = df_healthy.merge(df_uc, on="PerturbedFrac", how="inner", suffixes=["Healthy", "UC"])

        # Compute statistic (raw p-values)
        def fn(tbl):
            u = scipy.stats.mannwhitneyu(tbl["SteadyStateDiffHealthy"], tbl["SteadyStateDiffUC"], alternative="less")
            return u

        pvalues = df_merged.groupby("PerturbedFrac").apply(fn)
        pvalues_df = pd.DataFrame({"pvalue": pvalues.sort_values()})

        # Apply BH correction
        p_adjusted = []
        p_adj_prev = 0.0
        for i, (index, row) in enumerate(pvalues_df.iterrows()):
            p_adj = row["pvalue"].pvalue * pvalues_df.shape[0] / (i + 1)
            p_adj = min(max(p_adj, p_adj_prev), 1)
            p_adjusted.append(p_adj)
            p_adj_prev = p_adj

        pvalues_df["pvalue_adj"] = p_adjusted
        pvalues_df = pvalues_df.sort_values("PerturbedFrac").reset_index()

        print("Pvalues for deviations:")
        print(pvalues_df)

    def plot_diversity(
            self,
            ax
    ):
        diversity_df = self.random_pert_diversity_df
        healthy_baseline = self.healthy_random_pert_baseline_diversity
        uc_baseline = self.uc_random_pert_baseline_diversity

        sns.swarmplot(x="PerturbedFrac",
                      y="Diversity",
                      hue="Dataset",
                      ax=ax,
                      data=diversity_df,
                      palette={"Healthy": self.healthy_color, "UC": self.uc_color},
                      size=3,
                      dodge=True)

        sns.boxplot(
            x="PerturbedFrac",
            hue="Dataset",
            y="Diversity",
            data=diversity_df,
            ax=ax,
            whis=[2.5, 97.5],
            showfliers=False,
            palette={"Healthy": self.healthy_color, "UC": self.uc_color},
            boxprops=dict(alpha=.4)
        )

        ax.plot([-0.5, 7], [healthy_baseline] * 2, color='blue', linestyle='dashed')
        ax.plot([-0.5, 7], [uc_baseline] * 2, color='orange', linestyle='dashed')

        ax.set_ylabel("Diversity (Normalized Entropy)")
        ax.set_xlabel("Fraction of OTUs perturbed")

        # =========== P-values + Benjamini-Hochberg correction
        df_healthy = diversity_df.loc[diversity_df["Dataset"] == "Healthy", ["PerturbedFrac", "Diversity"]]
        df_uc = diversity_df.loc[diversity_df["Dataset"] == "UC", ["PerturbedFrac", "Diversity"]]
        df_merged = df_healthy.merge(df_uc, on="PerturbedFrac", how="inner", suffixes=["Healthy", "UC"])

        # Compute statistic (raw p-values)
        def fn(tbl):
            u = scipy.stats.mannwhitneyu(tbl["DiversityHealthy"], tbl["DiversityUC"], alternative="greater")
            return u

        pvalues = df_merged.groupby("PerturbedFrac").apply(fn)
        pvalues_df = pd.DataFrame({"pvalue": pvalues.sort_values()})

        # Apply BH correction
        p_adjusted = []
        p_adj_prev = 0.0
        for i, (index, row) in enumerate(pvalues_df.iterrows()):
            p_adj = row["pvalue"].pvalue * pvalues_df.shape[0] / (i + 1)
            p_adj = min(max(p_adj, p_adj_prev), 1)
            p_adjusted.append(p_adj)
            p_adj_prev = p_adj

        pvalues_df["pvalue_adj"] = p_adjusted
        pvalues_df = pvalues_df.sort_values("PerturbedFrac").reset_index()

        print("Pvalues for diversity:")
        print(pvalues_df)
