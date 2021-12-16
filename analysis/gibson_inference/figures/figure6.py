#@title
import mdsine2 as md2
from mdsine2.names import STRNAMES
import numpy as np
import scipy
import scipy.stats
import pandas as pd
from pathlib import Path
from tqdm.notebook import tqdm
import seaborn as sns


# COLORS
_default_colors = sns.color_palette()
_default_healthy_color = _default_colors[0]
_default_uc_color = _default_colors[1]


def stat_annotate(x1, x2, y, h, color, ax, lw=1.0, desc='*'):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=lw, c=color)
    ax.text((x1+x2)*.5, y+h, desc, ha='center', va='bottom', color=color)


class PerturbationSimFigure():
    """Render figures for perturbation simulations. (Figures 6A,6B)"""

    def __init__(self, data_dir: Path, healthy_color=_default_healthy_color, uc_color=_default_uc_color):
        self.healthy_color = healthy_color
        self.uc_color = uc_color

        # DATA DIR
        self.data_dir = data_dir

        # ============ Preprocessing
        print("Loading dataframes from disk.")
        healthy_random_pert_metadata_df = pd.read_hdf(
            data_dir / 'healthy_metadata.h5', key="df", mode="r")
        healthy_random_pert_fwsim_df = pd.read_hdf(
            data_dir / 'healthy_fwsim.h5', key="df", mode="r"
        )

        uc_random_pert_metadata_df = pd.read_hdf(
            data_dir / 'uc_metadata.h5', key="df", mode="r"
        )
        uc_random_pert_fwsim_df = pd.read_hdf(
            data_dir / 'uc_fwsim.h5', key="df", mode="r"
        )

        print("Merging dataframes.")
        healthy_random_pert_merged_df = self.posthoc_random_pert_helper(healthy_random_pert_fwsim_df, healthy_random_pert_metadata_df)
        uc_random_pert_merged_df = self.posthoc_random_pert_helper(uc_random_pert_fwsim_df, uc_random_pert_metadata_df)

        print("Computing difference levels.")
        self.random_pert_concat_df = self.random_pert_diff_figure_df(healthy_random_pert_merged_df, uc_random_pert_merged_df)

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
            metadata.loc[
                :,
                ["OTU", "PerturbedFrac", "Perturbation", "Trial", "IsPerturbed"]
            ],
            left_on=["OTU", "PerturbedFrac", "Perturbation", "Trial"],
            right_on=["OTU", "PerturbedFrac", "Perturbation", "Trial"]
        ).set_index(["OTU", "PerturbedFrac", "Perturbation", "Trial", "SampleIdx"])

        merged_df["SteadyStateDiff"] = np.abs(np.log10(merged_df["SteadyState"] + 1e5) - np.log10(merged_df["SteadyStateBase"] + 1e5))

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

        concat_df["key"] = r'$\alpha$:' + concat_df["PerturbedFrac"].astype(str) + "\nPert:" + concat_df["Perturbation"].astype(str)
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
            p_adj = row["pvalue"].pvalue * pvalues_df.shape[0] / (i+1)
            p_adj = min(max(p_adj, p_adj_prev), 1)
            p_adjusted.append(p_adj)
            p_adj_prev = p_adj

        pvalues_df["pvalue_adj"] = p_adjusted
        pvalues_df = pvalues_df.sort_values("PerturbedFrac").reset_index()
        sig_indices = pvalues_df.index[pvalues_df["pvalue_adj"] <= 1e-3]

        print("Pvalues for deviations")
        display(pvalues_df)

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
            p_adj = row["pvalue"].pvalue * pvalues_df.shape[0] / (i+1)
            p_adj = min(max(p_adj, p_adj_prev), 1)
            p_adjusted.append(p_adj)
            p_adj_prev = p_adj

        pvalues_df["pvalue_adj"] = p_adjusted
        pvalues_df = pvalues_df.sort_values("PerturbedFrac").reset_index()
        sig_indices = pvalues_df.index[pvalues_df["pvalue_adj"] <= 1e-3]

        print("Pvalues for diversity")
        display(pvalues_df)


class EigenvalueFigure():
    def __init__(self, healthy_pickle_path, uc_pickle_path, healthy_color=_default_healthy_color, uc_color=_default_uc_color):
        self.healthy_color = healthy_color
        self.uc_color = uc_color

        print("Computing Healthy dataset eigenvalues.")
        self.healthy_eig_X = self.compute_eigenvalues(healthy_pickle_path)

        print("Computing D dataset eigenvalues.")
        self.uc_eig_X = self.compute_eigenvalues(uc_pickle_path)

    def compute_eigenvalues(self, mcmc_pickle_path, upper_bound: float = 1e20):
        # ================ Data loading
        mcmc = md2.BaseMCMC.load(mcmc_pickle_path)
        si_trace = -np.absolute(mcmc.graph[STRNAMES.SELF_INTERACTION_VALUE].get_trace_from_disk(section='posterior'))
        interactions = mcmc.graph[STRNAMES.INTERACTIONS_OBJ].get_trace_from_disk(section='posterior')

        interactions[np.isnan(interactions)] = 0
        for i in range(len(mcmc.graph.data.taxa)):
            interactions[:,i,i] = si_trace[:,i]

        # ================ Eigenvalue computation
        N = interactions.shape[0]
        M = interactions.shape[1]
        eigs = np.zeros(shape=(N, M), dtype=np.complex)
        for i in tqdm(range(N)):  # range(arr.shape[0]):
            matrix = interactions[i]

            # only for interaction matrices
            matrix = np.nan_to_num(matrix, nan=0.0)

            slice_eigs = np.linalg.eigvals(matrix)

            # Throw out samples where eigenvalues blow up
            if np.sum(np.abs(slice_eigs) > upper_bound) > 0:
                print("Upper bound threshold {th} passed for sample {i}; skipping.".format(
                    th=upper_bound,
                    i=i
                ))
                continue

            eigs[i, :] = slice_eigs

        # Get real positive parts only.
        eigs = eigs.real.flatten()
        return eigs[eigs > 0]

    def plot(self, ax, alpha: float = 0.7):
        ####################################################
        # Eigenvalue histogram.
        ####################################################

        bins = np.linspace(0, 5e-9, 45)
        sns.histplot(self.uc_eig_X, ax=ax, bins=bins, label='Dysbiosis',
                     alpha=alpha, color=self.uc_color)
        sns.histplot(self.healthy_eig_X, ax=ax, bins=bins, label='Healthy',
                     alpha=alpha, color=self.healthy_color)

        ax.set_xlabel('Pos. Real Part of Eigenvalues', labelpad=20)
        ax.set_ylabel('Total count with multiplicity')
        ax.ticklabel_format(style="sci", scilimits=(0,0), useMathText=True, useOffset=False)


class CycleFigure():
    def __init__(self, healthy_pickle_path, uc_pickle_path, healthy_color=_default_healthy_color, uc_color=_default_uc_color):
        self.healthy_color = healthy_color
        self.uc_color = uc_color

        self.healthy_pickle_path = healthy_pickle_path
        self.uc_pickle_path = uc_pickle_path

        healthy_fixed_cluster = MdsineOutput(
            "Healthy",
            self.healthy_pickle_path
        )
        uc_fixed_cluster = MdsineOutput(
            "UC",
            self.uc_pickle_path
        )

        print("Computing cycles for Healthy dataset.")
        self.healthy_signed_cycles = self.compute_signed_statistics(
            healthy_fixed_cluster.get_clustered_interactions()
        )

        print("Computing cycles for Dysbiotic dataset.")
        self.uc_signed_cycles = self.compute_signed_statistics(
            uc_fixed_cluster.get_clustered_interactions()
        )
        self.n_samples = 10000

    def compute_signed_statistics(self, interactions):
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
        for idx, mat in tqdm(enumerate(interactions), total=N):
            signed_cycle_counts = self.count_signed_cycles(mat)
            for sgn, counts in ans.items():
                counts[idx] = signed_cycle_counts[sgn]
        return ans

    def count_signed_cycles(self, mat):
        '''
        Count length 2 and 3 cycles, given a particular interaction matrix (corresp. to a single gibbs sample).
        '''
        adj = np.copy(mat).T
        plus_adj = np.zeros(shape=adj.shape, dtype=np.int)
        plus_adj[adj > 0] = 1
        minus_adj = np.zeros(shape=adj.shape, dtype=np.int)
        minus_adj[adj < 0] = 1

        return {
            '++': self.count_cycles_with_sign(['++'], plus_adj, minus_adj) / 2,
            '--': self.count_cycles_with_sign(['--'], plus_adj, minus_adj) / 2,
            '+-': self.count_cycles_with_sign(['+-', '-+'], plus_adj, minus_adj) / 2,
            '+++': self.count_cycles_with_sign(['+++'], plus_adj, minus_adj) / 3,
            '---': self.count_cycles_with_sign(['---'], plus_adj, minus_adj) / 3,
            '++-': self.count_cycles_with_sign(['++-', '+-+', '-++'], plus_adj, minus_adj) / 3,
            '--+': self.count_cycles_with_sign(['--+', '-+-', '+--'], plus_adj, minus_adj) / 3,
        }

    @staticmethod
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

    def plot(self, ax):
        lengths = [2, 3]
        signs = ['++', '--', '+-', '+++', '---', '++-', '--+']
        sign_order = {
            "({})".format(" ".join(sgn)): i for i, sgn in enumerate(signs)
        }
        df = pd.DataFrame(columns=["Count"],
                          dtype=np.float,
                          index=pd.MultiIndex.from_product(
                              [signs, range(self.n_samples), ["Healthy", "UC"]],
                              names=["Sign", "Index", "Dataset"]
                          ))

        for pattern in signs:
            df.loc[(pattern, slice(None), "Healthy")] = self.healthy_signed_cycles[pattern].reshape(-1, 1)
            df.loc[(pattern, slice(None), "UC")] = self.uc_signed_cycles[pattern].reshape(-1, 1)

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

        medianprops = dict(linestyle='--', linewidth=2.5)

        sns.violinplot(x="Sign",
                       y="Count",
                       hue="Dataset", data=df,
                       ax=ax,
                       scale="count",
                       cut=0,
    #                    inner="quartile",
                       bw=0.5,
                       palette={"Healthy": self.healthy_color, "UC": self.uc_color})

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
            p_adj = row["pvalue"].pvalue * pvalues_df.shape[0] / (i+1)
            p_adj = min(max(p_adj, p_adj_prev), 1)
            p_adjusted.append(p_adj)
            p_adj_prev = p_adj

        pvalues_df["pvalue_adj"] = p_adjusted



        pvalues_df = pvalues_df.reset_index()
        sig_signs = pvalues_df.loc[pvalues_df["pvalue_adj"] <= 1e-3, "Sign"]
        display(pvalues_df)

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
            stat_annotate(idx-0.5, idx+0.5, y=y, h=0.5, ax=ax, color='black', desc=indicator)
        ax.set_ylabel("Number of cycles per sample")


# Preprocessing for Cycles (Figure F)
class MdsineOutput(object):
    '''
    A class to encode the data output by MDSINE.
    '''
    def __init__(self, dataset_name, pkl_path):
        self.dataset_name = dataset_name
        self.mcmc = md2.BaseMCMC.load(pkl_path)
        self.taxa = self.mcmc.graph.data.taxa
        self.name_to_taxa = {otu.name: otu for otu in self.taxa}

        self.interactions = None
        self.clustering = None

        self.clusters_by_idx = {
            (c_idx): [self.get_taxa(oidx) for oidx in cluster.members]
            for c_idx, cluster in enumerate(self.get_clustering())
        }

    @property
    def num_samples(self) -> int:
        return self.mcmc.n_samples

    def get_cluster_df(self):
        return pd.DataFrame([
            {
                "id": cluster.id,
                "idx": c_idx+1,
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
