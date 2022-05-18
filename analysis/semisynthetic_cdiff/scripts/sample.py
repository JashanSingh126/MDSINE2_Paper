"""
Python script for generating semisynthetic samples for a given seed + noise level.
"""
from typing import Tuple

import numpy as np
from mdsine2 import *
from mdsine2.names import STRNAMES


def parse_mdsine_bvs_samples() -> Tuple[np.ndarray, np.ndarray]:



def make_synthetic(
        name: str,
        taxa: TaxaSet,
        growth_rate_values: np.ndarray,
        interaction_values: np.ndarray
) -> Synthetic:
    syn = Synthetic(name=name)
    syn.taxa = taxa
    if set_times:
        syn.times = chain.graph.data.subjects.times('union')
    else:
        syn.times = None

    # Set the clustering
    # ------------------
    clustering = Clustering(clusters=None, G=syn.G, items=syn.taxa, name=STRNAMES.CLUSTERING_OBJ)

    # Set the interactions
    # --------------------
    # self_interactions = summary(chain.graph[STRNAMES.SELF_INTERACTION_VALUE])['mean']
    # A = pl.summary(chain.graph[STRNAMES.INTERACTIONS_OBJ], set_nan_to_0=True)['mean']
    # A_cluster = condense_fixed_clustering_interaction_matrix(A, clustering=chain.graph[STRNAMES.CLUSTERING_OBJ])

    # bf = generate_interation_bayes_factors_posthoc(mcmc=chain)
    # bf_cluster = condense_fixed_clustering_interaction_matrix(bf, clustering=chain.graph[STRNAMES.CLUSTERING_OBJ])

    interactions = Interactions(clustering=clustering, use_indicators=True, name=STRNAMES.INTERACTIONS_OBJ, G=syn.G)
    for interaction in interactions:
        # Set interaction values
        target_cid = interaction.target_cid
        source_cid = interaction.source_cid

        tcidx = clustering.cid2cidx[target_cid]
        scidx = clustering.cid2cidx[source_cid]

        if bf_cluster[tcidx, scidx] >= min_bayes_factor:
            interaction.value = A_cluster[tcidx, scidx]
            interaction.indicator = True
        else:
            interaction.value = 0
            interaction.indicator = False

    syn.model.interactions = interaction_values
    syn.model.growth = growth_rate_values


def main():
    synthetic = make_synthetic(name, taxa, growth_rates, interactions, seed=seed)

    # make subject names
    synthetic.set_subjects(['subj-{}'.format(i+1) for i in range(4)])
    # where to set subject timepoints?

    # Generate the trajectories.
    synthetic.generate_trajectories(
        dt=0.01,
        init_dist=variables.Uniform(low=1e5, high=1e7),
        processvar=model.MultiplicativeGlobal(0.1**2)
    )

    # Simulate noise.
    study = synthetic.simulateMeasurementNoise(
        a0=1e-10,
        a1=0.05,
        qpcr_noise_scale=0.25,
        approx_read_depth=50000,
        name='semi-synth'
    )


if __name__ == "__main__":
    main()
