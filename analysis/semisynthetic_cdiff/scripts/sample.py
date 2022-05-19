"""
Python script for generating semisynthetic samples for a given seed + noise level.
Takes as input MDSINE1's BVS sample matrix file
"""
from typing import Tuple, Iterator, List
from pathlib import Path
import argparse

import numpy as np
import h5py
from mdsine2 import *
from mdsine2.names import STRNAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mdsine_result_path', type=str, required=True,
                        help='<Required> The path to the MDSINE BVS output file (V7.3 .mat/.hdf5 format, from MATLAB)')
    parser.add_argument('-n', '--num_subjects', type=int, required=True,
                        help='<Required> ')
    # num_subjects
    # process_var (default 0.1**2 ??)
    # sim_dt (default 0.01 ??)
    # out_dir
    return parser.parse_args()


class MDSINEResultBVS(object):
    def __init__(self, bvs_path: Path):
        self.bvs_path = bvs_path

    def species_names(self) -> List[str]:
        """
        https://stackoverflow.com/questions/28541847/how-convert-this-type-of-data-hdf5-object-reference-to-something-more-readable
        """
        with h5py.File(self.bvs_path, 'r') as f:
            species_names: List[str] = []
            for species_name_ref in f[f.get("species_names_filtered")[0][0]][:][0]:
                species_name = ''.join(chr(i[0]) for i in f[species_name_ref][:])
                species_names.append(species_name)
            return species_names

    def model_samples(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        with h5py.File(self.bvs_path, 'r') as f:
            for hdf_sample_ref in f.get("Theta_samples_select")[0]:
                obj = f[hdf_sample_ref]
                sampled_values = obj[:]
                assert isinstance(sampled_values, np.ndarray)

                growth_rates = sampled_values[:, 0]
                interactions = sampled_values[:, 1:]
                yield growth_rates, interactions

    def interaction_indicators(self) -> np.ndarray:
        with h5py.File(self.bvs_path, 'r') as f:
            return f['Theta_select_indicator'][:]

    def glv_params(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        growth_rates = []
        interactions = []

        indicators = self.interaction_indicators()
        for growth_rate_sample, interaction_sample in self.model_samples():
            if np.sum((indicators > 0) != (interaction_sample > 0)) == 0:
                growth_rates.append(growth_rate_sample)
                interactions.append(interaction_sample)

        print("Found {} samples with matching sparsity pattern.".format(len(growth_rates)))
        growth_rates = np.median(np.stack(growth_rates, axis=0), axis=0)
        interactions = np.median(np.stack(interactions, axis=0), axis=0)
        return growth_rates, interactions, indicators


def make_synthetic(
        name: str,
        taxa: TaxaSet,
        growth_rate_values: np.ndarray,
        interaction_values: np.ndarray,
        interaction_indicators: np.ndarray,
        seed: int
) -> Synthetic:
    syn = Synthetic(name=name, seed=seed)
    syn.taxa = taxa

    clustering = Clustering(clusters=None, G=syn.G, items=syn.taxa, name=STRNAMES.CLUSTERING_OBJ)
    interactions = Interactions(clustering=clustering, use_indicators=True, name=STRNAMES.INTERACTIONS_OBJ, G=syn.G)
    for interaction in interactions:
        # Set interaction values
        target_cid = interaction.target_cid
        source_cid = interaction.source_cid

        tcidx = clustering.cid2cidx[target_cid]
        scidx = clustering.cid2cidx[source_cid]

        interaction.value = interaction_values[tcidx, scidx]
        interaction.indicator = interaction_indicators[tcidx, scidx]

    syn.model.interactions = interaction_values
    syn.model.growth = growth_rate_values
    return syn


def main():
    args = parse_args()
    bvs_result = MDSINEResultBVS(Path(args.bvs_path))
    growth_rates, interactions, interaction_indicators = bvs_result.glv_params()
    seed = args.seed

    taxa_names = bvs_result.species_names()
    taxa = TaxaSet()
    for taxa_name in taxa_names:
        taxa.add_taxon(taxa_name)

    synthetic = make_synthetic('cdiff_mdsine_bvs', taxa, growth_rates, interactions, interaction_indicators, seed=seed)

    # Make subject names
    synthetic.set_subjects([f'subj_{i}' for i in range(args.num_subjects)])
    # where to set subject timepoints?

    # Generate the trajectories.
    synthetic.generate_trajectories(
        dt=args.sim_dt,
        init_dist=variables.Uniform(low=1e5, high=1e7),
        processvar=model.MultiplicativeGlobal(args.process_var)
    )

    noise_levels = {
        'low': 0.01,
        'medium': 0.1,
        'high': 0.2
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    for noise_level_name, noise_level in noise_levels.items():
        # Simulate noise.
        study = synthetic.simulateMeasurementNoise(
            a0=1e-10,
            a1=0.05,
            qpcr_noise_scale=noise_level,
            approx_read_depth=50000,
            name=f'cdiff-semisynth-noise-{noise_level_name}'
        )
        study.save(out_dir / f'subjset_{noise_level_name}.pkl')


if __name__ == "__main__":
    main()
