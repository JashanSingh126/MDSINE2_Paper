"""
Forward simulate by perturbing a random collection of taxa.
"""
import argparse
from pathlib import Path
from typing import List, Dict, Iterator, Tuple

import numpy as np
from random import seed, randint
import pandas as pd

import mdsine2 as md2
from mdsine2.names import STRNAMES
from tqdm import tqdm


class Seed(object):
    def __init__(self, init: int = 0, min_value: int = 0, max_value: int = 1000000):
        self.value = init
        self.min_value = min_value
        self.max_value = max_value
        seed(self.value)

    def next_value(self):
        r = randint(self.min_value, self.max_value)
        return r


def parse_args():
    parser = argparse.ArgumentParser("Forward simulate by excluding a cluster from the day-20 levels.")
    parser.add_argument('--input-mcmc', '-i', type=str, dest='input_mcmc', required=True,
                        help='<Required> Location of input (either folder of the numpy arrays or ' \
                             'MDSINE2.BaseMCMC chain). MUST BE THE FIXED CLUSTER MCMC RUN.')

    parser.add_argument('--study', '-s', type=str, dest='study', required=True,
                        help='<Required> Path to the Study object to use for initial conditions')
    parser.add_argument('--out-dir', '-o', type=str, dest='out_dir', required=True,
                        help='<Required> The path to which to save the calculated DataFrame of '
                             'simulated steady states.')
    parser.add_argument('--perturbation', '-p', type=float, dest='pert_strength', required=True,
                        help='<Required> The strength of the perturbation. (Positive boosts abundances, '
                             'negative keeps abundances low.)')
    parser.add_argument('--pert-start-day', type=int, dest='pert_start_day', required=True,
                        help='<Required> The day at which the perturbation starts '
                             '(change is applied at the start of the day)')
    parser.add_argument('--pert-end-day', type=int, dest='pert_end_day', required=True,
                        help='<Required> The day at which the perturbation ends '
                             '(change is applied at the start of the day)')

    # ================ Optional params
    parser.add_argument('--num-trials', '-n', type=int, dest='num_trials', required=False,
                        default=30,
                        help='The number of trials to run for each setting. Default: 30')
    parser.add_argument('--seed', '-s', type=int, dest='seed', required=False,
                        default=31415,
                        help='The master seed to use (generates all other seeds).')
    parser.add_argument('--gibbs-subsample', '-g', type=int, dest='gibbs_subsample', required=False,
                        default=100,
                        help='The frequency at which to subsample from the provided MCMC samples. '
                             'Default: 100 (uses one out of every 100 samples)')

    # Other Simulation params
    parser.add_argument('--n-days', dest='n_days', type=int, required=False,
                        help='Total umber of days to simulate for', default=180)
    parser.add_argument('--simulation-dt', type=float, dest='simulation_dt', required=False,
                        help='Timesteps we go in during forward simulation', default=0.01)
    parser.add_argument('--limit-of-detection', dest='limit_of_detection', required=False,
                        help='If any of the taxa have a 0 abundance at the start, then we ' \
                             'set it to this value.', default=1e5, type=float)
    parser.add_argument('--sim-max', dest='sim_max', type=float, required=False,
                        help='Maximum value', default=1e20)

    return parser.parse_args()


def main():
    args = parse_args()

    mcmc = md2.BaseMCMC.load(args.input_mcmc)
    study = md2.Study.load(args.study)
    master_seed = Seed(args.seed)
    gibbs_indices = list(range(0, mcmc.n_samples, args.gibbs_subsample))

    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]  # Fraction of otus to perturb
    trials = list(range(args.num_trials))

    initial_conditions = generate_initial_condition(study, limit_of_detection=args.limit_of_detection)

    fwsim_df_entries = []
    metadata_df_entries = []
    for alpha in alphas:
        for trial in tqdm(trials, desc=f"Alpha={alpha}", total=len(trials)):
            for fwsim_entries, perturbed_otus in forward_simulation_results(
                    mcmc,
                    study,
                    alpha,
                    args.pert_strength,
                    pert_start_day=args.pert_start_day,
                    pert_end_day=args.pert_end_day,
                    initial_conditions=initial_conditions,
                    dt=args.simulation_dt,
                    sim_max=args.sim_max,
                    n_days=args.n_days,
                    gibbs_indices=gibbs_indices,
                    master_seed=master_seed
            ):
                for fwsim_entry in fwsim_entries:
                    fwsim_entry['PerturbedFrac'] = alpha
                    fwsim_entry['Perturbation'] = args.pert_strength
                    fwsim_entry['Trial'] = trial
                    fwsim_df_entries.append(fwsim_entry)

                perturbed_otus = set(perturbed_otus)

                for oidx, otu in enumerate(mcmc.graph.data.taxa):
                    metadata_df_entries.append({
                        'PerturbedFrac': alpha,
                        'Perturbation': args.pert_strength,
                        'Trial': trial,
                        'OTU': otu.name,
                        'IsPerturbed': oidx in perturbed_otus
                    })

    out_dir = Path(args.out_dir)

    fwsim_df = pd.DataFrame(fwsim_df_entries)
    del fwsim_df_entries
    fwsim_df.to_hdf(str(out_dir / "fwsim.h5"), key='df', mode='w')

    metadata_df = pd.DataFrame(metadata_df_entries)
    del metadata_df_entries
    metadata_df.to_hdf(str(out_dir / "metadata.h5"), key='df', mode='w')


def generate_initial_condition(study, limit_of_detection: float):
    print("Generating initial conditions from Day 20 measurements.")
    M = study.matrix(dtype='abs', agg='mean', times='intersection', qpcr_unnormalize=True)
    initial_conditions = M[:, 19]
    initial_conditions[initial_conditions < limit_of_detection] = limit_of_detection
    return initial_conditions


def forward_simulation_results(
        mcmc: md2.BaseMCMC,
        study: md2.Study,
        alpha: float,
        pert: float,
        pert_start_day,
        pert_end_day,
        initial_conditions,
        dt,
        sim_max,
        n_days,
        gibbs_indices: List[int],
        master_seed: Seed
) -> Iterator[Dict]:
    """
    Compute the steady state for the specified simulation, for each taxa.
    """
    for gibbs_idx, forward_sim, perturbed_otus in perturbed_forward_sims(
            mcmc,
            study,
            frac_otus_to_perturb=alpha,
            pert_strength=pert,
            pert_start_day=pert_start_day,
            pert_end_day=pert_end_day,
            initial_conditions=initial_conditions,
            dt=dt,
            sim_max=sim_max,
            n_days=n_days,
            gibbs_indices=gibbs_indices,
            master_seed=master_seed
    ):
        stable_levels = forward_sim[:, -50:].mean(axis=1)  # Last 50 timepoints

        fwsim_entries = []
        for oidx, otu in enumerate(mcmc.graph.data.taxa):
            fwsim_entries.append({
                "OTU": otu.name,
                "SampleIdx": gibbs_idx,
                "SteadyState": stable_levels[oidx]
            })

        yield fwsim_entries, perturbed_otus



def perturbed_forward_sims(
        mcmc: md2.BaseMCMC,
        study: md2.Study,
        frac_otus_to_perturb: float,
        pert_strength: float,
        pert_start_day,
        pert_end_day,
        initial_conditions,
        dt,
        sim_max,
        n_days,
        gibbs_indices: List[int],
        master_seed: Seed
) -> Iterator[Tuple[int, np.ndarray]]:
    growth = mcmc.graph[STRNAMES.GROWTH_VALUE].get_trace_from_disk(section="posterior")
    self_interactions = mcmc.graph[STRNAMES.SELF_INTERACTION_VALUE].get_trace_from_disk(section="posterior")
    interactions = mcmc.graph[STRNAMES.INTERACTIONS_OBJ].get_trace_from_disk(section="posterior")
    interactions[np.isnan(interactions)] = 0
    self_interactions = -np.absolute(self_interactions)
    for i in range(self_interactions.shape[1]):
        interactions[:, i, i] = self_interactions[:, i]

    # Create the perturbation effect matrix. (Key idea: Re-sample perturbed OTUs for each gibbs sample.)
    n_otus_perturb = max(0, int(len(study.taxa) * frac_otus_to_perturb))
    rng = np.random.default_rng(master_seed.next_value())
    otus_to_perturb = rng.choice(
        a=len(study.taxa),
        size=n_otus_perturb,
        replace=False
    )
    perturbations = np.zeros(shape=(growth.shape[0], len(study.taxa)))
    perturbations[:, otus_to_perturb] = pert_strength

    for gibbs_idx in gibbs_indices:
        yield gibbs_idx, _forward_sim(
            growth=growth,
            interactions=interactions,
            perturbations=perturbations,
            pert_start_day=pert_start_day,
            pert_end_day=pert_end_day,
            initial_conditions=initial_conditions,
            dt=dt,
            sim_max=sim_max,
            n_days=n_days,
            gibbs_idx=gibbs_idx
        ), otus_to_perturb


def _forward_sim(growth,
                 interactions,
                 perturbations,
                 pert_start_day,
                 pert_end_day,
                 initial_conditions,
                 dt,
                 sim_max,
                 n_days,
                 gibbs_idx):
    '''Forward simulate with the given dynamics. First start with the perturbation
    off, then on, then off.

    Parameters
    ----------
    growth : np.ndarray(n_gibbs, n_taxa)
        Growth parameters
    interactions : np.ndarray(n_gibbs, n_taxa, n_taxa)
        Interaction parameters
    perturbations : np.ndarray(n_gibbs, n_taxa)
        Perturbation effect
    initial_conditions : np.ndarray(n_taxa)
        Initial conditions of the taxa
    dt : float
        Step size to forward simulate with
    sim_max : float, None
        Maximum clip for forward sim
    pert_start_day : float
        Day to start the perturbation
    pert_end_day : float
        Day to end the perturbation
    n_days : float
        Total number of days
    '''
    dyn = md2.model.gLVDynamicsSingleClustering(growth=None, interactions=None,
                                                perturbation_ends=[pert_end_day], perturbation_starts=[pert_start_day],
                                                start_day=0, sim_max=sim_max)
    initial_conditions = initial_conditions.reshape(-1, 1)

    dyn.growth = growth[gibbs_idx]
    dyn.interactions = interactions[gibbs_idx]
    dyn.perturbations = [perturbations[gibbs_idx]]

    x = md2.integrate(dynamics=dyn, initial_conditions=initial_conditions,
                      dt=dt, n_days=n_days, subsample=False)
    gibbs_step_sim = x['X']
    return gibbs_step_sim


if __name__ == "__main__":
    main()
