from pathlib import Path
from typing import Union, List

import numpy as np
import pandas as pd
import mdsine2 as md2
from mdsine2 import BaseMCMC
from mdsine2.names import STRNAMES
from mdsine2.logger import logger
import argparse

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser("Forward simulate by excluding a cluster from the day-20 levels.")
    parser.add_argument('--input-mcmc-fixed-cluster', type=str, dest='input_mcmc', required=True,
                        help='<Required> Location of input (either folder of the numpy arrays or ' \
                             'MDSINE2.BaseMCMC chain). MUST BE THE FIXED CLUSTER MCMC RUN.')

    parser.add_argument('--study', type=str, dest='study', required=True,
                        help='<Required> Study object to use for initial conditions')
    parser.add_argument('--out-path', '-o', type=str, dest='out_path', required=True,
                        help='<Required> The path to which to save the calculated DataFrame '
                             'of simulated steady states.')

    # Simulation params
    parser.add_argument('--n-days', type=int, dest='n_days', required=False,
                        help='Total umber of days to simulate for', default=180)
    parser.add_argument('--simulation-dt', type=float, dest='simulation_dt', required=False,
                        help='Timesteps we go in during forward simulation', default=0.01)
    parser.add_argument('--limit-of-detection', dest='limit_of_detection', required=False,
                        help='If any of the taxa have a 0 abundance at the start, then we ' \
                             'set it to this value.', default=1e5, type=float)
    parser.add_argument('--sim-max', dest='sim_max', type=float, required=False,
                        help='Maximum value', default=1e20)

    return parser.parse_args()


def check_pytables():
    try:
        import tables
    except ImportError as e:
        raise RuntimeError("Package `pytables` is required for this script (originally, was optional for pandas).")


def main():
    check_pytables()
    args = parse_args()
    study = md2.Study.load(args.study)
    mcmc = md2.BaseMCMC.load(args.input_mcmc)
    initial_conditions_master = generate_initial_condition(study, args.limit_of_detection)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    df_entries = []

    # Cluster exclusion
    for cluster_idx, cluster in enumerate(mcmc.graph[STRNAMES.CLUSTERING_OBJ]):
        initial_conditions = exclude_cluster_from(initial_conditions_master, cluster)
        compute_keystoneness_of_cluster(
            mcmc,
            cluster_idx,
            initial_conditions,
            args.n_days,
            args.dt,
            args.sim_max,
            df_entries
        )

    # Baseline
    compute_keystoneness_of_cluster(
        mcmc,
        None,
        initial_conditions_master,
        args.n_days,
        args.dt,
        args.sim_max,
        df_entries
    )

    df = pd.DataFrame(df_entries)
    del df_entries

    df.to_hdf(args.out_path, key='df', mode='w')


def exclude_cluster_from(initial_conditions_master: np.ndarray, cluster):
    initial_conditions = np.copy(initial_conditions_master)
    for oidx in cluster.members:
        initial_conditions_master[oidx] = 0.0
    return initial_conditions


def compute_keystoneness_of_cluster(
        mcmc: BaseMCMC,
        cluster_idx: Union[int, None],
        initial_conditions: np.ndarray,
        n_days: int,
        dt: float,
        sim_max: float,
        df_entries: List,
):
    taxa = mcmc.graph.data.taxa

    # forward simulate and add results to dataframe.
    if cluster_idx is None:
        tqdm_disp = "Keystoneness Simulations (Baseline)"
    else:
        tqdm_disp = f"Keystoneness Simulations (Cluster {cluster_idx})"

    for gibbs_idx, fwsim in tqdm(do_fwsims(
        mcmc, initial_conditions, n_days, dt, sim_max
    ), total=mcmc.n_samples, desc=tqdm_disp):
        for entry in fwsim_entries(taxa, cluster_idx, fwsim, gibbs_idx):
            df_entries.append(entry)


def fwsim_entries(taxa, excluded_cluster_idx, fwsim, gibbs_idx):
    stable_states = fwsim[:, -50:].mean(axis=1)
    for otu in taxa:
        yield {
            "ExcludedCluster": str(excluded_cluster_idx),
            "OTU": otu.name,
            "SampleIdx": gibbs_idx,
            "StableState": stable_states[otu.idx]
        }


def generate_initial_condition(study: md2.Study, limit_of_detection: float):
    print("Generating initial conditions from Day 20 measurements.")
    M = study.matrix(dtype='abs', agg='mean', times='intersection', qpcr_unnormalize=True)
    day20_state = M[:, 19]

    abundances = day20_state
    abundances[abundances < limit_of_detection] = limit_of_detection
    return abundances


def do_fwsims(mcmc,
              initial_conditions: np.ndarray,
              n_days,
              dt: float,
              sim_max
              ):

    # Forward simulate if necessary
    # -----------------------------
    logger.info('Forward simulating')

    # Load the rest of the parameters
    growth = mcmc.graph[STRNAMES.GROWTH_VALUE].get_trace_from_disk(section="posterior")
    self_interactions = mcmc.graph[STRNAMES.SELF_INTERACTION_VALUE].get_trace_from_disk(section="posterior")
    interactions = mcmc.graph[STRNAMES.INTERACTIONS_OBJ].get_trace_from_disk(section="posterior")
    interactions[np.isnan(interactions)] = 0
    self_interactions = -np.absolute(self_interactions)
    for i in range(self_interactions.shape[1]):
        interactions[:, i, i] = self_interactions[:, i]

    num_samples = mcmc.n_samples

    # Do the forward sim.
    for gibbs_idx, gibbs_step_sim in _forward_sim(
            growth=growth,
            interactions=interactions,
            initial_conditions=initial_conditions,
            dt=dt,
            sim_max=sim_max,
            n_days=n_days,
            num_samples=num_samples
    ):
        yield gibbs_idx, gibbs_step_sim


def _forward_sim(growth,
                 interactions,
                 initial_conditions,
                 dt,
                 sim_max,
                 n_days,
                 num_samples):
    dyn = md2.model.gLVDynamicsSingleClustering(growth=None, interactions=None,
                                                perturbation_ends=[], perturbation_starts=[],
                                                start_day=0, sim_max=sim_max)
    initial_conditions = initial_conditions.reshape(-1, 1)
    for gibb in range(num_samples):
        dyn.growth = growth[gibb]
        dyn.interactions = interactions[gibb]
        dyn.perturbations = None

        x = md2.integrate(dynamics=dyn, initial_conditions=initial_conditions,
                          dt=dt, n_days=n_days, subsample=False)
        gibbs_step_sim = x['X']
        yield gibb, gibbs_step_sim


if __name__ == "__main__":
    main()
