from pathlib import Path
import argparse
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import mdsine2 as md2
from mdsine2.names import STRNAMES


def parse_args():
    parser = argparse.ArgumentParser(description="Plot the interaction eigenvalues.")

    # Input specification.
    parser.add_argument('-m', '--mcmc_path', required=True, type=str, action='append',
                        help='<Required> The path to the MCMC pickle files. Can pass in more than one.')
    parser.add_argument('-n', '--mcmc_names', required=True, type=str, action='append',
                        help='<Required> The names of the MCMC pickle files. '
                             'Length should match the number of MCMC pickles.')
    parser.add_argument('-p', '--plot_path', required=True, type=str,
                        help='<Required> The target output path of the plot')

    parser.add_argument('--format', required=False, type=str,
                        default='pdf',
                        help='<Optional> The plot output format. Default: `pdf`')

    return parser.parse_args()


def main():
    args = parse_args()
    eigenvalues = {}
    for mcmc_path, name in zip(args.mcmc_path, args.mcmc_names):
        mcmc_pickle_path = Path(mcmc_path)

        positive_real_eigs = compute_eigenvalues(mcmc_pickle_path).real.flatten()
        eigenvalues[name] = positive_real_eigs

    if len(eigenvalues) == 0:
        raise RuntimeError("Expected at least one pickle file.")

    plot_eigenvalues(eigenvalues, out_path=Path(args.plot_path), out_format=args.format)


def compute_eigenvalues(mcmc_pickle_path: Path, upper_bound: float = 1e20) -> np.ndarray:
    """
    Compute the eigenvalues of the provided interaction matrices.
    :param mcmc_pickle_path:
    :param upper_bound:
    :return:
    """
    # ================ Data loading
    mcmc = md2.BaseMCMC.load(str(mcmc_pickle_path))
    si_trace = -np.absolute(mcmc.graph[STRNAMES.SELF_INTERACTION_VALUE].get_trace_from_disk(section='posterior'))
    interactions = mcmc.graph[STRNAMES.INTERACTIONS_OBJ].get_trace_from_disk(section='posterior')

    interactions[np.isnan(interactions)] = 0
    for i in range(len(mcmc.graph.data.taxa)):
        interactions[:, i, i] = si_trace[:, i]

    # ================ Eigenvalue computation
    N = interactions.shape[0]
    M = interactions.shape[1]
    eigs = np.zeros(shape=(N, M), dtype=np.complex)
    for i in range(N):  # range(arr.shape[0]):
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

    return eigs


def plot_eigenvalues(eigenvalue_dict: Dict[str, np.ndarray], out_path: Path, out_format: str, alpha: float = 0.7):
    fig, ax = plt.subplots(figsize=(8, 6))
    bins = np.linspace(0, 5e-9, 45)
    for name, eigs in eigenvalue_dict.items():
        sns.histplot(eigs, ax=ax, bins=bins, label=name, alpha=alpha)

    ax.set_xlabel('Pos. Real Part of Eigenvalues', labelpad=20)
    ax.set_ylabel('Total count with multiplicity')
    ax.ticklabel_format(style="sci", scilimits=(0, 0), useMathText=True, useOffset=False)
    ax.legend()

    plt.savefig(out_path, format=out_format)


if __name__ == "__main__":
    main()
