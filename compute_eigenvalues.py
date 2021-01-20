import os

import numpy as np
import scipy.stats
import argparse
import mdsine2 as md2
from mdsine2.names import STRNAMES
from tqdm import tqdm

import logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute eigenvalues for stability analysis."
    )

    parser.add_argument('--healthy', required=True,
                        help='Path to the MCMC pickle file for healthy.')
    parser.add_argument('--uc', required=True,
                        help='Path to the MCMC pickle file for UC.')
    parser.add_argument('-o', '--out_dir',
                        help='Directory to output to.')
    parser.add_argument('-t', '--thresh', type=float, required=False, default=1e20)

    return parser.parse_args()


def load_matrices(mcmc_path):
    mcmc = md2.BaseMCMC.load(mcmc_path)
    si_trace = -np.absolute(mcmc.graph[STRNAMES.SELF_INTERACTION_VALUE].get_trace_from_disk(section='posterior'))
    interactions = mcmc.graph[STRNAMES.INTERACTIONS_OBJ].get_trace_from_disk(section='posterior')

    interactions[np.isnan(interactions)] = 0
    for i in range(len(mcmc.graph.data.taxa)):
        interactions[:,i,i] = si_trace[:,i]
    return interactions


def compute_eigenvalues(matrices, thresh):
    N = matrices.shape[0]
    M = matrices.shape[1]
    eigs = np.zeros(shape=(N, M), dtype=np.complex)
    for i in tqdm(range(N)):  # range(arr.shape[0]):
        matrix = matrices[i]

        # only for interaction matrices
        matrix = np.nan_to_num(matrix, nan=0.0)

        slice_eigs = np.linalg.eigvals(matrix)

        # Throw out samples where eigenvalues blow up
        if np.sum(np.abs(slice_eigs) > thresh) > 0:
            logging.info("Threshold {th} passed for sample {i}; skipping.".format(
                th=thresh,
                i=i
            ))
            continue

        eigs[i, :] = slice_eigs
    return eigs


def save_eigenvalues(healthy, uc, out_path):
    np.savez(out_path, healthy=healthy, uc=uc)


def main():
    args = parse_args()
    md2.config.LoggingConfig(level=logging.INFO)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    outpath = os.path.join(args.out_dir, "eigenvalues.npz")

    healthy_eig = compute_eigenvalues(load_matrices(args.healthy), args.thresh)
    uc_eig = compute_eigenvalues(load_matrices(args.uc), args.thresh)

    # ================== Some statistics
    print(" ----------------- Eigenvalue statistics: ----------------")
    X_healthy = healthy_eig.real
    healthy_pos_frac = np.sum(X_healthy > 0, axis=1) / X_healthy.shape[1]
    X_uc = uc_eig.real
    uc_pos_frac = np.sum(X_uc > 0, axis=1) / X_uc.shape[1]
    print("mean (Healthy > 0 frac): ", np.mean(healthy_pos_frac))
    print("mean (UC > 0 frac): ", np.mean(uc_pos_frac))
    stat, pval = scipy.stats.ranksums(healthy_pos_frac, uc_pos_frac)
    print("Rank-sum -- stat: {}, pval: {}".format(stat, pval))
    # ===========================================

    save_eigenvalues(healthy_eig, uc_eig, outpath)
    logging.info("Saved eigenvalues to {}.".format(outpath))


if __name__ == "__main__":
    main()
