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
    parser.add_argument('-o', '--out_path',
                        help='Path to output to.')
    parser.add_argument('-t', '--thresh', type=float, required=False, default=1e20)

    return parser.parse_args()


def load_matrices(mcmc_path):
    mcmc = md2.BaseMCMC.load(mcmc_path)
    return mcmc.graph[STRNAMES.INTERACTIONS_OBJ].get_trace_from_disk(section='posterior')


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

    healthy_eig = compute_eigenvalues(load_matrices(args.healthy), args.thresh)
    uc_eig = compute_eigenvalues(load_matrices(args.uc), args.thresh)

    # ================== Some statistics
    print(" ----------------- Eigenvalue statistics: ----------------")
    X_healthy = healthy_eig.real.flatten()
    healthy_pos_frac = np.sum(X_healthy > 0) / X_healthy.shape[0]
    X_uc = uc_eig.real.flatten()
    uc_pos_frac = np.sum(X_uc > 0) / X_uc.shape[0]
    print("Healthy > 0 frac: ", np.mean(healthy_pos_frac))
    print("UC > 0 frac: ", np.mean(uc_pos_frac))
    stat, pval = scipy.stats.ranksums(healthy_pos_frac, uc_pos_frac)
    print("Rank-sum -- stat: {}, pval: {}".format(stat, pval))
    # ===========================================

    save_eigenvalues(healthy_eig, uc_eig, args.out_path)
    logging.info("Saved eigenvalues to {}.".format(args.out_path))


if __name__ == "__main__":
    main()
