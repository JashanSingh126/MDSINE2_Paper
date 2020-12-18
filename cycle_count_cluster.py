"""
  Author: Younhun Kim
  Date: 2020/08/20

  Implementation of a depth-first search with pruning to find frequently occurring cycles and chains.
"""
import os
import time

import numpy as np
import scipy, scipy.stats, scipy.integrate
import argparse
from networkx import nx
from typing import List

from tqdm import tqdm
import mdsine2 as md2
from mdsine2.names import STRNAMES

import logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cycle counting for fixed cluster analysis."
                    "Output results in CSV format (separator is semicolon, not comma)"
    )

    # Input specification.
    parser.add_argument('-c', '--mcmc_path', required=True,
                        help='<Required> Path to saved markov chain pickle file.')
    parser.add_argument('-o', '--out_dir', required=True,
                        help='<Required> The directory to output results to.')
    parser.add_argument('--do_chains', action="store_true")
    parser.add_argument("--max_path_len", required=False, default=4, type=int)
    return parser.parse_args()


def tax_string(taxa, taxonomy_dict):
    phylum = taxonomy_dict["phylum"]
    class_ = taxonomy_dict["class"]
    order = taxonomy_dict["order"]
    family = taxonomy_dict["family"]
    genus = taxonomy_dict["genus"]
    species = taxonomy_dict["species"]
    if phylum == "NA":
        raise ValueError("Classification of Taxa {} not specified, even at Phylum level.".format(taxa))
    elif class_ == "NA":
        return "{}*****".format(phylum)
    elif order == "NA":
        return "{}****".format(class_)
    elif family == "NA":
        return "{}***".format(order)
    elif genus == "NA":
        return "{}**".format(family)
    elif species == "NA":
        return "{}, {}*".format(family, genus)
    else:
        return "{}, {} {}".format(family, genus, species)


def load_cluster_interactions(chain_path):
    mcmc = md2.BaseMCMC.load(chain_path)
    otu_interactions = mcmc.graph[STRNAMES.INTERACTIONS_OBJ].get_trace_from_disk(section='posterior')

    study = mcmc.graph.data.subjects
    taxa = study.taxa

    cluster_reps = []
    clusters = []

    for cluster in mcmc.graph[STRNAMES.CLUSTERING].clustering:
        cluster_reps.append(next(iter(cluster.members)))
        cluster_otu_arr = [
            "{} {}".format(taxa[idx].name, tax_string(taxa[idx].name, taxa[idx].taxonomy))
            for idx in cluster.members
        ]
        clusters.append(cluster_otu_arr)

    cluster_interactions = np.zeros(shape=(otu_interactions.shape[0],
                                           len(cluster_reps),
                                           len(cluster_reps)),
                                    dtype=np.float)

    for i in range(cluster_interactions.shape[0]):
        cluster_interactions[i] = otu_interactions[i][np.ix_(cluster_reps, cluster_reps)]
    return cluster_interactions, clusters


# ===================================================================================================
# =========================================== Path counting =========================================
# ===================================================================================================

def freq_cycles(interactions: np.ndarray,
                num_samples: int,
                min_thresh: int,
                max_len: int = 10,
                do_paths: bool = False,
                gamma_shape: float = 1e-5,
                gamma_scale: float = 1e5,
                beta1: float = 0.5,
                beta2: float = 0.5):
    """ A generator over frequent cycles (count at least count_thresh), implemented as a pruned depth-first-search. """
    N = interactions.shape[1]

    logging.info("Processing graph object...")
    # Instantiate a graph, associate each edge with a set of indices of matrices which contain the edge.
    start = time.time()
    graph = nx.DiGraph()
    graph.add_nodes_from(list(range(N)))
    for k, interaction_matrix in tqdm(enumerate(interactions), total=num_samples):
        for (i, j) in np.argwhere(~np.isnan(interaction_matrix)):
            add_idx_to_edge(graph, j, i, k, interaction_matrix[i, j])
    logging.info("Finished graph pre-processing ({:.2f} sec).".format(time.time() - start))
    logging.info("Performing depth-first exploration.")

    for src in tqdm(range(N)):
        for path, samples, path_signs in freq_paths_rooted(graph, src,
                                                           num_samples,
                                                           min_thresh=min_thresh,
                                                           max_len=max_len,
                                                           do_paths=do_paths):
            if path[0] == path[-1]:
                bayes = bayes_factor_cycle(path,
                                           len(samples),
                                           num_samples,
                                           gamma_shape=gamma_shape,
                                           gamma_scale=gamma_scale,
                                           beta1=beta1,
                                           beta2=beta2)
            else:
                bayes = bayes_factor_chain(path,
                                           len(samples),
                                           num_samples,
                                           gamma_shape=gamma_shape,
                                           gamma_scale=gamma_scale,
                                           beta1=beta1,
                                           beta2=beta2)
            yield path, list(samples), bayes, path_signs


def freq_paths_rooted(graph: nx.DiGraph,
                      start: int,
                      num_samples: int,
                      min_thresh: int,
                      max_len: int = 10,
                      do_paths: bool = False):
    rooted_path = [start]
    base_signs = ["" for _ in range(num_samples)]
    blocked_nodes = set()
    occurrences = set(range(num_samples))
    for path, samples, path_signs in freq_paths_rooted_recursive(graph,
                                                                 rooted_path,
                                                                 base_signs,
                                                                 blocked_nodes,
                                                                 occurrences,
                                                                 num_samples,
                                                                 min_thresh=min_thresh,
                                                                 max_len=max_len,
                                                                 do_paths=do_paths):
        yield path, samples, path_signs


def freq_paths_rooted_recursive(graph: nx.DiGraph,
                                cur_path: List,
                                path_signs: List[str],
                                blocked_nodes: set,
                                occurrences: set,
                                num_samples: int,
                                min_thresh: int,
                                max_len: int = 10,
                                do_paths: bool = False):
    if len(cur_path) == max_len + 1:
        return

    head = cur_path[0]
    tail = cur_path[-1]
    if do_paths:
        # Just make sure there is no cycle.
        neighbors = [v for v in graph[tail] if v not in blocked_nodes and v != head]
    else:
        # Induce canonical representation (head is lexicographically first in cycle)
        neighbors = [v for v in graph[tail] if v not in blocked_nodes and v >= head]

    for v in neighbors:
        path_valid_samples = occurrences.intersection(graph.edges[tail, v]["m_idx"])

        if len(path_valid_samples) >= min_thresh:
            cur_path.append(v)
            blocked_nodes.add(v)
            for i in path_valid_samples:
                path_signs[i] = path_signs[i] + get_sign(graph, cur_path[-2], cur_path[-1], i)

            if (not do_paths and v == head) or (do_paths and v != head):
                # Only return cycles, or only return chains depending on setting.
                yield (cur_path,
                       path_valid_samples,
                       [
                           path_sign
                           if idx in path_valid_samples
                           else ""
                           for idx, path_sign in enumerate(path_signs)
                       ])

            if v != head:
                # DFS recursion.
                for r_path, r_samples, r_signs in freq_paths_rooted_recursive(
                        graph, cur_path, path_signs, blocked_nodes, path_valid_samples, num_samples,
                        min_thresh=min_thresh,
                        max_len=max_len,
                        do_paths=do_paths
                ):
                    yield r_path, r_samples, r_signs
            del cur_path[-1]
            for i in path_valid_samples:
                path_signs[i] = path_signs[i][:-1]
            blocked_nodes.remove(v)


def add_idx_to_edge(graph: nx.DiGraph, u: int, v: int, idx: int, wt: float):
    """
    Add an edge (u -> v)
    :param graph: The graph object.
    :param u: Source vertex
    :param v: Sink vertex
    :param idx: The sample index to store in the edge.
    :param wt: The interaction weight for the specified sample index.
    """
    if not graph.has_edge(u, v):
        graph.add_edge(u, v, m_idx=set())
    graph.edges[u, v]['m_idx'].add(idx)
    graph.edges[u, v][idx] = weight_to_sign(wt)


def get_sign(graph: nx.DiGraph, u: int, v: int, idx: int):
    return graph.edges[u, v][idx]


def weight_to_sign(wt: float):
    if wt > 0:
        return '+'
    elif wt < 0:
        return '-'
    else:
        return '?'


# ===================================================================================================
# =========================================== Bayes factors =========================================
# ===================================================================================================

def chinese_restaurant_table_prob(partition, alpha, do_log=False):
    log_prob = 0.0
    running_total = 0
    for k in partition:
        log_prob += np.log(alpha) - np.log(alpha + running_total)
        running_total += 1
        for i in range(1, k):
            log_prob += np.log(i) - np.log(alpha + running_total)
            running_total += 1
    return log_prob if do_log else np.exp(log_prob)


def prior_probability_cycle_fixed_alpha(cycle, dirichlet_alpha, beta1, beta2):
    # beta_bernoulli_edge_log_prob = np.log(scipy.special.beta(beta1 + 1, beta2) / scipy.special.beta(beta1, beta2))
    beta_bernoulli_edge_log_prob = np.log(beta1 / (beta1 + beta2))  # Equivalent, but simplified

    if len(cycle)-1 == 2:
        log_prob = chinese_restaurant_table_prob(partition=(1, 1), alpha=dirichlet_alpha, do_log=True)
        log_prob += 2 * beta_bernoulli_edge_log_prob
        return np.exp(log_prob)
    if len(cycle)-1 == 3:
        log_prob = chinese_restaurant_table_prob(partition=(1, 1, 1), alpha=dirichlet_alpha, do_log=True)
        log_prob += 3 * beta_bernoulli_edge_log_prob
        return np.exp(log_prob)
    elif len(cycle)-1 == 4:
        prob_case1 = chinese_restaurant_table_prob(partition=(1, 1, 1, 1), alpha=dirichlet_alpha, do_log=False)
        prob_case1 += 2 * chinese_restaurant_table_prob(partition=(2, 1, 1), alpha=dirichlet_alpha, do_log=False)  # Either ind. set can be grouped
        log_prob_case1 = np.log(prob_case1) + 4 * beta_bernoulli_edge_log_prob

        log_prob_case2 = chinese_restaurant_table_prob(partition=(2, 2), alpha=dirichlet_alpha, do_log=True)
        log_prob_case2 += 2 * beta_bernoulli_edge_log_prob
        return np.exp(log_prob_case1) + np.exp(log_prob_case2)
    elif len(cycle)-1 == 5:
        prob_case1 = chinese_restaurant_table_prob(partition=(1, 1, 1, 1, 1), alpha=dirichlet_alpha, do_log=False)
        prob_case1 += 5 * chinese_restaurant_table_prob(partition=(2, 1, 1, 1), alpha=dirichlet_alpha, do_log=False)  # One of the five ind. sets
        log_prob_case1 = np.log(prob_case1) + 5 * beta_bernoulli_edge_log_prob

        log_prob_case2 = 5 * chinese_restaurant_table_prob(partition=(2, 2, 1), alpha=dirichlet_alpha, do_log=True)
        log_prob_case2 += 4 * beta_bernoulli_edge_log_prob
        return np.exp(log_prob_case1) + np.exp(log_prob_case2)
    else:
        raise RuntimeError("cycle too long!: ", cycle)


def prior_probability_chain_fixed_alpha(path, dirichlet_alpha, beta1, beta2):
    # beta_bernoulli_edge_log_prob = np.log(scipy.special.beta(beta1 + 1, beta2) / scipy.special.beta(beta1, beta2))
    beta_bernoulli_edge_log_prob = np.log(beta1 / (beta1 + beta2))  # Equivalent, but simplified

    if len(path) - 1 == 1:
        log_prob = chinese_restaurant_table_prob(partition=(1, 1), alpha=dirichlet_alpha, do_log=True)
        log_prob += beta_bernoulli_edge_log_prob
        return np.exp(log_prob)
    if len(path) - 1 == 2:
        log_prob = chinese_restaurant_table_prob(partition=(1, 1, 1), alpha=dirichlet_alpha, do_log=True)
        log_prob += chinese_restaurant_table_prob(partition=(2, 1), alpha=dirichlet_alpha, do_log=True)
        log_prob += 2 * beta_bernoulli_edge_log_prob
        return np.exp(log_prob)
    elif len(path) - 1 == 3:
        prob_case1 = chinese_restaurant_table_prob(partition=(1, 1, 1, 1), alpha=dirichlet_alpha, do_log=False)
        prob_case1 += 3 * chinese_restaurant_table_prob(partition=(2, 1, 1), alpha=dirichlet_alpha, do_log=False)  # Either ind. set can be grouped
        log_prob_case1 = np.log(prob_case1) + 3 * beta_bernoulli_edge_log_prob

        log_prob_case2 = chinese_restaurant_table_prob(partition=(2, 2), alpha=dirichlet_alpha, do_log=True)
        log_prob_case2 += 2 * beta_bernoulli_edge_log_prob
        return np.exp(log_prob_case1) + np.exp(log_prob_case2)
    elif len(path) - 1 == 4:
        cr_11111 = chinese_restaurant_table_prob(partition=(1, 1, 1, 1, 1), alpha=dirichlet_alpha,
                                                                 do_log=False)
        cr_2111 = chinese_restaurant_table_prob(partition=(2, 1, 1, 1), alpha=dirichlet_alpha,
                                                                 do_log=False)
        cr_221 = chinese_restaurant_table_prob(partition=(2, 2, 1), alpha=dirichlet_alpha,
                                                                 do_log=False)
        cr_311 = chinese_restaurant_table_prob(partition=(3, 1, 1), alpha=dirichlet_alpha,
                                                                 do_log=False)
        cr_32 = chinese_restaurant_table_prob(partition=(3, 2), alpha=dirichlet_alpha,
                                                                 do_log=False)

        edge_4 = beta_bernoulli_edge_log_prob * 4
        edge_3 = beta_bernoulli_edge_log_prob * 3
        edge_2 = beta_bernoulli_edge_log_prob * 2

        return (np.exp(np.log(cr_11111 + 6 * cr_2111 + 4 * cr_221 + cr_311) + edge_4)
                + np.exp(np.log(cr_221) + edge_3)
                + np.exp(np.log(cr_32) + edge_2))
    else:
        raise RuntimeError("path too long!: ", path)


def prior_probability_cycle(cycle, gamma_shape, gamma_scale, beta1, beta2):
    alphas = np.linspace(0, 20, num=1000)[1:]
    integrand = (
            prior_probability_cycle_fixed_alpha(cycle, alphas, beta1, beta2)
            * scipy.stats.gamma.pdf(x=alphas, a=gamma_shape, scale=gamma_scale)
    )
    return scipy.integrate.trapz(y=integrand, x=alphas)


def prior_probability_chain(path, gamma_shape, gamma_scale, beta1, beta2):
    alphas = np.linspace(0, 20, num=1000)[1:]
    integrand = (
            prior_probability_chain_fixed_alpha(path, alphas, beta1, beta2)
            * scipy.stats.gamma.pdf(x=alphas, a=gamma_shape, scale=gamma_scale)
    )
    return scipy.integrate.trapz(y=integrand, x=alphas)


def bayes_factor_cycle(cycle, sample_count, total_samples, gamma_shape, gamma_scale, beta1, beta2):
    posterior_prob = sample_count / total_samples
    prior_prob = prior_probability_cycle(cycle, gamma_shape, gamma_scale, beta1, beta2)
    numerator = posterior_prob * (1 - prior_prob)
    denominator = (1 - posterior_prob) * prior_prob
    return numerator / denominator


def bayes_factor_chain(path, sample_count, total_samples, gamma_shape, gamma_scale, beta1, beta2):
    posterior_prob = sample_count / total_samples
    prior_prob = prior_probability_chain(path, gamma_shape, gamma_scale, beta1, beta2)
    numerator = posterior_prob * (1 - prior_prob)
    denominator = (1 - posterior_prob) * prior_prob
    return numerator / denominator


# ===================================================================================================
# ============================================ Main script ==========================================
# ===================================================================================================

def output_clusters(clusters, cluster_path):
    with open(cluster_path, "w") as f:
        for i, cluster in enumerate(clusters):
            if i > 0:
                print("\n", file=f)
            print("Cluster {}".format(i+1), file=f)
            print("-------------", file=f)
            for otu_str in cluster:
                print(otu_str, file=f)


def main():
    args = parse_args()
    md2.config.LoggingConfig(level=logging.INFO)

    logging.info("Loading data from {}.".format(args.mcmc_path))
    interactions, clusters = load_cluster_interactions(args.mcmc_path)

    start = time.time()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    logging.info("Writing outputs to directory {}.".format(args.out_dir))

    output_clusters(clusters, os.path.join(args.out_dir, "clusters.txt"))
    cycle_out_path = os.path.join(args.out_dir, "paths.csv")

    with open(cycle_out_path, "w") as outfile:
        for cycle, samples, bayes, path_signs in freq_cycles(
                interactions,
                num_samples=len(interactions),
                min_thresh=10,
                max_len=args.max_path_len,
                do_paths=args.do_chains
        ):
            path_signs_str = ",".join(path_signs)

            print("{};{};{};{}".format(
                "->".join([str(c) for c in cycle]),
                len(samples),
                bayes,
                path_signs_str
            ), file=outfile)

    logging.info("Computed cycles in {} min.".format(
        (time.time() - start) / 60
    ))


if __name__ == "__main__":
    main()
