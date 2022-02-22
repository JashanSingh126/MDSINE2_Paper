#Phylogenetic neighborhood analysis

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy import stats
import pylab

def compute_coclustering_probability(otu_li1, otu_li2, data, order_d, type_):
    """computes the co-occurance probability between two agglomerations

       @parameters
       otu_li1, otu_li2 : ([str]) list of otus present in an agglomeration
       data : (np.ndarray) co-occurance probability matrix
       order_d : (dict)(str) otu_id ->  (int) index of the otu
    """
    epsilon = 1e-16
    total_p = []
    count = 0
    all_nan = True
    for otu1 in otu_li1:
        for otu2 in otu_li2:
            p = data[order_d[otu1], order_d[otu2]]
            all_nan = all_nan and np.isnan(p)
            if not np.isnan(p):
                if type_ == "arithmetic":
                    total_p.append(p)
                elif type_ == "geometric":
                    total_p.append(p + epsilon)
                count += 1
    if all_nan:
        return np.nan
    else:
        #print(len(total_p))
        mean = 0
        if len(total_p) != 0:
            if type_ == "geometric":
                mean = stats.gmean(total_p)#np.prod(total_p) ** (1 / len(total_p))
            elif type_ == "arithmetic":
                mean = np.mean(total_p)
        return mean

def merge_prob(agg_, ini_prob_mat, order_d, type_):
    """merge the probability vectors based on agglomeration in
       sequence_distance_space and returns the result as a (np.array) matrx

       @parameters
       agg_ : (dict) (int) agg id -> ([str]) list of otus in that agglomerate
       ini_prob_mat : (nd_array) otu- otu co-occurance probability matrix
    """

    N = len(agg_)
    c = 0
    agg_prob_matrix = np.zeros((N, N))
    for k1 in agg_:
        c += 1
        for k2 in agg_:
            if k1 == k2:
                agg_prob_matrix[k1, k2] = 0
            else:
                p = compute_coclustering_probability(agg_[k1], agg_[k2], ini_prob_mat,
                order_d, type_)
                agg_prob_matrix[k1, k2] = p
    return agg_prob_matrix

def linkage_info(A, mat, order):
    """
    finds the clusters with smallest distance and the corresponding distance

    @parameter
    A : (dict: (int) -> ([str])) the agglomerative clusters
    mat : the percent identity matrix
    order : (dict : (str) otu_name -> (int) index of the otu in mat)

    @returns
    float, int, int
    """

    all_keys = [keys for keys in A]
    dist_mat = 1 - mat
    d_min = np.inf
    min_1 = 0
    min_2 = 0
    for agg1 in A:
        for agg2 in A:
            if agg1 != agg2:
                distance = calc_cluster_distance(A[agg1], A[agg2], dist_mat,
                    order)
                if distance < d_min:
                    min_1 = agg1
                    min_2 = agg2
                    d_min = distance

    return d_min, min_1, min_2

def calc_cluster_distance(li1, li2, mat, order_d):
    """
    calculates the distance between two agglomerative clusters

    @parameters
    li1, li2 : ([str]) list containing the names of OTUs in the two clusters
    mat : (np.array) the distance matrix
    """
    distance = 0
    for otu1 in li1:
        index1 = order_d[otu1]
        for otu2 in li2:
            index2 = order_d[otu2]
            distance += mat[index1, index2]

    return distance / (len(li1)* len(li2))

def reorder_active_set(A, arg_min1, arg_min2):
    """
    rearranges the dictionary A

    @parameters
    arg_min1, arg_min2 : (int)
    """

    new_set = {}
    i = 0
    for key in A:
        if key != arg_min1 and key != arg_min2:
            new_set[i] = A[key]
            i += 1

    merged_set = A[arg_min1] + A[arg_min2]
    new_set[i] = merged_set

    return new_set

def clusterize(labels, otu_li):
    """
    groups otus according to the clusters they belong to

    @parameters
    labels : ([int]) the number at each index corresponds to the cluster to
             which the OTU given by th same index belongs
    otu_li : ([str]) a list containing the names of OTUs

    @returns
    (dict) (int) cluster_id -> ([str]) names of OTUs in the cluster
    """

    clusters = {}
    count = 0
    for cl in labels:
        if cl not in clusters:
            clusters[cl] = [otu_li[count]]
        else:
            clusters[cl].append(otu_li[count])

        count += 1

    return clusters

def agglomerative_clustering_scikit(pi_matrix, threshold, otu_ordered):
    """
    implements the agglomerative clustering algorithm using scikit

    @parametrs
    pi_matrix : (np.array) the percent identity matrix
    threshold : (float) the distance below which agglomerative clusters are merged
    otu_ordered : ([str]) a list containing the otus in order

    @return
    (dict) (int) cluster_id -> ([str]) names of OTUs in the cluster
    """

    dist_mat = 1 - pi_matrix

    agg_cl = AgglomerativeClustering(n_clusters = None, affinity = "precomputed",
    linkage = "average", distance_threshold = threshold).fit(dist_mat)
    clusters = clusterize(agg_cl.labels_, otu_ordered)

    return clusters


def agglomerative_clustering_thresh(pi_mat, thresh, order_d, otu_li):
    """
    implements the agglomerative clustering algorithm

    @parametrs
    pi_matrix : (np.array) the percent identity matrix
    thresh : (float) the distance below which agglomerative clusters are merged
    otu_li : ([str]) a list containing the otus in order
    order_d : (dict) (str) otu_name : (int) index of the otu

    @return
    (dict) (int) cluster_id -> ([str]) names of OTUs in the cluster
    """

    unique_pi = {}
    active_set = {}
    for i in range(len(otu_li)):
        active_set[i] = [otu_li[i]]
    unique_pi[thresh] = active_set
    min_d, arg_min_d1, arg_min_d2 = linkage_info(active_set, pi_mat, order_d)

    count = len(otu_li)
    while True:
        min_d, arg_min_d1, arg_min_d2 = linkage_info(active_set, pi_mat, order_d)
        if min_d > thresh:
            break
        active_set = reorder_active_set(active_set, arg_min_d1, arg_min_d2)
        if len(active_set) == 1:
            print("One cluster")
            unique_pi[thresh] = active_set
            break
        unique_pi[thresh] = active_set
        count += 1
        #print("unique:", unique_pi)
    return unique_pi[thresh]
