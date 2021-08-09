#helpers functions for coarsening

import numpy as np
import seaborn as sns
import pandas as pd
from mdsine2.names import STRNAMES
import mdsine2 as md2

def clusterize(labels, taxa_list):
    """
    returns the consenus cluster as a dictionary ((int) cluster_id ->
    ([str]) ids of OTUs belonging to the cluster)

    @parameters
    labels : ([int]) the consensus cluster id of OTUs in taxa_list
    taxa_list : ([str]) ids of OTU
    """

    cluster = {}
    for i in range(len(labels)):
        if labels[i] + 1 not in cluster:
            cluster[labels[i] + 1] = []
        cluster[labels[i] + 1].append(taxa_list[i])

    return cluster

def parse_cluster(mcmc):
    """
    obtains the consensus cluster from the mcmc (pl.Base.Study) file

    @returns
    (dict) : (int) cluster_id -> ([str]) OTUs in the cluster
    """


    clustering = mcmc.graph[STRNAMES.CLUSTERING].clustering
    consenus = md2.util.generate_cluster_assignments_posthoc(clustering=clustering,
    set_as_value=True)
    taxa_list = []
    taxas = mcmc.graph.data.taxa
    for taxa in taxas:
	    taxa_list.append(taxa.name)

    cluster = clusterize(consenus, taxa_list)
    return cluster

def generate_cocluster_prob(mcmc):
    """
    generates the co-cluster probability from the mcmc file

    @returns
    df : the cocluster probability; otu names are the row and column names
    """

    clustering = mcmc.graph[STRNAMES.CLUSTERING_OBJ]
    cocluster_trace = clustering.coclusters.get_trace_from_disk()
    coclusters = md2.summary(cocluster_trace)['mean']

    taxa_list = [otu.name for otu in mcmc.graph.data.taxa]

    df = pd.DataFrame(coclusters, index = taxa_list, columns = taxa_list)
    return df

def order_otus(otu_li):
    """
    sort the otus according to their id

    @parameters 
    otu_li : [str] list of OTU ids

    @returns
    list [str], dict {(str) -> int}

    """
    ordered = np.sort([int(x.split("_")[1]) for x in otu_li])
    ordered = ["OTU_" + str(id) for id in ordered]
    ordered_dict = {ordered[i] : i for i in range(len(ordered))}

    return ordered, ordered_dict

def load_cocluster_data(data_matrix, union_d, flora_d, union_li):
    """
    returns the co-cluster matrix (np.array)

    @parameters
    data_matrix : (np.arr) co_cluster probability matrix
    union_d : (dict) (str) name of ASV -> (int) index
    flora_d : (dict) (str) name of ASV -> (int) index
    union_li : [str] ids of asv present in the union
    """

    N = len(union_d)
    cocluster_matrix_union = np.zeros((N, N))
    for i in range(N):
        asv_i = union_li[i]
        for j in range(N):
            asv_j = union_li[j]
            if asv_i in flora_d and asv_j in flora_d:
                prob = data_matrix[flora_d[asv_i], flora_d[asv_j]]
                cocluster_matrix_union[i, j] = prob
            else:
                cocluster_matrix_union[i, j] = np.nan

    return cocluster_matrix_union
