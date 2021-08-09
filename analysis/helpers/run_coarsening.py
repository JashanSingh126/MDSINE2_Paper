#runs coarsening (Phylogenetic neighborhood Analysis)

import numpy as np
import spearman as sp
import matplotlib.pyplot as plt
import pandas as pd
import coarsening_helper as helper
from mdsine2.names import STRNAMES
import PNA as pna
import sequence_analyzer as analyzer
import Bio
import mdsine2 as md2
import argparse
import os

def parse_arguments():

    parser = argparse.ArgumentParser(description = "arguments for phylogenetic"\
        " neighborhood analysis")
    parser.add_argument("-mcmc1", "--mcmc_healthy", required = "True",
        help = ".pkl file containing the results of the mcmc run for healthy ")
    parser.add_argument("-mcmc2", "--mcmc_uc", required = "True",
        help = ".pkl file containing the results of the mcmc run for UC")
    parser.add_argument("-pkl1", "--pkl_healthy", required = "True",
        help = ".pkl file containing details of subjects in healthy cohort")
    parser.add_argument("-pkl2", "--pkl_uc", required = "True",
        help = ".pkl file containing details of subjects in UC cohort")
    parser.add_argument("-sf", "--seq_file", required  = True,
        help = "a .sto file containing the details of the alignment")
    parser.add_argument("-m_type", "--mean_type", required = False,
        default = "arithmetic", help = "the type of mean to use")
    parser.add_argument("-k", "--num_samples", required = True, type = int,
        help = "the number of bootstraped samples")
    parser.add_argument("-o", "--output_loc", required = True,
        help = "location of the folder where outputs are saved")

    return parser.parse_args()


def generate_null_distributions(otu_li, K = 10000):
    """
    generates K null distribution by randomly shuffling the list of otus

    @parameters
    otu_li : ([str]) a list containing the names of otus in order of appearence

    @returns
    a list of dictionaries with (str) otu_name -> (int) index
    """

    null_li1 = []
    null_li2 = []
    for i in range(K):
        random_order = np.arange(0, len(otu_li))
        np.random.shuffle(random_order)
        otu_union_random = [otu_li[i] for i in random_order]
        otu_union_random_d = {otu_union_random[i] : i for i in
                             range(len(otu_union_random))}
        null_li1.append(otu_union_random)
        null_li2.append(otu_union_random_d)
    pd.DataFrame(np.asarray(null_li1)).to_csv("shuffled_labels.csv",
    header = None, index = None)

    return null_li2

def get_name(taxo):
    """
    return the lowest defined hierachicy (str)

    @parameters
    taxo : pl.base.taxonomy (works like a dictionary)
    """

    if taxo["species"] != "NA":
        return taxo["genus"] + " " + taxo["species"]
    elif taxo["genus"] != "NA":
        return taxo["genus"]
    elif taxo["family"] != "NA":
        return taxo["family"]
    elif taxo["order"] != "NA":
        return  taxo["order"]
    elif taxo["class"] != "NA":
        return taxo["class"]
    elif taxo["phylum"] != "NA":
        return taxo["phylum"]
    else:
        return taxo["kingdom"]

def get_names(subjset_healthy, subjset_uc, union):

    """
    returns the names of all the ASVs in union as a dictionary ((str) otu_id ->
    (str) the otus name)

    @parameters
    subjset_healthy, subjset_uc : (pl.base.Study) contains information about
    the ASVs in healthy and UC subjset
    union : ([str]) names of ASVs in healthy and UC post filtering
    """

    taxa_names = {}
    for taxa in subjset_uc.taxa:
        if taxa.name in union:
            taxo = taxa.taxonomy
            #print(taxa)
            taxa_names[taxa.name] = get_name(taxo)

    for taxa in subjset_healthy.taxa:
        if taxa.name in union and taxa.name not in taxa_names:
            taxo = taxa.taxonomy
            taxa_names[taxa.name] = get_name(taxo)
    taxa_names_li = []
    for keys in taxa_names:
        taxa_names_li.append(taxa_names[keys])
    return taxa_names

if __name__ == "__main__":

    args = parse_arguments()
    subjset_healthy = md2.Study.load(args.pkl_healthy)
    subjset_uc = md2.Study.load(args.pkl_uc)
    seq_loc = args.seq_file

    mcmc_healthy = md2.BaseMCMC.load(args.mcmc_healthy + "mcmc.pkl")
    mcmc_uc = md2.BaseMCMC.load(args.mcmc_uc + "mcmc.pkl")

    cluster_healthy_d = helper.parse_cluster(mcmc_healthy)
    cluster_uc_d = helper.parse_cluster(mcmc_uc)

    healthy_cocluster_df = helper.generate_cocluster_prob(mcmc_healthy)
    uc_cocluster_df = helper.generate_cocluster_prob(mcmc_uc)

    otus_healthy = list(healthy_cocluster_df.columns)
    otus_uc = list(uc_cocluster_df.columns)

    otus_union = list(set(otus_healthy).union(set(otus_uc)))
    otus_union_ordered, otus_union_ordered_d = helper.order_otus(otus_union)
    otus_intersection = list(set(otus_healthy).intersection(set(otus_uc)))

    otus_healthy_ordered_d = {otus_healthy[i] : i for i in range(len(otus_healthy))}
    otus_uc_ordered_d = {otus_uc[i] : i for i in range(len(otus_uc))}

    seq_union = analyzer.get_sequences_scratch(otus_union_ordered, seq_loc)
    pi_d, pi_order = analyzer.compute_percent_identity_all(seq_union)
    names_d = get_names(subjset_healthy, subjset_uc, otus_union_ordered_d)

    all_keys = [keys for keys in pi_d]
    pi_matrix = np.asarray([pi_d[keys] for keys in all_keys])

    #loc = "_".join(args.mcmc_healthy.split("/")[-2].split("-")[0:] +
    #           args.mcmc_uc.split("/")[-2].split("-")[0:])
    loc = args.output_loc
    if not os.path.exists(loc):
        os.makedirs(loc, exist_ok = True)
    analyzer.sanity_check(pi_d, pi_order, names_d, loc)

    coclusters_healthy = helper.load_cocluster_data(healthy_cocluster_df.to_numpy(),
    otus_union_ordered_d, otus_healthy_ordered_d, otus_union_ordered)
    coclusters_uc = helper.load_cocluster_data(uc_cocluster_df.to_numpy(),
    otus_union_ordered_d, otus_uc_ordered_d, otus_union_ordered)

    thresholds1 = np.linspace(0.01, 0.15, 29)
    thresholds2 = [0.16, 0.165 ,0.18, 0.185, 0.19, 0.195, 0.2, 0.205, 0.21, 0.23,
    0.24, 0.25]
    thresholds = [0] + list(thresholds1) + thresholds2

    agg_clusters = {}
    print("Running PNA")
    for t in thresholds:
        clusters = pna.agglomerative_clustering_scikit(pi_matrix,t,
        otus_union_ordered)
        #print(t, len(clusters))
        agg_clusters[t] = clusters
        #print()

    K = args.num_samples
    print("Generating Null Distributions")
    null_ = generate_null_distributions(otus_union_ordered, K)
    type_ = args.mean_type
    rho_data_all = []
    rho_null_all = []
    rho_null_mean = []
    rho_null_std = []
    distance_all = []

    print("Running Analysis")
    for d in agg_clusters:
        if len(agg_clusters[d]) != 1:
            merged_uc_data = pna.merge_prob(agg_clusters[d], coclusters_uc,
            otus_union_ordered_d, type_)
            merged_healthy_data = pna.merge_prob(agg_clusters[d], coclusters_healthy,
            otus_union_ordered_d, type_)

            distance_all.append(d)
            rho_data = sp.compute_average_spearman(merged_healthy_data,
            merged_uc_data, "mean")
            print("d :", d, "rho data:", rho_data)
            rho_data_all.append(rho_data)

            rho_null_K = []
            for k in range(K):
                merged_uc_null = pna.merge_prob(agg_clusters[d], coclusters_uc,
                null_[k], type_)
                merged_healthy_null = pna.merge_prob(agg_clusters[d], coclusters_healthy,
                null_[k], type_)
                rho_k = sp.compute_average_spearman(merged_healthy_null,
                merged_uc_null, "mean")
                rho_null_K.append(rho_k)
            print("d:", d, "rho_null", np.mean(rho_null_K))
            print()
            rho_null_all.append(rho_null_K)
            rho_null_mean.append(np.mean(rho_null_K))
            rho_null_std.append(np.std(rho_null_K))

    name = type_ + "_mean"

    pd.DataFrame(rho_data_all).to_csv(loc + "/" + name + "_data.csv")
    pd.DataFrame(rho_null_all).to_csv(loc + "/" + name + "_null_all.csv")
    pd.DataFrame(distance_all).to_csv(loc + "/" + "distance.csv")
