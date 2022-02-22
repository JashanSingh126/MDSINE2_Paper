#functions that are useful in analyzing sequences directly

import pandas as pd
#import mdsine2 as md2
import numpy as np
from Bio import SeqIO
import mdsine2 as md2

def get_sequences_scratch(otu_li, path):
    """
    obtains the aligned consensus sequence

    @parameters
    otu_li :([str]) list containing the names of otus
    path : (str)  path to the stockholm file containing sequence details

    @returns
    (dict) (str) otu_id -> (str) aligned sequence
    """

    print("Loading sequences \n")

    seqs = SeqIO.to_dict(SeqIO.parse(path, format = "stockholm"))
    to_delete = []
    M = []
    #print(otu_li)
    for otu in otu_li:
        seq = list(str(seqs[otu].seq))
        M.append(seq)

    M = np.asarray(M)
    gaps = M == "-"
    n_gaps = np.sum(gaps, axis = 0)
    idxs = np.where(n_gaps == 0)[0]
    M = M[:, idxs]
    seq_dict = {}

    for i, otu in enumerate(otu_li):
        seq_dict[otu] = ''.join(M[i])
    #for keys in seq_dict:
    #    print(keys, seq_dict[keys])
    return seq_dict


def get_sequences_csv(seq_filename):
    """
    selects the aligned sequences of asvs of interest from a csv file

    @Paramters
    ----------------------------------------------------------------------------
    seq_filename : name of the file that contains the sequences

    @returns
    (dict) (str) otu_id -> (str) aligned sequence
    """

    seq_df = pd.read_csv(seq_filename, sep = "\t", index_col = 0)
    index = list(seq_df.index)
    array = seq_df.to_numpy()

    seq_dict = {}
    for i in range(len(index)):
        seq_dict[index[i]] = str(array[i][0])

    return seq_dict

def percent_identity(seq1, seq2):
    """
    computes the percent identity between two sequences
    """
    if len(seq1)  != len(seq2):
        print("Sequences have different length")
        return None
    else:
        n = 0
        for i in range(len(seq1)):
            if seq1[i] == "N" or seq2[i] == "N":
                n += 1
            else:
                if seq1[i] == seq2[i]:
                    n += 1
        return n / len(seq1)

def sanity_check(pi_d, pi_asv_d, names, location):
    """check if the percent identity scores makes sense or not

    @Paramters
    pi_d : (dict) (str) asv_id -> [float] percent identity scores
    pid_asv_id : ([str]) list of valid asvs
    names : (dict) (str) otu_id -> (str) corresponding name
    """
    all_ = ""
    for key in pi_d:
        values = pi_d[key]
        sorted_args = np.argsort(values)
        closest_otus = [pi_asv_d[key][sorted_args[-i]] + " " + names[pi_asv_d[key][sorted_args[-i]]]\
         + " " + str(pi_d[key][sorted_args[-i]]) for i in range(1, 6)]
        all_ = all_ + "(" + key + ")" + " " + names[key] + " : " +\
                " ; ".join(closest_otus) + "\n \n"

    file = open(location + "/sanity_test.txt", "w")
    file.write(all_)
    file.close()

def compute_percent_identity_all(seq_dict):
    """
    computers the percent identity between all the ASVs

    @Parameters
    seq : (dict) (str) otu_name -> (str) sequence

    @returns
    (dict) (str) otu_name -> [float] percent_identity with other ASVS}
    (dict) (str) otu_name -> [str] list of ASV ids
    """

    highest = 0
    lowest = np.inf
    highest_names = ""
    lowest_names = ""
    dict1 = {}
    dict2 = {}
    for k1 in seq_dict:
        dict1[k1] = []
        dict2[k1] = []
        for k2 in seq_dict:
            dict2[k1].append(k2)
            pi = percent_identity(seq_dict[k1], seq_dict[k2])
            if pi > highest and pi != 1:
                highest = pi
                highest_names = k1 + " " + k2
            if pi < lowest:
                lowest = pi
                lowest_names = k1 + " " + k2
            dict1[k1].append(pi)

    return dict1, dict2
