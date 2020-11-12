#convert the hairball network to a JSON file, which can be opened in cytoscape 

import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patch
import igraph
from itertools import permutations
import matplotlib.gridspec as gridspec
from textwrap import wrap
import json
from py2cytoscape.util import from_networkx
import argparse 

def parse_arguments():

    parser = argparse.ArgumentParser(description = "Files needed for visuaizing\
        cluster interaction")
    parser.add_argument("-c", "--cluster_file", required = True, 
        help = ".txt file containing the cluster composition")
    parser.add_argument("-b", "--bayes_factor", required = True,
        help = "a tsv file containing the Bayes Factors between the clusters")
    parser.add_argument("-i", "--interaction", required = True, 
        help = "a tsv file containing the sign of interaction between the clusters")
    parser.add_argument("-o", "--output_file", required = True, 
        help = "name of the output file")

    return parser.parse_args()

def parse_cluster(file_name):
    """ parses the .txt file containing the cluster

        @Parameters
        ------------------------------------------------------------------------
        file_name: name of the file containing the cluster information

        @returns
        ------------------------------------------------------------------------
        dict {(int) cluster_id -> ([str]) ASV numbers of OTUs in the cluster}
    """

    file = open(file_name, "r")
    c_size = 0
    c_name = 0
    c_dict = {}
    c_sizes ={}
    c_otu_names = {}
    count = 0
    for line in file:
        #cluster details
        if "Cluster" in line:
            #c_size = int(line.split()[-1])
            c_name += 1
            #c_sizes[c_name] = c_size
            c_dict[c_name] = []

        else:
            if "ASV" in line:
                c_dict[c_name].append(line.strip())

    #print(len(c_otu_names))
    return c_dict

def get_largest_weight(G):
    """return the largest edge weight

       @parameters 
       G : a graph object 

       @returns 
       (int)
    """

    largest = 0
    for edge in G.edges(data = True):
        w = edge[2]["weight"]
        if not np.isinf(w):
            if w > largest:
                largest = w

    return w

def get_bayes_category(bf):
    """classify bayes factor according to strength of evidence

    @parameters
    bf : (float) bayes factor 

    @returns 
    (int)
    """

    category = 0
    if bf > 10 ** 2 :
        category = 3
    elif 10 < bf <= 10 ** 2:
        category = 2
    elif 10 ** 0.5 < bf <= 10:
        category = 1
    elif 0 < bf <= 10 ** 0.5:
        category = 0

    return category

def main():

    args = parse_arguments()
    cluster_d = parse_cluster(args.cluster_file)
    print(cluster_d)
    bayes_factor_data = pd.read_csv(args.bayes_factor, sep = "\t", 
        index_col = 0).to_numpy()
    print(bayes_factor_data.shape)
    interaction_data = pd.read_csv(args.interaction, sep = "\t", 
        index_col = 0).to_numpy()
    x = [asv for id in cluster_d for asv in cluster_d[id]]

    columns = [i + 1 for i in range(0, bayes_factor_data.shape[0])]
    bayes_df = pd.DataFrame(bayes_factor_data.T, columns = columns, index = columns)
    graph_bayes = nx.from_pandas_adjacency(bayes_df, create_using = nx.DiGraph())

    largest = get_largest_weight(graph_bayes)
    all_edges = graph_bayes.edges()
    edge_attributes = {}

    for edge in graph_bayes.edges(data = True):
        
        int_strength = interaction_data[edge[1] -1 , edge[0] - 1]
        print(edge, int_strength)
        coord = (edge[0], edge[1])
        sign = 0 
        weight = edge[2]["weight"]
        if np.isinf(weight):
            weight = largest 

        if int_strength < 0:
            sign = -1 
        else:
            sign = 1

        category = get_bayes_category(weight)
        bend = False
        if (edge[1], edge[0]) in all_edges:
            bend = True

        edge_attributes[coord] = {"bayes_fac" : category, "sign" : sign, "weight" :
                                 weight, "bend" : bend}

    nx.set_edge_attributes(graph_bayes, edge_attributes)
    edges_li = graph_bayes.edges.data()

    nodes_attributes = {}
    for keys in cluster_d:
        print(keys)
        nodes_attributes[keys] = {"size" : len(cluster_d[keys])}

    nx.set_node_attributes(graph_bayes, nodes_attributes)
    data_json = from_networkx(graph_bayes)
    filename = args.output_file + ".json"
    print(data_json)
    with open(filename, "w") as f:
        json.dump(data_json, f)
  
if __name__ == "__main__":
    main()

