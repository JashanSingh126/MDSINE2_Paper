import numpy as np
import pandas as pd
import argparse
import copy
from scipy.special import comb
from scipy.stats import hypergeom
from statsmodels.stats import multitest as mtest
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib.colors as colors
from matplotlib.colors import LogNorm
import os
import pickle

import form_maps as mp
import make_plots as plot

def parse():
    """parse the command line arguments"""

    parser = argparse.ArgumentParser(description = "Parameters for"\
             "Enrichment Analysis")
    parser.add_argument("-l", "--loc", required = True,
        help = "<Required> directory of the supporting files")
    parser.add_argument("-ct", "--cluster_type", required = True,
          help = "<Required> source of data (healthy, unhealthy)")
    parser.add_argument('-c', "--cluster_file", required = True,
           help = "<Required> a .txt file containing cluster information")
    parser.add_argument("-m", "--group_funit_filename", required = True,
           help = "<Required> a .txt file containing the list of KO/EC\
           and associated modules, pathways, or Cazymes ")
    parser.add_argument("-t", "--group_table", required = True,
           help = "<Required> a tsv file containing the KO/ECs associated with \
           ASVs")
    parser.add_argument("-p", "--perturbation_file", help = "<Optional> a pkl"\
           "file that describes the pertubation effect in each cluster")
    parser.add_argument("-fc", "--funit_category", help = "<Optional> file\
           that contains the distribution of organism in a functional unit")
    parser.add_argument("-kec", "--map_ko_ec", help = "<Optional> a txt file \
           where each line contains KO and corresponding set of ECs")
    parser.add_argument("-a", "--abundance_file", help = "<Optional> a pkl file \
            that contains the abundance of species in each cluster")
    parser.add_argument("-n", "--names", help = "<Optional> a csv \
            that contains the abundance of phyla in each cluster")
    parser.add_argument("-kt", "--ko_type", help = "<Required> a text file\
           that includes the KO type")
    parser.add_argument("-f0", "--filter0", default = False, type = bool,
            help = "<Optional> Remove KOS that are predominantly Eukaryotic")
    parser.add_argument("-f1", "--filter1", type = float, required = True,
            help = "<Required> min number of OTUs in a cluster that must be annotated\
             to module/pathway of interest")
    parser.add_argument("-f2", "--filter2", type = int, required = True,
            help = "min number of OTUs in a cluster that must be annotated\
             to module/pathway of interest")
    parser.add_argument("-f3", "--filter3", type = int, required = True,
            help = "minimum number of KOs that must be associated with a functional\
             unit to be considered valid for enrichment")
    parser.add_argument("-f4", "--filter4", type = bool, default = False,
            help = "<Required> percentage")
    parser.add_argument("-fr", "--save_name", required = True,
            help = "<Required> the name of the file that holds the results")
    parser.add_argument("-ti", "--title", required = True,
            help = "<Required> the title of the plot")
    parser.add_argument("-ty", "--type", required = True,
            help = "<Required> the type of enrichment analysis")
    parser.add_argument("-le", "--level", required = True,
            help = "<Required> hierarchy level for abundance maps")

    return parser.parse_args()

def parse_cluster(file_name):
    """ parses the .txt file containing the cluster information and
         returns a dictionary with cluster number as the key and
         the OTUs in the cluster as the values

        @Parameters
        ------------------------------------------------------------------------
        file_name: name of the file containing the cluster information

        @returns 
        ------------------------------------------------------------------------
        dict {(int) cluster_id -> ([str]) ASV numbers of OTUs in the cluster}
        dict {(int) cluster_id -> ([str]) names of OTUs in the cluster}
    """

    file = open(file_name, "r")
    c_size = 0
    c_name = 0
    c_dict = {}
    c_sizes ={}
    c_otu_names = {}

    for line in file:
        #cluster details
        if "Cluster" in line:
            c_size = int(line.split()[-1])
            c_name += 1
            c_sizes[c_name] = c_size
            c_dict[c_name] = []
            c_otu_names[c_name] = []
        else:
            #OTU's in the cluster
            #print(line.split()[-1])
            c_dict[c_name].append(line.split()[2].strip("()"))
            c_otu_names[c_name].append(line.split()[-2] + " " + line.split()[-1])

    print("Checking cluster size.............................")
    if len(c_dict) != len(c_sizes):
        print("Error")
    else:
        correct= True
        for keys in c_dict:
            if len(c_dict[keys]) != c_sizes[keys]:
                correct = correct and False
        if correct:
            print("Clusters correctly parsed")
        else:
            print("Error in Cluster parsing")

    return c_dict, c_otu_names


def compute_p_value(N, M, n, k):
    '''computes and returns the hypergeometric p-values
       @Parameters
       ------------------------------------------------------------------
       N, M, n, k: (int)

       @returns
        ------------------------------------------------------------------------
       float
    '''
    lim = min(n, M)
    p = 0
    for i in range(k, lim + 1):
        p += comb(M, i) * comb(N - M, n - i) / comb(N, n)

    p1 = hypergeom.sf(k-1, N, M, n) #to check the accuracy of p

    return p

def obtain_p_vals(cluster_dict, funit_otu_dict, funit_group_dict, thresh,
                  min_otu, n_otu, min_ko):
    """performs enrichment analysis per module for each cluster

       @Parameters
       ---------------------------------------------------------------------
       cluster_dict : (str) cluster id ->([str]) OTUs
       funit_otu_dict :  (str) functional unit -> ([str]) OTUs
       funit_group_dict: (str) functional unit -> ([str]) KO's
       n_otu: (int) total number of OTUs
       thresh : (float) upper limit on the number of OTU associated with a module
       min_otu: (int) min number of annotated otu in a cluster
       min_ko : (int) min number of KOs related to a functional unit

       @returns
        ------------------------------------------------------------------------
       dict {(int) cluster_id -> ([float]) enriched p_values}
       dict {(int) cluster_id -> ([str]) names of enriched functional units}
       dict {(int) cluster_id -> ([float]) all p_values}
       dict {(int) cluster_id -> ([str]) names of all functional units}
    """

    print("Computing cluster wise p-values")
    cluster_enriched_p = {}
    cluster_enriched_funit = {}
    cluster_all_p = {}
    cluster_all_funit = {}
    max_n_funit = thresh  * n_otu
    print("threshold:", max_n_funit)

    for id in cluster_dict:
        print("Cluster ID:", id, "size:", len(cluster_dict[id]))
        n_otu_cluster = len(cluster_dict[id])
        cluster_enriched_p[id] = []
        cluster_enriched_funit[id] = []
        cluster_all_p[id] = []
        cluster_all_funit[id] = []

        for funit in funit_otu_dict:

            if len(funit_otu_dict[funit]) != 0:
                #ensure #otu is less than the threshold and KO size is greater than 0
                #print(funit, "size_funit:", len(funit_otu_dict[funit]), "size_ko:",
                #en(funit_group_dict[funit]))

                if len(funit_otu_dict[funit]) < max_n_funit and len(funit_group_dict[funit]) > min_ko:
                    n_otu_annotated = len(funit_otu_dict[funit])
                    n_otu_cluster_annotated = 0
                    for otu in cluster_dict[id]:
                        if otu in funit_otu_dict[funit]:
                            n_otu_cluster_annotated += 1
                    p_val = compute_p_value(n_otu, n_otu_annotated,
                                 n_otu_cluster, n_otu_cluster_annotated)
                    print(funit, "p:", p_val, "cluster #:", id, "N:",n_otu, "M:",
                    n_otu_annotated, "n:",  n_otu_cluster, "k:", n_otu_cluster_annotated)
   
                    cluster_all_p[id].append(p_val)
                    cluster_all_funit[id].append(funit)
                    if n_otu_cluster_annotated > min_otu:
                        cluster_enriched_p[id].append(p_val)
                        cluster_enriched_funit[id].append(funit)
    return cluster_enriched_p, cluster_enriched_funit, cluster_all_p, cluster_all_funit

def get_names(loc, type):

    """ get the names of the pathway or module or cazyme
    @Parameters
    ------------------------------------------------------------------------
    file_name : (str) a csv file containing the code and the corresponding name

    @returns
    ------------------------------------------------------------------------
    dictionary {(str) KM/KP/Cazy -> (str) name }
    """
    file_name = ""
    if type == "kegg_modules":
        file_name = loc + "module_names.csv"
    elif type == "kegg_pathways":
        file_name = loc + "pathway_names.csv"
    elif type == "cazymes":
        file_name = loc + "cazyme_names.csv"

    name_file = pd.read_csv(file_name, sep = ",", header = None).values
    name_dict = {}
    for row in name_file:
        name_dict[row[0]] = row[0] + ", " + row[1].split(",")[0]

    return name_dict

def export_results(corr_p_vals, p_val, funits, names, type, f_name):
    """save the results in  a txt file

       @Parameters
        ------------------------------------------------------------------------
       corr_p_vals : ([[bool], [float]]) enrichment results and corrected p_vals
       p_vals : ([float]) non-corrected p_values
       funits: ([str])names functional units
       type: (str) the type of enrichment analysis
       f_name: (str) name of the file that is exported
    """

    enriched_funits = []
    enriched_p_values = []
    enriched_adjusted_p = []

    for j in range(len(corr_p_vals[0])):
        if corr_p_vals[0][j] == True:
            enriched_funits.append(funits[j])
            enriched_p_values.append(p_val[j])
            enriched_adjusted_p.append(corr_p_vals[1][j])

    enriched_cluster_module = {}
    enriched_p = {}
    enriched_adj_p = {}

    for i in range(len(enriched_funits)):
        info = enriched_funits[i].split()
        cluster_id = info[1]
        if cluster_id not in enriched_cluster_module:
            enriched_cluster_module[cluster_id] = []
        if cluster_id not in enriched_p:
            enriched_p[cluster_id] = []
        if cluster_id not in enriched_adj_p:
            enriched_adj_p[cluster_id] = []
        name = info[2]
        enriched_cluster_module[cluster_id].append(names[name])
        enriched_p[cluster_id].append(enriched_p_values[i])
        enriched_adj_p[cluster_id].append(enriched_adjusted_p[i])

    all_lines = "Total Enrichment: " + str(len(enriched_funits)) + "\n"
    for keys in enriched_cluster_module:
        all_lines = all_lines + "Cluster " + keys + "\n"
        all_lines = all_lines + "module name" + "\t" + "p value" + "\t" + "adjusted_p" + "\n"
        n = len(enriched_cluster_module[keys])
        for i in range(n):
            name = enriched_cluster_module[keys][i]
            p1 = enriched_p[keys][i]
            p2 = enriched_adj_p[keys][i]
            all_lines += name + "\t" + str(p1) + "\t" + str(p2) + "\n"
        all_lines = all_lines + "\n"
    results = open(f_name +  "_results.txt", "w")
    results.write(all_lines)

def pivot_df(values_dict, funit_dict, enriched_res, names, f_name):
    """arrange the data in format that's best for plotting

       @Parameters
        ------------------------------------------------------------------------
       values_dict : (dict) map from (str) cluster_id to ([float]) p-values
       funit_dict : (dict) map from (str) cluster_id to ([str]) KM/KP
       enriched_res : (list of lists) results containing ([bool]) enrichment
                    results and ([float]) adjusted p-values
       names : (dict) (str) KM / KP -> (str) corresponding name
       f_name : (str) name of the pickle file

       @Returns
        ------------------------------------------------------------------------
       pandas dataframe
    """

    cluster_row = []
    module_names = []
    values = []
    enriched = []
    count = 0
    true_count = 0
    print("Pivoting df")
    for keys in funit_dict:
        for i in range(len(funit_dict[keys])):
            #print(enriched_res[1][count], keys, names[funit_dict[keys][i]], values_dict[keys][i])
            if enriched_res[1][count] < 0.05:
                #print("count:", count, "corrected_p:", enriched_res[1][count],
                #"uncorrected_p:", values_dict[keys][i], "enriched:",
                #enriched_res[0][count], "name:", funit_dict[keys][i])
                #print(funit_dict[keys][i], names[funit_dict[keys][i]])
                module_names.append(names[funit_dict[keys][i]])
                values.append(enriched_res[1][count])
                cluster_row.append("Cluster " + str(keys))
                #enriched.append(enriched_res[count])
            count += 1
    data_frame = pd.DataFrame({"cluster_id":cluster_row,
            "module_name":module_names, "p_value":values})
    df_pivot = data_frame.pivot(values = "p_value", index = "module_name",
                columns = "cluster_id")
    df_pivot = df_pivot.fillna(1)
    df_pivot.to_pickle(f_name + ".pkl")
    ##print()

    return df_pivot


def get_categories(category_txt):
    """parse the category count of each ko and return the count as
       a dictionary

       @Parameters
        ------------------------------------------------------------------------
       category_txt : a txt file containing the numbers of prokaryotic,
                      eukaryotic or both KOs

        @Returns
        ------------------------------------------------------------------------
        {dictionary (str) KO -> {(str) organism type -> (int) # of organism}}
    """
    if category_txt is not None:
        file = open(category_txt, "r")
        dict_all = {}
        for line in file:
            line_sp = line.strip().split("\t")
            line_dict = {}
            further_split = line_sp[1].split(",")
            for i in range(len(further_split)):
                cate = further_split[i].split(":")
                line_dict[cate[0].strip()] = int(cate[1])
                dict_all[line_sp[0].split(",")[0]] = line_dict
        return dict_all
    else:
        return {}

def classify_otu(pk_file, cluster):

    class_d = pickle.load(open(pk_file, "rb"))
    cluster_class_d = {}
    gram_positive = class_d["gram-positive"]
    gram_negative = class_d["gram-negative"]
    for id in cluster:
        sub_dict = {"gp" : 0, "gn": 0, "nan":0}
        for otu in cluster[id]:
            if otu.split()[0] in gram_positive:
                #print(otu, "gram_positive")
                sub_dict["gp"] += 1
            elif otu.split()[0] in gram_negative:
                #print(otu, "gram_negative")
                sub_dict["gn"] += 1
            else:
                #print(otu, "nan")
                sub_dict["nan"] += 1
        cluster_class_d[id] = sub_dict
    return cluster_class_d

def main(**kwargs):

    args = parse()
    dir_cm = args.loc + "/ref_files/common/"
    dir_sp = args.loc + "/ref_files/" + args.cluster_type + "/"
    cluster_d, cluster_w_names = parse_cluster(dir_sp + args.cluster_file)
    genus_class_file = dir_cm + "genus_classification.pickle"
    family_class_file = dir_cm + "family_classification.pickle"
    genus_classification = classify_otu(genus_class_file, cluster_w_names)

    print("total clusters:", len(cluster_d))
    print()

    categories = {}
    if args.filter0 and args.type != "cazymes":
        categories = get_categories(dir_cm + args.funit_category)

    #print(categories)
    gene_otu_d = mp.map_gene_group_to_otu(dir_sp + args.group_table)
    #print(gene_otu_d.keys())
    funit_group_d = mp.map_funit_to_gene_family(dir_cm + args.group_funit_filename,
                    args.filter0, categories, gene_otu_d)
    funit_otu_d = mp.map_funit_to_otu(funit_group_d, gene_otu_d)
    t = 0

    n_otu = 0
    for key in cluster_d:
        n_otu += len(cluster_d[key])

    filter2_dict = funit_group_d

    valid_p, valid_funits, all_p, all_funits = obtain_p_vals(cluster_d,
    funit_otu_d, filter2_dict, args.filter1, args.filter2 , n_otu, args.filter3)


    all_valid_p = [] #gather all p values in a single list
    all_valid_funits = [] #corresponding functional units
    for keys in valid_p:
        all_valid_p = all_valid_p + valid_p[keys]
        modules_cluster = []
        for k in valid_funits[keys]:
            modules_cluster.append("Cluster " + str(keys) + " " + k)
        all_valid_funits = all_valid_funits + modules_cluster

    adjusted_p = []

    if len(all_valid_p) != 0:
        adjusted_p = mtest.multipletests(copy.deepcopy(all_valid_p),
        alpha = 0.05, method = "fdr_bh", is_sorted = False)
    else:
        print("corrected p is of size 0")

    funit_names = get_names(dir_cm, args.type) #names of functional units
    pivoted_df = pivot_df(valid_p, valid_funits, adjusted_p, funit_names, 
                          args.save_name)

    x_lab = "Cluster ID"
    y_lab = ""

    if args.type == "kegg_pathways":
        y_lab = "KEGG Pathways"
    elif args.type == "kegg_modules":
        y_lab = "KEGG Modules"
    elif args.type == "cazymes":
        y_lab = 'Cazymes'

    location = "{0}/results/{1}/{2}/".format(args.loc, args.type,
                 args.cluster_type) #location for saving results
    path_exists = os.path.exists(location)
    if not path_exists:
        path1 = "{0}/results".format(args.loc)
        if not os.path.exists(path1):
            print("Making path:", path1)
            os.mkdir(path1)
        path2 = "{0}/results/{1}/".format(args.loc, args.type)
        if not os.path.exists(path2):
            print("Making path:", path2)
            os.mkdir(path2)
        path3 = "{0}/results/{1}/{2}/".format(args.loc, args.type,
                     args.cluster_type)
        if not os.path.exists(path3):
            print("Making path:", path3)
            os.mkdir(path3)

    full_title = args.title + " for " + args.type + " " + args.cluster_type
    figname = location + args.save_name + "_" + args.cluster_type + "_" + args.type
    #produce_plot1(pivoted_df, x_lab, y_lab, full_title,
    #              figname + "_heatmap")
    export_results(adjusted_p, all_valid_p, all_valid_funits, funit_names,
                  args.type,  figname)

    plot.produce_plot2(pivoted_df, dir_sp + args.perturbation_file, dir_sp + 
        args.abundance_file, args.level, x_lab, y_lab, full_title,
        args.type, len(cluster_d), figname, genus_classification)

    #return adjusted_p[0]

if __name__ == "__main__":
    main()

