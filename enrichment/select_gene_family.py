#Selects the gene family abundance of OTU that are present in the cluster file . 
#The intial output from picrust contains abundances of 1000 + bugs. But we are
#interested in a select few only (those that are part of the cluster)

import numpy as np
import pandas as pd
import argparse
import copy

def parse():

    parser = argparse.ArgumentParser(description = "Parameters for gene family\
             selection")
    parser.add_argument("-c", "--cluster_file", required = True)
    parser.add_argument("-ct", "--cluster_type", required = True)
    parser.add_argument("-pt", "--predicted_table", required = True)
    parser.add_argument("-o", "--output_file", required = True)
    parser.add_argument("-p", "--path", required = True)

    return parser.parse_args()

def obtain_clusters(file_name):
    """ parses the .txt file containing the cluster information and
         returns a dictionary with cluster number as the key and
         the OTUs in the cluster as the values

        @Parameters
        ------------------------------------------------------------------------
        file_name: name of the file containing the cluster information
    """
    file = open(file_name, "r")
    c_size = 0
    c_name = 0
    c_dict = {}
    c_sizes ={}
    names = []
    for line in file:
        #cluster details
        if "Cluster" in line:
            c_size = int(line.split()[-1])
            c_name += 1
            c_sizes[c_name] = c_size
            c_dict[c_name] = []
        else:
            #OTU's in the cluster
            c_dict[c_name].append(line.split()[2].strip("()"))
            names.append(line.split()[2].strip("()"))
    return names

def main():

    args = parse()
    loc_sp = args.path + "/ref_files/" + args.cluster_type + "/"
    loc_cm = args.path + "/ref_files/common/"
    otus = obtain_clusters(loc_sp + args.cluster_file)
    #print("OTU:", otus)
    x = set(otus)
    #print(len(x))
    #print(len(otus))

    table_file = loc_cm + args.predicted_table
    whole_data = pd.read_csv(table_file, sep = "\t")
    header = whole_data.columns
    new_data = []
    new_data.append(header)

    done = []
    count = 0
    values = whole_data.values
    for i in range(len(values)):
        name = values[i][0]
        #print(name)
        if name in otus:
            new_data.append(values[i])
            done.append(name)
        count += 1
    #print(new_data)
    pd.DataFrame(new_data).to_csv(loc_sp + args.output_file + ".tsv", sep = "\t",
                index = None, header = None)


main()
