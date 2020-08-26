#This program counts the number of organism types associated with functional unit 
import numpy as np
import pandas as pd
import argparse

def parse():

    parser = argparse.ArgumentParser(description = "Parameters for"\
             "Enrichment Analysis")
    parser.add_argument("-f1", "--file1", required = True)
    parser.add_argument("-f2", "--file2", required = True)
    parser.add_argument("-f3", "--file3", required = True)
    parser.add_argument("-o", "--output_file", required = True)

    return parser.parse_args()

def map_funit_to_ko(file_name):
    """ create a map between (EC/KO) to gene family(Caz/pathway/module)

        @Parameters
        ---------------------------------------------------------------------
        file_name : (str) name of the .txt file
        enable_filter0: (bool) whether or not filter0 is active

        @Returns
        dictionary {key : (str) KM/KP/Caz; value : ([str]) KO/EC}
    """

    dict_ = {}
    file = open(file_name, "r")
    total = 0
    for lines in file:
        line_split = lines.strip().split()
        if len(line_split) > 1:
            group = line_split[0] #KO or # -*- coding: utf-8 -*-
            for i in range(1, len(line_split)):
                func_unit = line_split[i]
                if func_unit not in dict_:
                    dict_[func_unit] = [group]
                else:
                    dict_[func_unit].append(group)

    return dict_

def get_ko_category(kegg_file):
    """ get the category of KO file
        @Prameters
        kegg_file : (str) name of the file containing KOs and their category

        @returns
        dictionary {(str) ko_name : (str) category}
    """

    file = open(kegg_file, "r")
    ko_dict = {}
    for line in file:
        line = line.strip().split()
        ko_dict[line[0]] = line[1]

    return ko_dict

def get_funit_category(funit_dict, category_dict):
    """get the distribution of organism types for KM/KP
       @Pathways
       funit_dict : (dict) (str) KM/KP name -> ([str]) list of KO
       category_dict : (dict) (str) KO -> (str) KO category

       @Returns
       dictionary{(str) KM / KP name -> {dict (str) organism type -> (int) #
          of the organism}}
    """

    dict_ = {}
    for unit in funit_dict:
        unit_dict = {"eu":0, "pro":0, "both":0}
        kos = funit_dict[unit]
        for ko in kos:
            type = category_dict[ko]
            if type == "prokaryote":
                unit_dict["pro"] += 1
            elif type == "eukaryote":
                unit_dict["eu"] += 1
            elif type == "both":
                unit_dict["both"] += 1
            else:
                print("Nothing")
        dict_[unit] = unit_dict

    return dict_

def get_names(file_name):
    """get the names of KEGG module / pathway
       @Parameters
       file_name : name of the CSV file containing the names
       @returns
       dictionary {(str) KM / KP -> (str) corresponding name}
       """

    name_file = pd.read_csv(file_name, sep = ",", header = None).values
    name_dict = {}
    for row in name_file:
        name_dict[row[0]] = row[0] + ", " + row[1].split(",")[0]

    return name_dict

def main():

    args = parse()
    dir = "/Users/sawal386/Desktop/enrichment_/ref_files/common/"
    funit_ko_dict = map_funit_to_ko(dir + args.file1)
    ko_category_d = get_ko_category(dir + args.file2)
    names_d = get_names(dir + args.file3)

    all_ = ""

    funit_category_d = get_funit_category(funit_ko_dict, ko_category_d)
    for funit in funit_category_d:
        elements = funit_category_d[funit]
        el_str = [str(keys) + " : " + str(elements[keys]) for keys in elements]
        all_ = all_ + names_d[funit] + "\t" + ",".join(el_str) + "\n"

    res = open(dir + args.output_file, "w")
    res.write(all_)
    res.close()


main()
