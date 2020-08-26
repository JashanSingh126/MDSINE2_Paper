#This program contains functions that creates a map(dictionary) between two
#biological entities

import numpy as np
import pandas as pd

def update_dict(d, key, value):
    """updates the dictionary by adding value to the list mapped to the key

       @Parameters
       d : (dictionary)
       key : (string/ int)
       value: (string/ int)

       @returns
       None
    """

    if key not in d:
        d[key] = [value]
    else:
        if value not in d[key]:
            d[key].append(value)

def run_filter0(unit, unit_category_dict):
    """checks whether or not a KM/KP passeses filter0

        @Parameters
        unit : (str) name of the functional unit
        unit_category_dict : (dict of dict)  (int) count of (str) organism types
                             associated with the (str) functional unit

        @returns
        bool
    """
    unit_info = unit_category_dict[unit]
    #print(unit, unit_info)
    if unit_info["eu"] != 0:
        return False
    #if unit_info["eu"] >= unit_info["both"] + unit_info["pro"] :
    #    return False
    else:
        return True

def map_funit_to_gene_family(file_name, enable_filter0, *args):
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
    observed = []
    repeated = []
    passed = []
    failed = []
    print("Mapping Functional Unit to Gene Family Group")
    for lines in file:
        line_split = lines.strip().split()
        if len(line_split) > 1:
            group = line_split[0] #KO or # -*- coding: utf-8 -*-
            if group not in observed:
                observed.append(group)
            else:
                repeated.append(group)
            for i in range(1, len(line_split)):
                func_unit = line_split[i]
                if not enable_filter0:
                    if group in args[1]:
                        update_dict(dict_, func_unit, group)
                    #else:
                    #    print(group)
                #check if filter 0 is passed i.e. the
                else:
                    passed_filter0 = run_filter0(func_unit, args[0])
                    if passed_filter0 and group in args[1]:
                        #print(func_unit, args[0][func_unit])
                        update_dict(dict_, func_unit, group)
                        if func_unit not in passed:
                            passed.append(func_unit)
                    else:
                        if func_unit not in failed:
                            failed.append(func_unit)

    return dict_

def map_gene_group_to_otu(tsv_file):

    """map gene family group to OTU
       @Parameters
       --------------------------------------------------------------------
       tsv_file: (str) name of tsv file containing the OTUs and the corresponding
                 gene family groups

        @Returns
        dictionary {(str) KO/EC -> ([str]) list of associated OTUs}
    """
    print("Mapping gene family to OTUs")
    whole_table = pd.read_csv(tsv_file, sep = "\t")
    group_names = whole_table.columns[1: ]
    values = whole_table.values[:, 1:].T
    otu_names = whole_table.values[:, 0]
    group_dict = {}
    for i in range(len(group_names)):
        name = group_names[i]
        group_dict[name] = []
        for j in range(len(values[i])):
            if values[i][j] != 0:
                if otu_names[j] not in group_dict[name]:
                    group_dict[name].append(otu_names[j])
    final_ = {}
    for keys in group_dict:
        if len(group_dict[keys]) != 0:
            final_[keys] = group_dict[keys]
    return final_
    #return group_dict

def map_funit_to_otu(funit_to_group, group_to_otu):

    """map cazy or KEGG module/pathway to OTUs
       @Parameters
       --------------------------------------------------------------------
       funit_to_group : (dict) (str) KM/KP/Caz -> ([str])list of KO/EC
       group_to_otu : (dict) (str) KO/EC -> ([str])list of OTUs

       @returns
       dictionary {(str) KM/KP/Caz -> ([str]) list of OTUs}
    """
    print("Mapping Functional units to OTU")
    n_invalid = 0
    dict_ = {}
    for funit in funit_to_group:
        dict_[funit] = []
        for group in funit_to_group[funit]:
            if group in group_to_otu:
                for otu in group_to_otu[group]:
                    if otu not in dict_[funit]:
                        dict_[funit].append(otu)

            #gene family not present in output from picrust 2
            #print(group + " not found in gene family")
    #for keys in dict_:
    #    print(keys, len(dict_[keys]))
    #print(len(dict_))
    return dict_

def map_ec_to_ko(ko_ec_filename):
    """
       maps EC(key) to associated KOs (value)
       @Parameters
       ko_ec_filename : a txt file containing KO and associated ECs

       @returns
       dictionary {(str) ec -> ([str] associated KOs)}
    """

    ec_dict = {}
    print("mapping EC")
    file = open(ko_ec_filename, "r")
    for lines in file:
        line_li = lines.strip().split()
        if len(line_li) > 1:
            value = line_li[0]
            for i in range(1, len(line_li)):
                key = "EC:" + line_li[i]
                if key not in ec_dict:
                    ec_dict[key] = [value]
                else:
                    ec_dict[key].append(value)
    return ec_dict

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

def map_caz_to_ko(caz_dict, ec_ko_dict, enable_filter4, *args):
    """map Cazymes(key) to KOs(values)

       @Parameters
       caz_dict : (dict) mapping (str)cazymes to ([str]) list of EC
       ec_ko_dict : (dict) mapping (str) ec to ([str])list of KOs
       enable_filter4 : (bool)
       @Returns
       dictionary {(str) cazyme name -> ([str]) list of KO}
    """
    
    dict_ = {}
    type_d = {}
    if enable_filter4:
        type_d = get_ko_category(args[0])
    #print("type:", type_d)
    print("filtrer 4:", enable_filter4)
    for caz in caz_dict:
        #print("mod", mod)
        dict_[caz] = []
        for ec in caz_dict[caz]:
            #print("gene_fam", gene_fam)
            if ec in ec_ko_dict:
                for ko in ec_ko_dict[ec]:
                    if enable_filter4:
                        #print(type_d[ko])
                        if type_d[ko] == "prokaryote":
                            if ko not in dict_[caz]:
                                dict_[caz].append(ko)
                    else:
                        if ko not in dict_[caz]:
                            dict_[caz].append(ko)

    return dict_
