#contains functions to process the deseq results (csv format)
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
from matplotlib import rcParams
from matplotlib import font_manager


def get_deseq_info_phylum(loc, donor):

    hfd_names = set(pd.read_csv("{}/{}1.csv".format(loc, donor), index_col=0).index)
    vanc_names = set(pd.read_csv("{}/{}2.csv".format(loc, donor), index_col=0).index)
    gent_names = set(pd.read_csv("{}/{}3.csv".format(loc, donor), index_col=0).index)

    names_union = hfd_names.union(vanc_names.union(gent_names))
    #print(names_union)
    #print(len(names_union))
    cleaned_names = []
    for name in names_union:
        if "unknown" in name:
            if name !="unknown unknown":
                cleaned_names.append(name.replace("unknown", "NA"))
        else:
            cleaned_names.append(name)

    return set(cleaned_names)

def get_deseq_info_order(loc, donor):
    def get_names(df):
        names = []
        index = df.index
        np_df = df.to_numpy()
        i = 0
        for row in np_df:
            if row[-1] <= 0.05:
                names.append(index[i])
            i += 1
        return set(names)

    #hfd_names = set(pd.read_csv("{}/{}1.csv".format(loc, donor), index_col=0).index)
    #vanc_names = set(pd.read_csv("{}/{}2.csv".format(loc, donor), index_col=0).index)
    #gent_names = set(pd.read_csv("{}/{}3.csv".format(loc, donor), index_col=0).index)

    hfd_names = get_names(pd.read_csv("{}/{}1.csv".format(loc, donor), index_col=0))
    vanc_names = get_names(pd.read_csv("{}/{}2.csv".format(loc, donor), index_col=0))
    gent_names = get_names(pd.read_csv("{}/{}3.csv".format(loc, donor), index_col=0))
    names_union = hfd_names.union(vanc_names.union(gent_names))

    cleaned_names = []
    names_dict = {}
    i = 0
    for name in names_union:
        name_split = name.split()
        order_family = " ".join(name_split[2:])

        if "unknown" in order_family:
            if order_family !="unknown unknown":
                cleaned_names.append(order_family.replace("unknown", "NA"))
            else:
                cleaned_names.append("{},phylum".format(name_split[0]) + " " +
                    "Phylum")
        else:
            cleaned_names.append(order_family)
        names_dict[cleaned_names[-1]] = name
        i += 1

    return set(cleaned_names), names_dict
