import mdsine2 as md2
from mdsine2.names import STRNAMES
import pandas as pd
import logging
import re
import sys
import numpy as np
import matplotlib.pyplot as plt

md2.LoggingConfig(level=logging.INFO)
# study = md2.Study.load('processed_data/gibson_healthy_agg.pkl')

# for otu in study.taxas:

#     if md2.isotu(otu):
#         print(otu.name)
#         data = []
#         for asvname in otu.aggregated_taxas:
#             temp = []
#             for k in otu.aggregated_taxonomies[asvname]:
#                 temp.append(otu.aggregated_taxonomies[asvname][k])
#             data.append(temp)

#         df = pd.DataFrame(data, columns=list(otu.aggregated_taxonomies[asvname].keys()))
#         print(df)
#         print(list(otu.taxonomy.values()))

# study.taxas.generate_consensus_taxonomies()



#  rows, taxa fmt
# columns are the clusters

from gibson_dataset.scripts.util import load_gibson_dataset

study = load_gibson_dataset(dset='healthy')

# study = md2.Study.load('processed_data/gibson_healthy_agg.pkl')

for perturbation in study.perturbations:
    print(perturbation)