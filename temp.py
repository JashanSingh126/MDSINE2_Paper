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

# dset = md2.Study.load('processed_data/gibson_healthy_agg_taxa_filtered.pkl')

# to_delete = []
# for taxa in dset.taxas:
#     if taxa.idx > 10:
#         to_delete.append(taxa.name)
# dset.pop_taxas(to_delete)


# reads = pd.read_csv('bindertutorials/data/healthy/reads.tsv', sep='\t')

# reads = reads.set_index('name')
# for col in reads.columns:
#     print(col)

# print(reads.head())


dset = md2.dataset.parse(name='example-dataset', 
    taxonomy='bindertutorials/data/healthy/taxonomy.tsv', 
    reads='bindertutorials/data/healthy/reads.tsv', 
    qpcr='bindertutorials/data/healthy/qpcr.tsv', 
    perturbations='bindertutorials/data/healthy/perturbations.tsv', 
    metadata='bindertutorials/data/healthy/metadata.tsv')


