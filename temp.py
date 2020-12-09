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

study = md2.Study.load('processed_data/gibson_healthy_agg.pkl')

md2.visualization.taxonomic_distribution_over_time(study['2'], 
    taxlevel='family', label_formatter='%(family)s')
plt.show()