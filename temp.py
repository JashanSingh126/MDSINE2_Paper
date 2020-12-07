import mdsine2 as md2
from mdsine2.names import STRNAMES
import pandas as pd
import logging
import re
import sys
import numpy as np

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

chain2 = 'output/negbin2/replicates/mcmc.pkl'
chain2 = md2.BaseMCMC.load(chain2)
a1_2 = chain2.graph[STRNAMES.NEGBIN_A1].get_trace_from_disk()

chain1 = 'output/negbin/replicates/mcmc.pkl'
chain1 = md2.BaseMCMC.load(chain1)
a1_1 = chain1.graph[STRNAMES.NEGBIN_A1].get_trace_from_disk()

print(np.sum(np.absolute(a1_2-a1_1)))


