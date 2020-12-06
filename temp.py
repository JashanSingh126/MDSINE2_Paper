import mdsine2 as md2
import pandas as pd
import logging
import re
import sys

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


s_pred = 'healthy-cv3-3-start1.5-ndays7.5.npy'
s_full = 'healthy-cv3-3-full.npy'

re_tla_pred = re.compile(r'^(.*)-(.*)-start(.*)-ndays(.*).npy$')
re_full_pred = re.compile(r'^(.*)-(.*)-full.npy$')

print(re_tla_pred.findall(s_pred)[0])
print(re_full_pred.findall(s_full)[0])
