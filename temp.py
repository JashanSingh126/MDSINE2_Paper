import mdsine2 as md2
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


pred_fname = 'output/mdsine2/cv/forward_sims/healthy-cv2-validate-2-start1.0-ndays1.0.npy'
truth_fname = 'output/mdsine2/cv/forward_sims/healthy-cv2-validate-2-start1.0-ndays1.0-truth.npy'
times_fname = 'output/mdsine2/cv/forward_sims/healthy-cv2-validate-2-start1.0-ndays1.0-times.npy'

pred = np.load(pred_fname)
truth = np.load(truth_fname)
times = np.load(times_fname)

print(pred.shape)
print(truth.shape)

# print(pred[0])
print(times)
print(np.all(np.isnan(pred)))