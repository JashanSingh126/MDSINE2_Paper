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



#  rows, taxa fmt
# columns are the clusters

# Get the objects
mcmc = md2.BaseMCMC.load('output/mdsine2/cv/healthy-cv2/mcmc.pkl')
clustering = mcmc.graph[STRNAMES.CLUSTERING_OBJ]
study = mcmc.graph.data.subjects

# Get all the times
times = study.times(agg='union')

# Get the first time of the perturbation between all of the subjects
first_time = None
for subj in study:
    if first_time is None:
        first_time = study.perturbations[0].starts[subj.name]
    else:
        curr_time = study.perturbations[0].starts[subj.name]
        if curr_time < first_time:
            first_time = curr_time
t_end = np.searchsorted(times, first_time)

# Get mean abundance over subjects for times preperturbation then take mean
times = times[:t_end]
M = study.matrix(dtype='abs', agg='mean', times=times)
abunds = np.mean(M, axis=1)

# Get the clustering
cidx_assign = md2.generate_cluster_assignments_posthoc(clustering=clustering)
clustering.from_array(cidx_assign)

# Get the average abundances for each OTU in each cluster
data = []
for cluster in clustering:
    oidxs = list(cluster.members)
    temp = np.zeros(len(abunds))
    temp[oidxs] = abunds[oidxs]
    data.append(temp.reshape(-1,1))
data = np.hstack(data).shape

# Make the taxonomic heatmap as a dataframe
fmt = '%(family)s'
df = md2.condense_matrix_with_taxonomy(M, taxas=study.taxas, fmt=fmt)
# rows are the family, columns are the clusters


