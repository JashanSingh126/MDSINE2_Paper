import logging
import numpy as np
import sys
import os
import pickle
import copy
import random
import pandas as pd
import argparse

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats

import pylab as pl

import config

sys.path.append('..')
import synthetic
import main_base
from names import STRNAMES
import preprocess_filtering as filtering
import model
import data


basepath = 'icml20_filtering/'
nt = 65
p = 0.05
m = 0.4
nr = 5
n_data_seeds = 10
nb = 4000
ns = 10000
perts = 'addit'

graphpath = 'graph_ds{}_is0_b{}_ns{}_coTrue_perts{}_nr{}_no20_nd65_ms{}_pv{}_ma1.00E+08_np5_nt{}'.format

# Do coclusters
fig = plt.figure(figsize=(50,20))
for d in range(n_data_seeds):
    print(d)
    mcmcpath = basepath + graphpath(d, nb, ns, perts, nr, m, p, nt) + '/mcmc.pkl'
    try:
        mcmc = pl.inference.BaseMCMC.load(mcmcpath)
    except:
        continue
    CLUSTERING = mcmc.graph[STRNAMES.CLUSTERING_OBJ]
    
    # Coclustering
    cocluster_trace = CLUSTERING.coclusters.get_trace_from_disk(section='posterior')
    coclusters = pl.variables.summary(cocluster_trace, section='posterior')['mean']
    for i in range(coclusters.shape[0]):
        coclusters[i,i] = 1
    pl.visualization.render_cocluster_proportions(ax=fig.add_subplot(2,5,d+1),
        coclusters=coclusters, asvs=mcmc.graph.data.asvs, clustering=CLUSTERING,
        include_tick_marks=False, yticklabels='%(name)s %(index)s',
        xticklabels='%(index)s', include_colorbar=False,
        title='Data seed {}'.format(d), order=list(np.arange(20)))

    # # N_clusters
    # pl.visualization.render_trace(var=CLUSTERING.n_clusters, plt_type='trace', 
    #     ax=fig.add_subplot(2,5,d+1), section='posterior', include_burnin=True, 
    #     title='Data seed {}'.format(d), rasterized=True)

    # Process variance
    # pv = mcmc.graph[STRNAMES.PROCESSVAR]
    # mean_post = pl.variables.summary(pv)['mean']
    # pl.visualization.render_trace(var=pv, plt_type='trace', 
    #     ax=fig.add_subplot(2,5,d+1), section='posterior', include_burnin=True, 
    #     title='Data seed {}. mean post: {}'.format(d, mean_post), rasterized=True)

    # # Data
    # subj = mcmc.graph.data.subjects.iloc(0)
    # pl.visualization.abundance_over_time(subj=subj, dtype='abs', legend=True,
    #     taxlevel=None, set_0_to_nan=True, yscale_log=True, 
    #     color_code_clusters=True, clustering=CLUSTERING, 
    #     ax=fig.add_subplot(2,5,d+1))

fig.suptitle('{} Time points'.format(nt), size=30)

plt.savefig(basepath + 'coclustering{}{}.png'.format(nt,m))



