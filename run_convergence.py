'''Run rhat for all (most) the variables
'''
import numpy as np
import logging
import sys
import pandas as pd
import h5py
import inspect
import random
import copy
import os
import shutil
import math
import pickle

import pylab as pl

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

import config
from names import STRNAMES


if __name__ == '__main__':

    start = 10000
    ends = [15000,20000,25000,30000]

    basepath1 = 'output_real/healthy0_5_0.00025_rel_2_7/ds0_is0_b5000_ns30000_mo-1_logTrue_pertsaddit/graph_leave_out3/'
    basepath2 = 'output_real/healthy0_5_0.00025_rel_2_7/ds1_is1_b6000_ns30000_mo-1_logTrue_pertsaddit/graph_leave_out3/'

    chains = []
    for basepath in [basepath1, basepath2]:
        chains.append(pl.inference.BaseMCMC.load(basepath + 'mcmc.pkl'))

    OTUS = chains[0].graph.data.otus

    print(len(chains[0].graph[STRNAMES.GROWTH_VALUE].get_trace_from_disk(section='entire')[:,0]))
    print(len(chains[1].graph[STRNAMES.GROWTH_VALUE].get_trace_from_disk(section='entire')[:,0]))
    

    fmt = '%(name)s %(lca)s'

    f = open('convergence2_2subj.txt', 'w')

    f.write('Growths\n')
    f.write('=======\n')
    rets = []
    for end in ends:
        rets.append(pl.inference.r_hat(chains=chains, vname=STRNAMES.GROWTH_VALUE, start=start, end=end))
    rets = np.array(rets).T

    f.write('{} : Gibb steps\n'.format([end-start for end in ends]))
    for oidx in range(rets.shape[0]):
        otu = OTUS[oidx]
        f.write('{} - {}\n'.format(rets[oidx, :], pl.util.otuname_formatter(otu=otu, otus=OTUS, 
            format=fmt)))

    # f.write('Interactions\n')
    # for end in ends:
    #     interacts = pl.inference.r_hat(chains=chains, vname=STRNAMES.INTERACTIONS_OBJ, start=start, end=end)

    f.write('\n\n\nConcentration\n')
    f.write('=============\n')
    for end in ends:
        print('conc', end)
        ret = pl.inference.r_hat(chains=chains, vname=STRNAMES.CONCENTRATION, start=start, end=end)
        f.write('\tAfter {} Gibb steps: {}\n'.format(end-start, ret))

    f.write('\n\n\nNumber of clusters\n')
    f.write('==================\n')
    for end in ends:
        print('num clusters', end)
        ret = pl.inference.r_hat(chains=chains, vname=STRNAMES.CLUSTERING_OBJ + '_n_clusters', start=start, end=end)
        f.write('\tAfter {} Gibb steps: {}\n'.format(end-start, ret))

    f.write('\n\n\nProcess variance\n')
    f.write('================\n')
    for end in ends:
        print('pv', end)
        ret = pl.inference.r_hat(chains=chains, vname=STRNAMES.PROCESSVAR, start=start, end=end)
        f.write('\tAfter {} Gibb steps: {}\n'.format(end-start, ret))

    f.close()

    # fig = plt.figure(figsize=(15,10))
    # ax = fig.add_subplot(111)
    # ax = sns.heatmap(ret, annot=True, fmt='.2f', ax=ax)
    # fig.suptitle(r'$\hat{R}$' + ' interactions', size=20)
    # plt.savefig(basepath + 'interactions.pdf')
    # plt.close()

    


