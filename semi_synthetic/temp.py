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
import argparse

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as plticker

import pylab as pl
# import config

sys.path.append('..')
import synthetic
import model
import config


# subjset_real = pl.SubjectSet.load('../pickles/real_subjectset.pkl')
subjset_semi = pl.SubjectSet.load('output/base_data/subjset_ds0_nNone_pv0.1_mn0.1_nr2_nt35_usTrue_exactFalse.pkl')







# syn = synthetic.SyntheticData.load('base_data/preprocessed_semisynthetic_healthy.pkl')

# M = syn.get_full_interaction_matrix()
# pl.visualization.render_interaction_strength(M, asvs=syn.dynamics.asvs, log_scale=True, 
#     clustering=syn.dynamics.clustering)
# plt.savefig('true_interaction_matrix.pdf')
# sys.exit()


for subj in subjset_semi:
    print(subj.name)
    pl.visualization.abundance_over_time(subj=subj, dtype='abs', taxlevel=None, set_0_to_nan=True,
        yscale_log=True, title='Semi {}'.format(subj.name))
    plt.savefig('semi' + subj.name + '.pdf')
    plt.close()
sys.exit()

# for subj in subjset_real:
#     if subj.name not in ['6', '7', '8', '9', '10']:
#         continue
#     print(subj.name)
#     pl.visualization.abundance_over_time(subj=subj, dtype='abs', taxlevel=None, set_0_to_nan=True,
#         yscale_log=True, title='Real {}'.format(subj.name))
#     plt.savefig('real' + subj.name + '.pdf')
#     plt.close()

    
fname = '../output_real/pylab24/real_runs/strong_priors/fixed_top/healthy0_5_0.0001_rel_2_5/ds0_is3_b5000_ns15000_mo-1_logTrue_pertsmult/graph_leave_out-1/'

syn = synthetic.make_semisynthetic(fname+'mcmc.pkl', 10, set_times=True, init_dist_timepoint=0.5, 
    hdf5_filename=fname+'traces.hdf5')
processvar = model.MultiplicativeGlobal(asvs=syn.asvs)
processvar.value = 0.05
for i in range(5):
    syn.generate_trajectories(dt=0.001, processvar=processvar)
a0,a1 = config.calculate_reads_a0a1(0.0001)
subjset = syn.simulateRealRegressionDataNegBinMD(a0=a0, a1=a1, qpcr_noise_scale=0.1, subjset='../pickles/real_subjectset.pkl')

for subj in subjset:
    pl.visualization.abundance_over_time(subj=subj, dtype='abs', taxlevel=None, set_0_to_nan=True,
        yscale_log=True, title='Semi {}'.format(subj.name))
    plt.savefig('unhealthy{}.pdf'.format(subj.name))



