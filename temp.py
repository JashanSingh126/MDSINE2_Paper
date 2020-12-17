import mdsine2 as md2
from mdsine2.names import STRNAMES
import pandas as pd
import logging
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
import os

md2.LoggingConfig(level=logging.INFO)

fname = 'output/mdsine2/healthy-seed0/mcmc.pkl'

syn = md2.synthetic.make_semisynthetic(chain=fname, min_bayes_factor=5)

init_dist = md2.variables.Uniform(5e5, 1e7)
processvar = md2.model.MultiplicativeGlobal(0.2**2)

syn.times = md2.synthetic.subsample_timepoints(syn.times, N=35)

syn.generate_trajectories(dt=0.01, init_dist=init_dist, processvar=processvar)

# print(syn.model.growth)
# print(syn.times)

# for pert in syn.G.perturbations:
#     print(pert)

# sys.exit()

study = syn.simulateMeasurementNoise(a0=1e-10, a1=0.06, qpcr_noise_scale=0.3, 
    approx_read_depth=60000, name='ssss')

md2.visualization.abundance_over_time(study['2'], dtype='abs', yscale_log=True)




plt.show()