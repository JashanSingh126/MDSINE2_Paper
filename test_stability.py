#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 09:50:21 2019

@author: vbucci
"""

# load packages
import numpy as np
import numpy.random as npr
import logging
import copy
import math
import time
import pickle
import sys
import random
import multiprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.spatial.distance
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from scipy.stats import truncnorm, bernoulli
from numpy import linalg as LA

from pathlib import Path
cwd = Path().resolve()
print(cwd)

import linear_stability

# initialize linear stability data
stability_data = linear_stability.StabilityData("None","None","None")

# load parameters file from MDSINE-1
#fname='~/mnt_donatello/vbucci/mnt_bucci_TEX227/Dropbox/klebsiella_th17_modeling/donor_F_qpcr_results/bvs_parameters.parameters.txt'
fname='~/mnt_donatello/vbucci/mnt_bucci_TEX227/Dropbox/klebsiella_th17_modeling/donor_F_16S_results/BVS.results.parameters_withSpline.txt'
#fname='/Users/vbucci/Dropbox/klebsiella_th17_modeling/donor_F_qpcr_results/bvs_parameters.parameters.txt'
sep='\t'
stability_data = stability_data.get_inferred_growth_and_interactions(fname,sep)

stability_data.growth_rates
stability_data.interactions
stability_data.species_names

# calculate steady states of the full system
stability_data = stability_data.get_steady_state()
stability_data.steady_state

# caculate jacobian and eigenvalues of the full system
stability_data = stability_data.get_jacobian()
stability_data = stability_data.get_eigenvalues()

# plot eigenvalues
stability_data.plot_eigenvalues()

# combinatorial analysis - find all combos of certain size
stability_data = stability_data.get_combinations_of_size_n(10)
stability_data.combinations_of_size_n[1]

# for every determined combination determine if stable and print file with
# parameters and stable profile for simulations            
#stability_data.perform_combinatorial_stability("~/temp")

# for every determined combination determine if stable, add invader and invader density 
# and print file with for simulations            
#stability_data.perform_combinatorial_stability_w_invasion("~/temp_2",'Kp2H7',500, 50000)
stability_data.perform_combinatorial_stability_w_invasion("~/temp_2",'Klebsiella_pneumoniae',500,200000)



