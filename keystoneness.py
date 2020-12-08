'''Compute the keystoneness

Record the steady state at each gibb step

Author: David Kaplan
Date: 12/07/20
MDSINE2 version: 4.0.6

Input format
------------
There are two different input formats that you can pass in:
    1) MDSINE2.BaseMCMC pickle
    2) Folder of numpy arrays
For most users, passing in the MDSINE2.BaseMCMC file is the way to go. Users
might want to pass in the folder only if they are running jobs in parallel
for this simulation and have many different jobs accessing the data at once. 

You can load the MCMC chain from either a `mcmc.pkl` file or from a folder.
If you load it from a folder, then it must have the following structure
folder/
    growth.npy # np.ndarray(n_gibbs, n_taxa)
    interactions.npy # np.ndarray(n_gibbs, n_taxa, n_taxa)

NOTE: perturbations can be specified in the folder but since they are not used
they are ignored

Leave-out table
---------------

'''
import mdsine2 as md2
from mdsine2.names import STRNAMES
import argparse
import logging
import numpy as np
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage=__doc__)
    parser.add_argument('--input', type=str, dest='input',
        help='Location of input (either folder of the numpy arrays or ' \
            'MDSINE2.BaseMCMC chain)')
    parser.add_argument('--study', type=str, dest='study',
        help='Study object to use for initial conditions')
    parser.add_argument('--leave-out-table', type=str, dest='leave_out_table',
        help='Table of which taxas to leave out')
    parser.add_argument('--leave-out-index', type=int, dest='leave_out_index',
        help='Index of the table to leave out. If Nothing is provided then it will ' \
             'do the entire table', default=None)
    parser.add_argument('--simulation-dt', type=float, dest='simulation_dt',
        help='Timesteps we go in during forward simulation', default=0.01)
    parser.add_argument('--n-days', type=str, dest='n_days',
        help='Number of days to simulate for', default=60)
    parser.add_argument('--limit-of-detection', dest='limit_of_detection',
        help='If any of the taxas have a 0 abundance at the start, then we ' \
            'set it to this value.',default=1e5)
    parser.add_argument('--sim-max', dest='sim_max',
        help='Maximum value', default=1e20)
    parser.add_argument('--output-basepath', '-o', type=str, dest='basepath',
        help='Where to save the output')
    args = parser.parse_args()

    md2.config.LoggingConfig(level=logging.INFO)
    study = md2.Study.load(args.study)
    basepath = args.basepath
    os.makedirs(basepath)

    # Get the traces of the parameters
    # --------------------------------
    if '.pkl' in args.input:
        # This is the chain
        logging.info('Input is an MDSINE2.BaseMCMC object')
        mcmc = md2.BaseMCMC.load(args.input)

        growth = mcmc.graph[STRNAMES.GROWTH_VALUE].get_trace_from_disk()
        self_interactions = mcmc.graph[STRNAMES.SELF_INTERACTION_VALUE].get_trace_from_disk()
        interactions = mcmc.graph[STRNAMES.INTERACTIONS_OBJ].get_trace_from_disk()
        interactions[np.isnan(interactions)] = 0
        self_interactions = -np.absolute(self_interactions)
        for i in range(self_interactions.shape[1]):
            interactions[:,i,i] = self_interactions[:, i]

    else:
        # This is a folder
        logging.info('input is a folder')
        growth = np.load(os.path.join(args.input, 'growth.npy'))
        interactions = np.load(os.path.join(args.input, 'interactions.npy'))

    