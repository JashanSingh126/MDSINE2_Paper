'''Compute the keystoneness

Forward simulate the dynamics with taxa(s) left out of the system to see how the steady
state changes

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
The table passed into `--leave-out-table` is a table that indicates the index of
taxon(s) that should be left out for each keystoneness. Each line represents the taxon(s)
that should be left out. Each line is a separated list of the indexes that 
should be left out. Example:
table.csv
    0,8,4
    1,2,3
    6
    5,9
specify the separator for each element with the `--sep` command. The default separator
is ','

To specify which line you want to compute, set the index with the command `--leave-out-index`.
If you want to compute the steady state without leaving any out, do not specify the 
`--leave-out-index` command. If you want to compute all of the indices, 
use the argument `--leave-out-index all`.

Forward simulate vs. making the table
-------------------------------------
Once you have forward simulated each index of the table, make the table of the steady states.
taxa that are left out are set to nan.

Default is to forward simulate all of the indexes and to make the table.
To forward simulate, set `--forward-sim 1`. To not forward simulate, set `--forward-sim 0`.
To make the table, set `--make-table 1`. To not make the table, set `--make-table 0`.
'''
import mdsine2 as md2
from mdsine2.names import STRNAMES
import argparse
import logging
import numpy as np
import os
import time
import re
import pandas as pd

def _forward_sim(growth, interactions, initial_conditions, dt, sim_max, n_days):
    '''Forward simulate with the given dynamics 
    (n_gibbs, n_taxa, n_times)
    '''

    dyn = md2.model.gLVDynamicsSingleClustering(growth=None, interactions=None, 
        start_day=0, sim_max=sim_max)
    times = np.arange(n_days+1)
    initial_conditions = initial_conditions.reshape(-1,1)

    start_time = time.time()
    pred_matrix = np.zeros(shape=(growth.shape[0], growth.shape[1], len(times)))
    for gibb in range(growth.shape[0]):
        if gibb % 5 == 0 and gibb > 0:
            logging.info('{}/{} - {}'.format(gibb,growth.shape[0], 
                time.time()-start_time))
            start_time = time.time()
        dyn.growth = growth[gibb]
        dyn.interactions = interactions[gibb]

        x = md2.integrate(dynamics=dyn, initial_conditions=initial_conditions, 
            dt=dt, n_days=times[-1]+dt, subsample=True, times=times)
        pred_matrix[gibb] = x['X']
    return pred_matrix

def _compute_distance(base, leftout, distance, mask):
    '''Compute the distance between `leftout` and `base` with the 
    distance `distance`. Note that `leftout` will have some NaNs that need to 
    be indexed out for base.

    Parameters
    ----------
    base : np.ndarray(n_taxa)
        steady-state of the system without any taxa left out
    leftout : np.ndarray(n_taxa)
        steady-state of the system with taxa left out
    distance : str
        distance metric
    '''
    def _l2(arr1, arr2):
        diff = arr1 - arr2
        return np.sqrt(np.sum(np.square(diff)))
    base = base[mask]
    if distance == 'l2':
        d = _l2(base, leftout)
    else:
        raise ValueError('`distance` ({}) metric not recognized'.format(distance))
    return d

if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage=__doc__)
    parser.add_argument('--input', type=str, dest='input',
        help='Location of input (either folder of the numpy arrays or ' \
            'MDSINE2.BaseMCMC chain)')
    parser.add_argument('--study', type=str, dest='study',
        help='Study object to use for initial conditions')
    parser.add_argument('--leave-out-table', type=str, dest='leave_out_table',
        help='Table of which taxa to leave out')
    parser.add_argument('--leave-out-index', type=str, dest='leave_out_index',
        help='Index of the table to leave out. If you want to compute all of ' \
             'the lines in `--leave-out-table`, then set to "all". If nothing ' \
             'is passed in, then it will compute the keystoneness with nothing ' \
             'left out.', default='all')
    parser.add_argument('--forward-simulate', '-fs', type=int, dest='forward_simulate',
        help='If 1, compute the forward simulation of the index. If 0, do not', default=1)
    parser.add_argument('--make-table', '-mt', type=int, dest='make_table',
        help='If 1, make the table of the steady states. If 0, do not', default=1)
    parser.add_argument('--compute-keystoneness', type=int, dest='compute_keystoneness',
        help='If 1, compute the keystoneness. Otherwise dont.', default=1)
    parser.add_argument('--distance', type=str, dest='distance', default='l2',
        help='This is the distance to compute from the base and the left out set.')
    parser.add_argument('--sep', type=str, dest='sep', default=',',
        help='separator for the leave out table')
    parser.add_argument('--simulation-dt', type=float, dest='simulation_dt',
        help='Timesteps we go in during forward simulation', default=0.01)
    parser.add_argument('--n-days', type=int, dest='n_days',
        help='Number of days to simulate for', default=60)
    parser.add_argument('--limit-of-detection', dest='limit_of_detection',
        help='If any of the taxa have a 0 abundance at the start, then we ' \
            'set it to this value.',default=1e5, type=float)
    parser.add_argument('--sim-max', dest='sim_max', type=float,
        help='Maximum value', default=1e20)
    parser.add_argument('--output-basepath', '-o', type=str, dest='basepath',
        help='Where to save the output')
    args = parser.parse_args()

    md2.config.LoggingConfig(level=logging.INFO)
    study = md2.Study.load(args.study)
    basepath = os.path.join(args.basepath, study.name)
    os.makedirs(basepath, exist_ok=True)

    # Get the initial conditions
    # --------------------------
    M = study.matrix(dtype='abs', agg='mean', times='intersection')
    initial_conditions_master = M[:,0]
    initial_conditions_master[initial_conditions_master==0] = args.limit_of_detection

    # Load table
    # ----------
    f = open(args.leave_out_table, 'r')
    tbl = f.read()
    f.close()
    lines = tbl.split('\n')

    # Remove empty lines
    _lines = []
    for line in lines:
        if line == '':
            continue
        _lines.append(line)
    lines = _lines

    # Index out if necessary
    # ----------------------
    leave_out_index = args.leave_out_index
    if md2.isstr(leave_out_index):
        leave_out_index = leave_out_index.lower()
    if leave_out_index is None or leave_out_index == 'none':
        logging.info('No index provided, not indexing out any taxon')
        idxs = [None]
    elif leave_out_index == 'all':
        logging.info('"all" provided as index. Iterating over each index')
        idxs = np.arange(len(lines))
        idxs = [None] + idxs.tolist()
    else:
        try:
            idxs = int(leave_out_index)
        except:
            logging.critical('--leave-out-index ({}) not recognized as an index'.format(
                leave_out_index))
            raise
        idxs = [idxs]

    for idx in idxs:
        if idx is not None:
            if idx >= len(lines):
                raise ValueError('index {} is out range ({} total)'.format(idx, len(lines)))

    if args.forward_simulate == 1:
        logging.info('Forward simulating')
        # Get the traces of the parameters
        # --------------------------------
        if '.pkl' in args.input:
            # This is the chain
            logging.info('Input is an MDSINE2.BaseMCMC object')
            mcmc = md2.BaseMCMC.load(args.input)

            growth = mcmc.graph[STRNAMES.GROWTH_VALUE].get_trace_from_disk(mode='r')
            self_interactions = mcmc.graph[STRNAMES.SELF_INTERACTION_VALUE].get_trace_from_disk(mode='r')
            interactions = mcmc.graph[STRNAMES.INTERACTIONS_OBJ].get_trace_from_disk(mode='r')
            interactions[np.isnan(interactions)] = 0
            self_interactions = -np.absolute(self_interactions)
            for i in range(self_interactions.shape[1]):
                interactions[:,i,i] = self_interactions[:, i]
            growth_master = growth
            interactions_master = interactions

        else:
            # This is a folder
            logging.info('input is a folder')
            growth_master = np.load(os.path.join(args.input, 'growth.npy'))
            interactions_master = np.load(os.path.join(args.input, 'interactions.npy'))

        # Forward simulate
        for idx in idxs:
            if idx is not None:
                oidxs = [int(ele) for ele in lines[idx].split(args.sep)]
                logging.info('indexing out {}'.format(oidxs))

                mask = np.ones(len(initial_conditions_master), dtype=bool)
                mask[oidxs] = False
                growth = growth_master[:, mask]
                interactions = interactions_master[:, mask, :]
                interactions = interactions[:, :, mask]
                initial_conditions = initial_conditions_master[mask]
            else:
                logging.info('not indexing out anything')
                growth = growth_master
                interactions = interactions_master
                initial_conditions = initial_conditions_master

            pred_matrix = _forward_sim(growth=growth, interactions=interactions, 
                initial_conditions=initial_conditions, dt=args.simulation_dt,
                sim_max=args.sim_max, n_days=args.n_days)

            # Save the forward sims
            if idx is None:
                name = 'none'
            else:
                name = str(idx)
            fname = os.path.join(basepath, 'study{}-lo{}-forward-sims.npy'.format(study.name, name))
            np.save(fname, pred_matrix)
    
    if args.make_table == 1:
        # Make the table
        # --------------
        logging.info('Make the table')
        re_find = re.compile(r'^study(.*)-lo(.*)-forward-sims.npy$')

        # Get the steady-state of the base
        fname = os.path.join(basepath, 'study{studyname}-lonone-forward-sims.npy'.format(
            studyname=study.name))
        if os.path.isfile(fname):
            base = np.load(fname)
            base = base[:,:,-1] # Get last timepoint
            ss_base = np.mean(base, axis=0)
        else:
            raise ValueError('`base` ({}) not found'.format(fname))

        fnames = os.listdir(basepath)
        
        data = []
        idxs = []

        if args.compute_keystoneness:
            keystoneness = []
        for fname in fnames:
            try:
                studyname, leaveout = re_find.findall(fname)[0]
            except:
                logging.warning('{} not recognized as a pattern. skipping'.format(fname))
                continue
            if studyname != study.name:
                logging.warning('{} not the same study as input {}. skipping'.format(
                    studyname, study.name))
                continue
            if leaveout == 'none':
                continue
            else:
                leaveout = int(leaveout)

            M = np.load(os.path.join(basepath, fname))
            M = M[:, :, -1]  # Get last timepoint
            steady_state = np.nanmean(M, axis=0)

            # Set the respective elements to nan
            oidxs = [int(ele) for ele in lines[leaveout].split(args.sep)]
            mask = np.ones(len(study.taxa), dtype=bool)
            mask[oidxs] = False
            ss = np.zeros(len(study.taxa)) * np.nan
            ss[mask] = steady_state
            steady_state = ss

            # compute the distance over each gibb step
            if args.compute_keystoneness:
                dists = np.zeros(M.shape[0])
                for i in range(len(dists)):
                    dists[i] = _compute_distance(base[i], M[i], distance=args.distance, mask=mask)
                keystoneness.append(np.mean(dists))
            
            data.append(steady_state)
            idxs.append(leaveout)
        
        if len(data) != len(lines):
            raise ValueError('You are making the table. {} total lines were found instead of {}'.format( 
                len(data), len(lines)+1))
        
        # order the table
        data = np.asarray(data)
        idxs = np.argsort(idxs)
        data = data[idxs, :]
        data = np.vstack((ss_base.reshape(1,-1), data))

        columns = [taxa.name for taxa in study.taxa]
        index = ['base'] + lines

        print(data.shape)
        print(columns)
        print(index)

        df = pd.DataFrame(data, index=index, columns=columns)
        df.to_csv(os.path.join(basepath, 'steady-state-table.tsv'), sep='\t', index=True, header=True)

        # Make the keystoneness
        if args.compute_keystoneness:

            columns = ['{} distance'.format(args.distance)]
            data = np.asarray(keystoneness).reshape(-1,1)

            df = pd.DataFrame(data, index=index[1:], columns=columns)
            df.to_csv(os.path.join(basepath, 'keystoneness.tsv'), sep='\t', index=True, header=True)