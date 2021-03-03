'''Forward simulate to steady state - first without a perturbation, then with a perturbation,
then with a perturbation again

'''
import numpy as np
import os
import mdsine2 as md2
from mdsine2.names import STRNAMES
from mdsine2.logger import logger
import argparse
import pickle
import time
import re

def _forward_sim(growth, interactions, perturbation, initial_conditions, dt, sim_max, 
    pert_start_day, pert_end_day, n_days):
    '''Forward simulate with the given dynamics. First start with the perturbation
    off, then on, then off.

    Parameters
    ----------
    growth : np.ndarray(n_gibbs, n_taxa)
        Growth parameters
    interactions : np.ndarray(n_gibbs, n_taxa, n_taxa)
        Interaction parameters
    perturbation : np.ndarray(n_gibbs, n_taxa)
        Perturbation effect
    initial_conditions : np.ndarray(n_taxa)
        Initial conditions of the taxa
    dt : float
        Step size to forward simulate with
    sim_max : float, None
        Maximum clip for forward sim
    pert_start_day : float
        Day to start the perturbation
    pert_end_day : float
        Day to end the perturbation
    n_days : float
        Total number of days
    '''
    dyn = md2.model.gLVDynamicsSingleClustering(growth=None, interactions=None, 
        perturbation_ends=[pert_end_day], perturbation_starts=[pert_start_day], 
        start_day=0, sim_max=sim_max)
    initial_conditions = initial_conditions.reshape(-1,1)

    n_steps = int(n_days/dt) + 1

    start_time = time.time()
    pred_matrix = np.zeros(shape=(growth.shape[0], growth.shape[1], n_steps))
    for gibb in range(growth.shape[0]):
        if gibb % 5 == 0 and gibb > 0:
            logger.info('{}/{} - {}'.format(gibb,growth.shape[0], time.time()-start_time))
            start_time = time.time()

        dyn.growth = growth[gibb]
        dyn.interactions = interactions[gibb]
        dyn.perturbations = [perturbation[gibb]]

        x = md2.integrate(dynamics=dyn, initial_conditions=initial_conditions, 
            dt=dt, n_days=n_days, subsample=False)
        pred_matrix[gibb] = x['X']
    return pred_matrix

if __name__ == "__main__":
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
    parser.add_argument('--perturbation', '-p', type=str, dest='perturbation_name',
        help='Name of the perturbation you want to forward sim with')
    parser.add_argument('--sep', type=str, dest='sep', default=',',
        help='separator for the leave out table')
    parser.add_argument('--output-basepath', '-o', type=str, dest='basepath',
        help='Where to save the output')

    # Simulation params
    parser.add_argument('--forward-simulate', '-fs', type=int, dest='forward_simulate',
        help='If 1, compute the forward simulation of the index. If 0, do not', default=1)
    parser.add_argument('--start-perturbation', type=float, dest='start_pert_day',
        help='Start day of the perturbation', default=60)
    parser.add_argument('--end-perturbation', type=float, dest='end_pert_day',
        help='End day of the perturbation', default=120)
    parser.add_argument('--n-days', type=int, dest='n_days',
        help='Total umber of days to simulate for', default=180)
    parser.add_argument('--simulation-dt', type=float, dest='simulation_dt',
        help='Timesteps we go in during forward simulation', default=0.01)
    parser.add_argument('--limit-of-detection', dest='limit_of_detection',
        help='If any of the taxa have a 0 abundance at the start, then we ' \
            'set it to this value.',default=1e5, type=float)
    parser.add_argument('--sim-max', dest='sim_max', type=float,
        help='Maximum value', default=1e20)

    # statistics
    parser.add_argument('--compute-statistics', type=int, default=1, dest='compute_statistics',
        help='If 1, compute the summary statistics of the trajectory')
    parser.add_argument('--last-unstable-point', type=float, default=[1,5], nargs='+',
        help='Time of the last unstable point. The unstable point is set to X% deviation ' \
             'from the steady state', dest='last_unstable_point')

    args = parser.parse_args()
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
        raise ValueError('Must provide an index')
    elif leave_out_index == 'all':
        logger.info('"all" provided as index. Iterating over each index')
        idxs = np.arange(len(lines))
        idxs = idxs.tolist()
    else:
        try:
            idxs = int(leave_out_index)
        except:
            logger.critical('--leave-out-index ({}) not recognized as an index'.format(
                leave_out_index))
            raise
        idxs = [idxs]

    for idx in idxs:
        if idx is not None:
            if idx >= len(lines):
                raise ValueError('index {} is out range ({} total)'.format(idx, len(lines)))

    # Forward simulate if necessary
    # -----------------------------
    if args.forward_simulate == 1:
        logger.info('Forward simulating')
        # Get the traces of the parameters
        # --------------------------------
        if '.pkl' in args.input:
            # This is the chain
            logger.info('Input is an MDSINE2.BaseMCMC object')
            mcmc = md2.BaseMCMC.load(args.input)

            # Check if the respective perturbation is there and load it
            if mcmc.graph.perturbations is None:
                raise ValueError('There are no perturbations')
            pn = args.perturbation_name
            print(type(study.perturbations))
            if pn not in mcmc.graph.perturbations:
                raise ValueError('`perturbation` ({}) not found ({})'.format(
                    np, list(mcmc.graph.perturbations.keys())))
            perturbation_master = mcmc.graph.perturbations[pn].get_trace_from_disk()
            perturbation_master[np.isnan(perturbation_master)] = 0

            # Load the rest of the parameters
            growth = mcmc.graph[STRNAMES.GROWTH_VALUE].get_trace_from_disk()
            self_interactions = mcmc.graph[STRNAMES.SELF_INTERACTION_VALUE].get_trace_from_disk()
            interactions = mcmc.graph[STRNAMES.INTERACTIONS_OBJ].get_trace_from_disk()
            interactions[np.isnan(interactions)] = 0
            self_interactions = -np.absolute(self_interactions)
            for i in range(self_interactions.shape[1]):
                interactions[:,i,i] = self_interactions[:, i]
            growth_master = growth
            interactions_master = interactions

        else:
            # This is a folder
            logger.info('input is a folder')

            # Check if the respective perturbation is there and load it
            path = os.path.join(args.input, 'perturbations.pkl')
            if os.path.isfile(path):
                with open(path, 'rb') as handle:
                    perturbations = pickle.load(handle)
            else:
                raise ValueError('Could not find `perturbations.pkl`')
            if args.perturbation_name not in perturbations:
                raise ValueError('`perturbation` ({}) not found in pkl ({})'.format(
                    args.perturbation_name, list(perturbations.keys())))
            perturbation_master = perturbations[args.perturbation_name]

            # Load the rest of the parameters
            growth_master = np.load(os.path.join(args.input, 'growth.npy'))
            interactions_master = np.load(os.path.join(args.input, 'interactions.npy'))

        # Forward simulate
        for idx in idxs:
            oidxs = [int(ele) for ele in lines[idx].split(args.sep)]
            logger.info('indexing out {}'.format(oidxs))

            mask = np.ones(len(initial_conditions_master), dtype=bool)
            mask[oidxs] = False
            growth = growth_master[:, mask]
            interactions = interactions_master[:, mask, :]
            interactions = interactions[:, :, mask]
            perturbation = perturbation_master[:, mask]
            initial_conditions = initial_conditions_master[mask]

            pred_matrix = _forward_sim(growth=growth, interactions=interactions, 
                pert_end_day=args.end_pert_day, perturbation=perturbation, 
                pert_start_day=args.start_pert_day,
                initial_conditions=initial_conditions, dt=args.simulation_dt,
                sim_max=args.sim_max, n_days=args.n_days)

            # Save the forward sims
            name = str(idx)
            fname = os.path.join(basepath, 'study{}-lo{}-forward-sims.npy'.format(study.name, name))
            np.save(fname, pred_matrix)

    
    if args.compute_statistics:
        logger.info('Make the table')
        re_find = re.compile(r'^study(.*)-lo(.*)-forward-sims.npy$')

        # Get all of the files
        fnames = os.listdir(basepath)

        
        for fname in fnames:
            try:
                studyname, leaveout = re_find.findall(fname)[0]
            except:
                logger.warning('{} not recognized as a pattern. skipping'.format(fname))
                continue
            if studyname != study.name:
                logger.warning('{} not the same study as input {}. skipping'.format(
                    studyname, study.name))
                continue
            leaveout = int(leaveout)
            oidxs = [int(ele) for ele in lines[leaveout].split(args.sep)]
            
            # leave out index is `leaveout`
            # The indices of the OTUs that are left out for this file are `oidxs`
            # start time of the perturbation is `args.start_pert_day`
            # end time of the perturbation is `args.end_pert_day`
            # total number of days is `args.n_days`

            # matrix of the forward sims
            # shape (n_gibbs, n_taxa, n_steps), n_steps = n_days/dt
            M = np.load(os.path.join(basepath, fname))


    
