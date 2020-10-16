'''Run the keystoneness metric.

Calculate a baseline
'''

import numpy as np
import pandas as pd
import logging
import sys
import scipy.stats
import scipy.sparse
import scipy.spatial
import numba
import time
import collections
import h5py
import argparse

import names
import pylab as pl
import model

logging.basicConfig(level=logging.DEBUG)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--type', type=str, dest='keystoneness_type',
        help='What type of keystoneness to do. Options: `perturbation`, `leave-one-out`')
    parser.add_argument('--model', '-m', type=str, dest='model',
        help='Posterior chain we want to do inference over')
    parser.add_argument('--data', '--input', type=str, dest='input',
        help='File describing which ASVs to leave out')
    parser.add_argument('--output-tbl', '-otbl', type=str, dest='output_tbl',
        help='Where to save the output table of all the abundances')
    parser.add_argument('--max-posterior', type=int, dest='max_posterior',
        help='TESTING USE ONLY', default=None)
    parser.add_argument('--n-cpus', '-mp', type=int, dest='n_cpus',
        help='How many cpus to use for multiprocessing', default=None)
    parser.add_argument('--compute-base', '-cb', type=int, dest='compute_base',
        help='If 1, compute the base, else do not', default=1)

    args = parser.parse_args()
    return args

def keystoneness_perturbation(chain_fname, fname, outfile_table, initial_conditions=None, 
    max_posterior=None):

    pass

# def keystoneness_knockdown():
#     pass
    
def keystoneness_leave_one_out(chain_fname, fname, outfile_table, initial_conditions=None, 
    max_posterior=None, mp=None, compute_base=1):
    '''The file(s) show a list of asvs to delete, comma separated
    All of the asvs on a single line should be deleted at once. Note that 
    this is all a single process

    Parameters
    ----------
    chain_fname : str
        This is the location of the Pylab MCMC chain filename that is saved from inference
    fname : str
        This is the location of the file that describes which ASVs to be held out
    output_table : str
        This is where to save the table of the concentrations and the base
    '''

    chain = pl.inference.BaseMCMC.load(chain_fname)
    subjset = chain.graph.data.subjects

    SECTION = 'posterior'
    if max_posterior is None:
        max_posterior = chain.n_samples - chain.burnin
    
    read = False
    i = 0
    while not read:
        try:
            time.sleep(0.5)
            growth_master = chain.graph[names.STRNAMES.GROWTH_VALUE].get_trace_from_disk(section=SECTION)[:max_posterior, ...]
            read = True
        except:
            print('failed growth')   
            i += 1
            if i > 10:
                raise
    read = False
    i = 0
    while not read:
        try:
            time.sleep(0.5)
            si_master =  chain.graph[names.STRNAMES.SELF_INTERACTION_VALUE].get_trace_from_disk(section=SECTION)[:max_posterior, ...]
            read = True
        except:
            print('failed si')   
            i += 1
            if i > 10:
                raise
    read = False
    i = 0
    while not read:
        try:
            time.sleep(0.5)
            A_master =  chain.graph[names.STRNAMES.INTERACTIONS_OBJ].get_trace_from_disk(section=SECTION)[:max_posterior, ...]
            read = True
        except:
            print('failed A')       
            i += 1
            if i > 10:
                raise

    print(growth_master.shape)

    dyn = model.gLVDynamicsSingleClustering(asvs=subjset.asvs, log_dynamics=True, 
        perturbations_additive=False)
    dyn.growth = growth_master
    dyn.self_interactions = si_master
    dyn.interactions = A_master

    if initial_conditions is None:
        df = subjset.df(dtype='abs', agg='mean', times='union')
        initial_conditions = df[0.5].to_numpy()

        for i in range(len(initial_conditions)):
            if initial_conditions[i] == 0:
                initial_conditions[i] = pl.random.truncnormal.sample(mean=1e5, std=1e5, low=1e2)
        initial_conditions = initial_conditions.reshape(-1,1)
    else:
        inital_conditions = np.load(initial_conditions)

    days = 20
    sim_dt = 0.01
    BASE_CONCENTRATIONS = np.zeros(shape=growth_master.shape, dtype=float)
    # Generate the base concentrations

    if compute_base == 1:
        if mp is None:
            for i in range(growth_master.shape[0]):
                start_time = time.time()
                pred_dyn = model.gLVDynamicsSingleClustering(asvs=subjset.asvs, 
                    log_dynamics=True, perturbations_additive=False, sim_max=1e20, start_day=0)
                pred_dyn.growth = growth_master[i]
                pred_dyn.self_interactions = si_master[i]
                pred_dyn.interactions = A_master[i]

                _d = pl.dynamics.integrate(dynamics=pred_dyn, 
                    initial_conditions=initial_conditions,
                    dt=sim_dt, n_days=days, 
                    subsample=True, times=np.arange(days), log_every=None)
                BASE_CONCENTRATIONS[i] = _d['X'][:,-1]
                if i %10 == 0:
                    logging.debug('{}/{}: {}'.format(i,growth_master.shape[0], time.time()-start_time))
                # print(BASE_CONCENTRATIONS[i,:])
        else:
            # Integrate over posterior with multiprocessing
            raise NotImplementedError('Not working on windows')
            pool = pl.multiprocessing.PersistentPool(ptype='dasw')
            try:
                for i in range(mp):
                    pool.add_worker(_ForwardSimWorker(asvs=subjset.asvs,
                        initial_conditions=initial_conditions, start_day=0,
                        sim_dt=sim_dt, n_days=days+sim_dt, log_integration=True,
                        perturbations_additive=False, sim_times=np.arange(days+sim_dt),
                        name='worker{}'.format(i)))
                    # pool.staged_map_start(func='integrate')
                for i in range(growth_master.shape[0]):
                    kwargs = {
                        'i': i,
                        'growth': growth_master[i],
                        'self_interactions': si_master[i],
                        'interactions': A_master[i],
                        'perturbations': None}
                    pool.staged_map_put(kwargs)

                ret = pool.staged_map_get()
                pool.kill()
                BASE_CONCENTRATIONS = np.asarray(ret, dtype=float)

            except:
                pool.kill()
                raise

        row_names = ['base']
    else:
        row_names = []
        BASE_CONCENTRATIONS = None

    dists = {}

    f = open(fname, 'r')
    args = f.read().split('\n')
    f.close()

    names_to_del_lst = []
    for arg in args:
        # Get rid of replicates
        lst_ = arg.split(',')
        lst = []
        for ele in lst_:
            if ele not in lst:
                lst.append(ele)
        _tpl_names = tuple(lst)
        names_to_del_lst.append(_tpl_names)
        row_names.append(str(_tpl_names))

    output_tbl = np.zeros(shape=(len(row_names), 
        len(chain.graph.data.asvs)), dtype=float)
    if compute_base == 1:
        output_tbl[0] = np.mean(BASE_CONCENTRATIONS, axis=0)

    print('output_tbl.shape', output_tbl.shape)
    
    for names_iii, names_to_del in enumerate(names_to_del_lst):
        idxs_to_del = [subjset.asvs[name].idx for name in names_to_del]
        # Take out asv aidxs and do the forward simulation
        logging.debug('{}/{}: {}'.format(names_iii, len(args), names_to_del))

        mask = np.ones(len(subjset.asvs), dtype=bool)
        mask[idxs_to_del] = False

        print(mask.shape)

        temp_growth = growth_master[:, mask]
        temp_self_interactions = si_master[:, mask]

        print('A_master', A_master.shape)
        temp_interactions = np.delete(A_master, idxs_to_del, 1)
        print(temp_interactions.shape)
        temp_interactions = np.delete(temp_interactions, idxs_to_del, 2)
        print(temp_interactions.shape)

        init_conc = initial_conditions.ravel()[mask].reshape(-1,1)
        concentrations = np.zeros(shape=(temp_growth.shape[0], np.sum(mask)))

        for i in range(growth_master.shape[0]):
            pred_dyn = model.gLVDynamicsSingleClustering(asvs=subjset.asvs, 
                log_dynamics=True, perturbations_additive=False, sim_max=1e20, start_day=0)
            pred_dyn.growth = temp_growth[i]
            pred_dyn.self_interactions = temp_self_interactions[i]
            pred_dyn.interactions = temp_interactions[i]

            iii = pl.dynamics.integrate(pred_dyn, initial_conditions=init_conc, 
                dt=0.01, n_days=days, times=np.arange(days), subsample=True)
            concentrations[i] = iii['X'][:,-1]
            if i % 20 == 0:
                logging.debug('\t{}/{}'.format(i, growth_master.shape[0]))

        output_tbl[names_iii + compute_base, mask] = np.mean(concentrations, axis=0)
        output_tbl[names_iii+compute_base, ~mask] = np.nan
        diff = concentrations - BASE_CONCENTRATIONS[:,mask]
        mean_diff = np.mean(diff, axis=0)
        dists[names_to_del] = np.sqrt(np.sum(np.square(mean_diff)))

    idxs = (np.argsort(list(dists.values())))[::-1]
    keys = list(dists.keys())
    
    # Save the table of the concentrations
    df = pd.DataFrame(data=output_tbl, index=row_names, 
        columns=chain.graph.data.asvs.names.order)
    df.to_csv(outfile_table, sep='\t', header=True, index=True)

class _ForwardSimWorker(pl.multiprocessing.PersistentWorker):
    '''Multiprocessed forward simulation.
    '''
    def __init__(self, asvs, initial_conditions, sim_dt, n_days, name, 
        log_integration, perturbations_additive, sim_times, start_day):
        self.asvs = asvs
        self.initial_conditions = initial_conditions
        self.sim_dt = sim_dt
        self.n_days = n_days
        self.name = name
        self.log_integration = log_integration
        self.perturbations_additive = perturbations_additive
        self.sim_times = sim_times
        self.start_day = start_day

    def integrate(self, growth, self_interactions, interactions, perturbations, i):
        '''forward simulate
        '''

        pred_dyn = model_module.gLVDynamicsSingleClustering(asvs=self.asvs, 
            log_dynamics=self.log_integration, start_day=self.start_day,
            perturbations_additive=self.perturbations_additive)
        pred_dyn.growth = growth
        pred_dyn.self_interactions = self_interactions
        pred_dyn.interactions = interactions
        pred_dyn.perturbations = perturbations

        _d = pl.dynamics.integrate(dynamics=pred_dyn, initial_conditions=self.initial_conditions,
            dt=self.sim_dt, n_days=self.n_days, subsample=True, 
            times=self.sim_times, log_every=None)
        print('integrate {} from process {}'.format(i, self.name))
        return _d['X']

if __name__ == '__main__':

    args = parse_args()
    
    if args.keystoneness_type == 'leave-one-out':
        keystoneness_leave_one_out(chain_fname=args.model, fname=args.input, 
            outfile_table=args.output_tbl, 
            max_posterior=args.max_posterior, mp=args.n_cpus)
    elif args.keystoneness_type == 'perturbations':
        raise NotImplementedError('Not Implemented')
    else:
        raise ValueError('`type` ({}) not recognized'.format(args.keystoneness_type))