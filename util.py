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

import names
import pylab as pl
import model

def parse_mdsine1_cdiff(basepath):
    '''Biomass is in `biomass.txt`
    counts are in `counts.txt`
    metadata are in `metadata.txt`

    Parameters
    ----------
    basepath : str
        Path to the folder holding all of the information
    
    Returns
    -------
    pl.base.SubjectSet
    '''
    biomass = basepath + 'biomass.txt'
    dfbiomass = pd.read_csv(biomass, sep='\t')
    rows = ['{}'.format(i) for i in range(1, dfbiomass.shape[0]+1)]
    dfbiomass = pd.DataFrame(dfbiomass.values, index=rows, columns=dfbiomass.columns)

    counts = basepath + 'counts.txt'
    dfcounts = pd.read_csv(counts, sep='\t')
    dfcounts = dfcounts.set_index('#OTU ID')
    # print(dfcounts.head())

    metadata = basepath + 'metadata.txt'
    dfmeta = pd.read_csv(metadata, sep='\t')
    dfmeta = pd.DataFrame(dfmeta.values, index=rows, columns=dfmeta.columns)
    # print(dfmeta)

    asvs = pl.base.ASVSet(use_sequences=False)
    for row in dfcounts.index:
        asvs.add_asv(row)
    subjset = pl.base.SubjectSet(asvs=asvs)


    for ridx, row in enumerate(dfmeta.index):
        meta = dfmeta.loc[row]
        if not meta['isIncluded']:
            continue

        sid = str(int(meta['subjectID']))
        t = float(meta['measurementid'])

        if sid not in subjset:
            subjset.add(name=sid)
        
        subj = subjset[sid]
        subj.add_reads(timepoints=t, reads=np.asarray(list(dfcounts[row])))
        subj.add_qpcr(timepoints=t, qpcr=np.asarray(list(dfbiomass.loc[row])))
    return subjset

def parse_mdsine1_diet(basepath):
    '''There are no qPCR measurements here
    '''
    counts = basepath + 'counts.txt'
    dfcounts = pd.read_csv(counts, sep='\t')
    dfcounts = dfcounts.set_index('#OTU ID')

    metadata = basepath + 'metadata.txt'
    dfmeta = pd.read_csv(metadata, sep='\t')
    dfmeta = dfmeta.set_index('sampleID')


    asvs = pl.base.ASVSet(use_sequences=False)
    for row in dfcounts.index:
        asvs.add_asv(row)
    subjset = pl.base.SubjectSet(asvs=asvs)

    pert_start = None
    pert_end = None
    cont_pert = False
    for idx, row in enumerate(dfmeta.index):
        meta = dfmeta.loc[row]
        if meta['isIncluded']:

            sid = str(int(meta['subjectID']))
            perturbID = int(meta['perturbID'])
            t = float(meta['measurementID'])

            if sid not in subjset:
                subjset.add(name=sid)
            
            subj = subjset[sid]
            subj.add_reads(timepoints=t, reads=np.asarray(list(dfcounts[str(row)])))
            subj.add_qpcr(timepoints=t, qpcr=np.asarray([1]))

            # Add the perturbation if it is included
            if perturbID != 0:
                if not cont_pert:
                    cont_pert = True
                    pert_start = t
                    pert_end = t
                else:
                    pert_end = t
            else:
                if cont_pert:
                    # Finished looking at the perturbation
                    cont_pert = False
                    if subjset.perturbations is not None:
                        for pert in subjset.perturbations:
                            if pert_start == pert.start and pert_end == pert.end:
                                # dont need to add it because it is the same perturbations
                                pass
                            else:
                                subjset.add_perturbation(pert_start, end=pert_end)
                    else:
                        subjset.add_perturbation(pert_start, end=pert_end)
            

    return subjset

def make_data_like_mdsine1(subjset, basepath):
    '''Parse subjset into data format like mdsine1
    '''
    os.makedirs(basepath, exist_ok=True)

    n_times = 0 # for `#OTU ID`
    for subj in subjset:
        n_times += len(subj.times)

    metadata_columns = ['sampleID', 'isIncluded', 'subjectID', 'measurementid', 'perturbid']
    

    # Make the counts
    counts_columns = ['{}'.format(i+1) for i in range(n_times)]
    count_Ms = []
    for subj in subjset:
        d = subj.matrix()
        M = d['raw']
        count_Ms.append(M)
    
    counts = np.hstack(count_Ms)
    df_counts = pd.DataFrame(counts, columns=counts_columns, index=[asv.name for asv in subjset.asvs])
    df_counts.index = df_counts.index.rename('#OTU ID')
    df_counts.to_csv(basepath + 'counts.txt', sep='\t', header=True, index=True)

    # Make the biomass
    biomass_columns = ['mass1', 'mass2', 'mass3']
    qs = []
    for subj in subjset:
        for t in subj.times:
            qs.append(subj.qpcr[t].data)
    biomass = np.vstack(qs)
    df_biomass = pd.DataFrame(data=biomass, columns=biomass_columns, index=counts_columns)
    df_biomass.to_csv(basepath + 'biomass.txt', sep='\t', header=True, index=False, float_format='%.2E')
    

    # Make metadata
    sampleID = counts_columns
    is_included = [1 for i in range(len(sampleID))]
    subjectID = []
    measurementid = []
    perturbid = []
    for subjidx, subj in enumerate(subjset):
        for t in subj.times:
            subjectID.append(int(subj.name))
            measurementid.append(t)

            p = 0
            for pidx, pert in enumerate(subjset.perturbations):
                if t >= pert.start and t <= pert.end:
                    p = pidx+1
                    break
            perturbid.append(p)

    print(perturbid)

    df_metadata = pd.DataFrame({
        'sampleID': sampleID, 
        'isIncluded': is_included, 
        'subjectID': subjectID,
        'measurementid': measurementid,
        'perturbid': perturbid})
    df_metadata.to_csv(basepath + 'metadata.txt', sep='\t', header=True, index=False)

def keystoneness(chain_fname, fname, outfile, max_posterior=None, mp=None):
    '''The file(s) show a list of asvs to delete, comma separated
    All of the asvs on a single line should be deleted at once. Note that 
    this is all a single process

    Parameters
    ----------
    chain_fname : str
        This is the location of the Pylab MCMC chain filename that is saved from inference
    fname : str
        This is the location of the file that describes which ASVs to be held out
    outfile : str
        This is the location where to print the output

    '''

    chain = pl.inference.BaseMCMC.load(chain_fname)
    subjset = chain.graph.data.subjects

    SECTION = 'posterior'
    if max_posterior is None:
        max_posterior = chain.n_samples - chain.burnin
    growth_master = chain.graph[names.STRNAMES.GROWTH_VALUE].get_trace_from_disk(section=SECTION)[:max_posterior, ...]
    si_master =  chain.graph[names.STRNAMES.SELF_INTERACTION_VALUE].get_trace_from_disk(section=SECTION)[:max_posterior, ...]
    A_master =  chain.graph[names.STRNAMES.INTERACTIONS_OBJ].get_trace_from_disk(section=SECTION)[:max_posterior, ...]

    print(growth_master.shape)

    dyn = model.gLVDynamicsSingleClustering(asvs=subjset.asvs, log_dynamics=True, 
        perturbations_additive=False)
    dyn.growth = growth_master
    dyn.self_interactions = si_master
    dyn.interactions = A_master

    df = subjset.df(dtype='abs', agg='mean', times='union')
    initial_conditions = df[0.5].to_numpy()

    for i in range(len(initial_conditions)):
        if initial_conditions[i] == 0:
            initial_conditions[i] = pl.random.truncnormal.sample(mean=1e5, std=1e5, low=1e2)
    initial_conditions = initial_conditions.reshape(-1,1)

    days = 20
    sim_dt = 0.01
    BASE_CONCENTRATIONS = np.zeros(shape=growth_master.shape, dtype=float)
    # Generate the base concentrations

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
            if i %20 == 0:
                print('{}/{}: {}'.format(i,growth_master.shape[0], time.time()-start_time))
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
        names_to_del_lst.append(tuple(lst))
    
    for names_iii, names_to_del in enumerate(names_to_del_lst):
        idxs_to_del = [subjset.asvs[name].idx for name in names_to_del]
        # Take out asv aidxs and do the forward simulation
        print('{}/{}: {}'.format(names_iii, len(args), names_to_del))

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
                print('\t{}/{}'.format(i, growth_master.shape[0]))

        diff = concentrations - BASE_CONCENTRATIONS[:,mask]
        mean_diff = np.mean(diff, axis=0)
        dists[names_to_del] = np.sqrt(np.sum(np.square(mean_diff)))

    idxs = (np.argsort(list(dists.values())))[::-1]
    keys = list(dists.keys())

    f = open(outfile, 'w')

    f.write('Concise results\n')
    for i, idx in enumerate(idxs):
        f.write('{}: {} (was {} on bfs)\n'.format(i+1, keys[idx], idx+1))

    f.write('Spearman correlation on ranking: {}\n'.format(
        scipy.stats.spearmanr(idxs, np.arange(len(idxs)))[0]))

    f.write('expanded results')
    for i, idx in enumerate(idxs):
        
        names_ = keys[idx]
        f.write('\n\n---------------------------------------------\n{}\n'.format(names_))
        temp_asvs = [subjset.asvs[name] for name in names_]
        for asv in temp_asvs:
            f.write('{}\n'.format(str(asv)))

        f.write('Effect: {:.4E}\n'.format(dists[names_]))

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