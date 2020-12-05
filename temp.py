import mdsine2 as md2
import pandas as pd
import logging

md2.LoggingConfig(level=logging.INFO)
study = md2.Study.load('processed_data/gibson_healthy_agg.pkl')

# for otu in study.taxas:

#     if md2.isotu(otu):
#         print(otu.name)
#         data = []
#         for asvname in otu.aggregated_taxas:
#             temp = []
#             for k in otu.aggregated_taxonomies[asvname]:
#                 temp.append(otu.aggregated_taxonomies[asvname][k])
#             data.append(temp)

#         df = pd.DataFrame(data, columns=list(otu.aggregated_taxonomies[asvname].keys()))
#         print(df)
#         print(list(otu.taxonomy.values()))

# study.taxas.generate_consensus_taxonomies()

from mdsine2.names import STRNAMES
import mdsine2 as md2
mcmc = md2.BaseMCMC.load(...)

taxas = mcmc.graph.data.taxas

growth = mcmc.graph[STRNAMES.GROWTH_VALUE].get_trace_from_disk() # np.ndarray(n_gibbs, n_taxa)
summ = md2.summary(growth) # dict 'mean', 'median', '75th percentile' '25th percentile'

# Cocluster matrix
clustering = mcmc.graph[STRNAMES.CLUSTERING_OBJ]
cocluster_trace = clustering.coclusters.get_trace_from_disk() # np.ndarray (n_gibbs, n_taxa, n_taxa)
coclusters = md2.summary(cocluster_trace)['mean']

# Interactions
interactions = mcmc.graph[STRNAMES.INTERACTIONS_OBJ]
interactions_trace = interactions.get_trace_from_disk() # np.ndarray (n_gibbs, n_taxa, n_taxa)
interactions_value = md2.summary(interactions_trace, set_nan_to_0=True)

# bayes factors
bf = md2.generate_interation_bayes_factors_posthoc(mcmc=mcmc, section='posterior') # np.ndarray (n_gibbs, n_taxa, n_taxa)

# Only show interactions with bf > 10
interactions_value[bf < 10] = 0



def _integrate_glv_no_perturbations_no_processvar_fast(initial_conditions, growth, interactions, 
    sim_max, dt, times):
    '''Integrate gLV dynamics with no perturbations
    '''
    '''Integrate gLV dynamics with no process variance. This has a faster execution time
    than calling mdsine2.integrate

    Parameters
    ----------
    initial_conditions : np.ndarray
        These are the initial conditions for each taxa
    growth : np.ndarray
        These are the growth rates
    interactions : np.ndarray
        Square array of the interaction matrix
    dt : 

    '''
    times = np.sort(times)
    n_days = times[-1]
    times_tmp = np.arange(n_days+dt, step=dt)
    ret = np.zeros(shape=(len(times_tmp), len(growth)))
    ret[0,:] = initial_conditions.ravel()
    dtgrowth = growth.ravel() * dt
    dtinteractions = interactions * dt

    prev_logx = np.log(ret[0,:])
    for i in range(1,ret.shape[0]):
        x = ret[i-1, :]
        prev_logx = prev_logx + (dtgrowth + dtinteractions.dot(x))
        ret[i, :] = np.exp(prev_logx)

        if np.any(ret[i] >= sim_max):
            print('mer')

    # Subsample times
    idxs = []
    for t in times:
        tidx = np.searchsorted(times_tmp, t)
        idxs.append(tidx)
    ret = ret[np.asarray(idxs), :]
    ret = ret.T
    return ret, times