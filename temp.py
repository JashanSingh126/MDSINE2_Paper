import mdsine2 as md2
import pandas as pd
import logging

md2.LoggingConfig(level=logging.INFO)
study = md2.Study.load('processed_data/gibson_healthy_agg.pkl')

for otu in study.taxas:

    if md2.isotu(otu):
        print(otu.name)
        data = []
        for asvname in otu.aggregated_taxas:
            temp = []
            for k in otu.aggregated_taxonomies[asvname]:
                temp.append(otu.aggregated_taxonomies[asvname][k])
            data.append(temp)

        df = pd.DataFrame(data, columns=list(otu.aggregated_taxonomies[asvname].keys()))
        print(df)
        print(list(otu.taxonomy.values()))

study.taxas.generate_consensus_taxonomies()


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