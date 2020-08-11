'''Implement the Maximum Likelihood Constrained Ridge Regression specified in
the original MDSINE [1] paper.

[1] B. Vanni, et al., "Mdsine: Microbial dynamical systems inference engine for 
    microbiome time-series analysis," Genome Biology, 17(1):121, 2016.
'''

import logging
import numpy as np
import sys
import os
import os.path
import pickle
import pandas
import argparse

import matplotlib.pyplot as plt
import seaborn

import pylab as pl

import config
import main_ICML

sys.path.append('..')
import model
import synthetic
import names
import preprocess_filtering as filtering
import data
import main_base

STRNAMES = names.STRNAMES
REPRNAMES = names.REPRNAMES

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-seed', '-d', type=int,
        help='Seed to initialize the data',
        dest='data_seed')
    parser.add_argument('--init_seed', '-i', type=int,
        help='Seed to initialize the inference',
        dest='init_seed')
    parser.add_argument('--basepath', '-b', type=str,
        help='Folder to save the output', default='output_mlcrr/',
        dest='basepath')
    parser.add_argument('--data-path', '-db', type=str,
        help='Folder to lead the data from', dest='data_path')
    parser.add_argument('--healthy', '-hy', type=bool,
        help='Whether or not to use the healthy patients or not',
        default=False, dest='healthy')
    parser.add_argument('--measurement-noise-level', '-m', type=float,
        help='Amount of measurement noise for the counts',
        default=0.1, dest='measurement_noise_level')
    parser.add_argument('--process-variance-level', '-p', type=float,
        help='Amount of process variance',
        default=0.05, dest='process_variance_level')
    parser.add_argument('--n-asvs', '-n', type=int,
        help='Number of ASVs', default=20,
        dest='n_asvs')
    parser.add_argument('--n-times', '-nt', type=int,
        help='Number of time points', 
        dest='n_times')
    parser.add_argument('--n-replicates', '-nr', type=int,
        help='Number of replicates to use for the data of inference',
        dest='n_replicates', default=5)
    parser.add_argument('--n-cpus', '-c', type=int,
        help='Number of cpus to use for the data of inference',
        dest='n_cpus', default=1)
    parser.add_argument('--uniform-sampling', '-us', type=int,
        help='Whether or not to use uniform sampling or not',
        default=0, dest='uniform_sampling')

    return parser.parse_args()

def make_comparison(syndata_filename, exact_subjset):
    '''Create a graph and data that has all the true values in it.

    Needs to all be in the same graph and needs to have the same variable names
    as in the inference graph.
    '''
    synth = synthetic.SyntheticData.load(syndata_filename)

    # Set the variables in the graph
    GRAPH = pl.graph.Graph(name='synthetic')
    d = data.Data(asvs=exact_subjset.asvs, subjects=exact_subjset, G=GRAPH, 
        data_logscale=True)
    n_asvs = d.n_asvs

    growth = pl.variables.Variable(G=GRAPH, name=STRNAMES.GROWTH_VALUE, 
        shape=(n_asvs,))
    growth.value = synth.dynamics.growth

    self_interactions = pl.variables.Variable(G=GRAPH, name=STRNAMES.SELF_INTERACTION_VALUE,
        shape=(n_asvs,))
    self_interactions.value = synth.dynamics.self_interactions

    interactions = pl.variables.Variable(G=GRAPH, name=STRNAMES.CLUSTER_INTERACTION_VALUE, 
        shape=(n_asvs, n_asvs))
    interactions.value = synth.dynamics.interactions.get_datalevel_value_matrix(
        set_neg_indicators_to_nan=False)

    if synth.dynamics.perturbations is not None:
        for synth_perturbation in synth.dynamics.perturbations:
            perturbation = pl.contrib.Perturbation( 
                start=synth_perturbation.start, 
                end=synth_perturbation.end,
                name=synth_perturbation.name,
                G=GRAPH, asvs=exact_subjset.asvs)
            indicators = synth_perturbation.indicator.item_bool_array()
            values = synth_perturbation.item_array(only_pos_ind=True)
            iii = 0
            for oidx in range(n_asvs):
                if indicators[oidx]:
                    perturbation.indicator.value[oidx] = True
                    perturbation.magnitude.value[oidx] = values[iii]
                    iii += 1
                else:
                    perturbation.indicator.value[oidx] = False
                    perturbation.magnitude.value[oidx] = 0
    
    # set the inference
    mlcrr = pl.inference.MLRR(constrain=True, graph=GRAPH)
    return mlcrr

def run(params, graph_name, subjset):
    '''Run MLCRR
    '''
    # Initialize graph and data
    if pl.isstr(params):
        params = config.MLCRRConfig.load(params)
    if pl.isstr(subjset):
        subjset = pl.base.SubjectSet.load(subjset)
    
    GRAPH = pl.graph.Graph(name=graph_name, seed=params.INIT_SEED)
    GRAPH.as_default()

    d = data.Data(asvs=subjset.asvs, subjects=subjset, 
        min_rel_abund=False, data_logscale=True, G=GRAPH, add_eps_to_data=True)
    n_asvs = len(subjset.asvs)

    # Initialize the variables
    growth = pl.variables.Variable(
        name=STRNAMES.GROWTH_VALUE, G=GRAPH, shape=(n_asvs,))
    self_interactions = pl.variables.Variable(
        name=STRNAMES.SELF_INTERACTION_VALUE, shape=(n_asvs, ), G=GRAPH)
    interactions = pl.variables.Variable(name=STRNAMES.CLUSTER_INTERACTION_VALUE, 
        shape=(n_asvs * (n_asvs - 1), ), G=GRAPH)
    if subjset.perturbations is not None:
        for pidx, subj_pert in enumerate(subjset.perturbations):
            perturbation = pl.contrib.Perturbation(start=subj_pert.start, 
                end=subj_pert.end, asvs=subjset.asvs, G=GRAPH, name=subj_pert.name)
        perturbations = pl.variables.Variable(name=STRNAMES.PERT_VALUE, 
            shape=(len(subjset.perturbations)*n_asvs, ), G=GRAPH)

    # Set up the name spaces
    REPRNAMES.set(G=GRAPH)

    # Set up inference and theta parameter order
    mlcrr = pl.inference.MLRR(constrain=True, graph=GRAPH)
    order = [
        STRNAMES.GROWTH_VALUE,
        STRNAMES.SELF_INTERACTION_VALUE,
        STRNAMES.CLUSTER_INTERACTION_VALUE]
    if subjset.perturbations is not None:
        order.append(STRNAMES.PERT_VALUE)
    mlcrr.set_parameter_order(order=order)

    # Set up the constraints
    # We want the constraints so that growth is always positive and self-interactions
    # are always positive. interactions and perturbations are unbounded
    diag = np.append(np.full(2*n_asvs, -1), np.ones(n_asvs*(n_asvs-1)))
    if subjset.perturbations is not None:
        diag = np.append(diag, np.ones(n_asvs * len(subjset.perturbations)))
    G = np.diag(diag)
    h = np.append(
        np.zeros(2*n_asvs), 
        np.full(G.shape[0]-2*n_asvs, float('inf'))).reshape(-1,1)
    mlcrr.set_constraints(G=G, h=h)

    # Set up penalties
    mlcrr.set_penalty(parameter=growth.id, penalty=1e-3)
    mlcrr.set_penalty(parameter=self_interactions.id, penalty=1e-3)
    mlcrr.set_penalty(parameter=interactions.id, penalty=1e-3)
    if subjset.perturbations is not None:
        mlcrr.set_penalty(parameter=perturbations.id, penalty=1e-3)

    # Set up the design matrices
    growthDM = data.GrowthDesignMatrix(G=GRAPH, name='growth_design_matrix', 
        data_logscale=True, perturbations_additive=True)
    selfinteractionsDM = data.SelfInteractionDesignMatrix(G=GRAPH,
        name='self_interactions_design_matrix',
        data_logscale=True)
    interactionsDM = data.InteractionsBaseDesignMatrix(G=GRAPH, data_logscale=True,
        name=STRNAMES.CLUSTER_INTERACTION_VALUE)
    if subjset.perturbations is not None:
        perturbationsDM = data.PerturbationBaseDesignMatrix(data_logscale=True, additive=True, 
            name=STRNAMES.PERT_VALUE, G=GRAPH)
        perturbationsDM.build()

    growthDM.build()
    selfinteractionsDM.build()
    interactionsDM.build()

    # Set up observation matrices
    lhs = data.LHSVector(G=GRAPH, data_logscale=True, name='lhs_vector')
    lhs.build()

    y = d.construct_lhs()
    rhs = [
        REPRNAMES.GROWTH_VALUE, 
        REPRNAMES.SELF_INTERACTION_VALUE,
        REPRNAMES.CLUSTER_INTERACTION_VALUE]
    if subjset.perturbations is not None:
        rhs.append(REPRNAMES.PERT_VALUE)
    X = d.construct_rhs(keys=rhs, toarray=True)

    mlcrr.run_single(X=X, y=y)

    # Set interactions as a matrix
    value = np.zeros(shape=(n_asvs, n_asvs), dtype=float)
    iii = 0
    for i in range(n_asvs):
        for j in range(n_asvs):
            if i == j:
                value[i,j] = np.nan
                continue
            value[i,j] = interactions.value[iii]
            iii += 1
    interactions.value = value

    # Divide up the perturbations
    i = 0
    for perturbation in GRAPH.perturbations:
        perturbation.indicator.value[:] = True
        perturbation.magnitude.value = perturbations[i:i+n_asvs]
        i += n_asvs

    mlcrr.save(config.MLCRR_RESULTS_FILENAME)
    GRAPH.save(config.GRAPH_FILENAME)

    return mlcrr

def runCV(params, graph_name, subjset):
    '''Run MLCRR cross validation
    '''
    # Initialize graph and data
    if pl.isstr(params):
        params = config.MLCRRConfig.load(params)
    if pl.isstr(subjset):
        subjset = pl.base.SubjectSet.load(subjset)
    
    GRAPH = pl.graph.Graph(name=graph_name, seed=params.INIT_SEED)
    GRAPH.as_default()

    d = data.Data(asvs=subjset.asvs, subjects=subjset, 
        min_rel_abund=False, data_logscale=True, G=GRAPH)
    n_asvs = len(subjset.asvs)

    # Initialize the variables
    growth = pl.variables.Variable(
        name=STRNAMES.GROWTH_VALUE, G=GRAPH, shape=(n_asvs,))
    self_interactions = pl.variables.Variable(
        name=STRNAMES.SELF_INTERACTION_VALUE, shape=(n_asvs, ), G=GRAPH)
    interactions = pl.variables.Variable(name=STRNAMES.CLUSTER_INTERACTION_VALUE, 
        shape=(n_asvs * (n_asvs - 1), ), G=GRAPH)
    if subjset.perturbations is not None:
        for pidx, subj_pert in enumerate(subjset.perturbations):
            perturbation = pl.contrib.Perturbation(start=subj_pert.start, 
                end=subj_pert.end, asvs=subjset.asvs, G=GRAPH, name=subj_pert.name)
        perturbations = pl.variables.Variable(name=STRNAMES.PERT_VALUE, 
            shape=(len(subjset.perturbations)*n_asvs, ), G=GRAPH)

    # Set up the name spaces
    REPRNAMES.set(G=GRAPH)

    # Set up inference and theta parameter order
    mlcrr = pl.inference.MLRR(constrain=True, graph=GRAPH)
    order = [
        STRNAMES.GROWTH_VALUE,
        STRNAMES.SELF_INTERACTION_VALUE,
        STRNAMES.CLUSTER_INTERACTION_VALUE]
    if subjset.perturbations is not None:
        order.append(STRNAMES.PERT_VALUE)
    mlcrr.set_parameter_order(order=order)

    # Set up the constraints
    # We want the constraints so that growth is always positive and self-interactions
    # are always positive. interactions and perturbations are unbounded
    diag = np.append(np.full(2*n_asvs, -1), np.ones(n_asvs*(n_asvs-1)))
    if subjset.perturbations is not None:
        diag = np.append(diag, np.ones(n_asvs * len(subjset.perturbations)))
    G = np.diag(diag)
    h = np.append(
        np.zeros(2*n_asvs), 
        np.full(G.shape[0]-2*n_asvs, float('inf'))).reshape(-1,1)
    mlcrr.set_constraints(G=G, h=h)

    # Set up the design matrices
    growthDM = data.GrowthDesignMatrix(G=GRAPH, name='growth_design_matrix', 
        data_logscale=True, perturbations_additive=True)
    selfinteractionsDM = data.SelfInteractionDesignMatrix(G=GRAPH,
        name='self_interactions_design_matrix',
        data_logscale=True)
    interactionsDM = data.InteractionsBaseDesignMatrix(G=GRAPH, data_logscale=True,
        name=STRNAMES.CLUSTER_INTERACTION_VALUE)
    if subjset.perturbations is not None:
        perturbationsDM = data.PerturbationBaseDesignMatrix(data_logscale=True, 
            perturbations_additive=True, name=STRNAMES.PERT_VALUE, G=GRAPH)
        perturbationsDM.build()

    growthDM.build()
    selfinteractionsDM.build()
    interactionsDM.build()

    # Set up observation matrices
    lhs = data.LHSVector(G=GRAPH, data_logscale=True, name='lhs_vector')
    lhs.build()

    y = d.construct_lhs()
    rhs = [
        REPRNAMES.GROWTH_VALUE, 
        REPRNAMES.SELF_INTERACTION_VALUE,
        REPRNAMES.CLUSTER_INTERACTION_VALUE]
    if subjset.perturbations is not None:
        rhs.append(REPRNAMES.PERT_VALUE)
    X = d.construct_rhs(keys=rhs, toarray=True)

    # Index out each subject Xs
    Xs = []
    ys = []

    i = 0
    for ridx in range(d.n_replicates):
        n_dts = d.n_dts_for_replicate[ridx]
        l = n_dts * n_asvs
        X_temp = X[i:i+l,:]
        y_temp = y[i:i+l,:]

        rows_to_keep = []
        for rowidx in range(len(y_temp)):
            if np.all(np.isfinite(y_temp[rowidx])):
                rows_to_keep.append(rowidx)
        X_temp = X_temp[rows_to_keep, :]
        y_temp = y_temp[rows_to_keep, :]

        Xs.append(X_temp)
        ys.append(y_temp)
        i += l

    # Set up workers
    n_perts = None
    if subjset.perturbations is not None:
        n_perts = len(subjset.perturbations)
    
    # Set up penalty meshes:
    growth_penalty_mesh = np.logspace(start=params.CV_MAP_MIN, stop=params.CV_MAP_MAX, 
        num=params.CV_MAP_N)
    interaction_penalty_mesh = np.logspace(start=params.CV_MAP_MIN, stop=params.CV_MAP_MAX, 
        num=params.CV_MAP_N)

    total = params.CV_MAP_N * params.CV_MAP_N
    if subjset.perturbations is not None:
        perturbation_penalty_mesh = np.logspace(start=params.CV_MAP_MIN, stop=params.CV_MAP_MAX, 
            num=params.CV_MAP_N)
        total *= params.CV_MAP_N

    total *= d.n_replicates

    if params.N_CPUS is not None:
        pool = pl.multiprocessing.PersistentPool(ptype='dasw')
        for i in range(params.N_CPUS):
            pool.add_worker(MLCRRWorker(Xs=Xs, ys=ys, 
                G=G, h=h, n_asvs=n_asvs, n_perts=n_perts, total=total))
        pool.staged_map_start('sim')        

    # calculate
    reverse_index = {}
    iii = 0
    for i, gp in enumerate(growth_penalty_mesh):
        reverse_index[gp] = i
        for ip in interaction_penalty_mesh:
            for lo in range(d.n_replicates):

                if subjset.perturbations is not None:
                    for pp in perturbation_penalty_mesh:
                        kwargs = {'gp': gp, 'ip': ip, 'pp': pp, 'leave_out': lo, 'i': iii}
                        pool.staged_map_put(kwargs)
                        iii += 1
                else:
                    kwargs = {'gp': gp, 'ip': ip, 'pp': None, 'leave_out': lo, 'i': iii}
                    pool.staged_map_put(kwargs)
                    iii += 1

    ret = pool.staged_map_get()
    pool.kill()

    # Find the minimum - take the geometric mean of the leave outs
    if subjset.perturbations is not None:
        arr = np.ones(shape=(params.CV_MAP_N,params.CV_MAP_N,params.CV_MAP_N, d.n_replicates))*np.nan
        for (gp, ip, pp, iii, rmse_loss) in ret:
            i = reverse_index[gp]
            j = reverse_index[ip]
            k = reverse_index[pp]
            arr[i,j,k, iii] = rmse_loss
    else:
        arr = np.ones(shape=(params.CV_MAP_N,params.CV_MAP_N, d.n_replicates))*np.nan
        for (gp, ip, pp, iii, rmse_loss) in ret:
            i = reverse_index[gp]
            j = reverse_index[ip]
            arr[i,j, iii] = rmse_loss

    # Multiply over the last axis
    arr = np.nanprod(arr, axis=-1)

    # Get the index of the N dimensional arr of the lowest
    ind = np.unravel_index(np.nanargmin(arr, axis=None), arr.shape)
    gp = growth_penalty_mesh[ind[0]]
    ip = interaction_penalty_mesh[ind[1]]

    D = np.append(np.full(n_asvs, gp), np.full(n_asvs**2, ip))
    if subjset.perturbations is not None:
        pp = perturbation_penalty_mesh[ind[2]]
        D = np.append(D, np.full(n_perts*n_asvs, pp))

    # Get the parameters
    D = np.diag(D)
    y = np.vstack(ys)
    X = np.vstack(Xs)

    mlcrr.run_single(X=X, y=y, D=D)

    # Set interactions as a matrix
    value = np.zeros(shape=(n_asvs, n_asvs), dtype=float)
    iii = 0
    for i in range(n_asvs):
        for j in range(n_asvs):
            if i == j:
                value[i,j] = np.nan
                continue
            value[i,j] = interactions.value[iii]
            iii += 1
    interactions.value = value

    # Divide up the perturbations
    i = 0
    for perturbation in GRAPH.perturbations:
        perturbation.indicator.value[:] = True
        perturbation.magnitude.value = perturbations[i:i+n_asvs]
        i += n_asvs

    mlcrr.save(config.MLCRR_RESULTS_FILENAME)
    GRAPH.save(config.GRAPH_FILENAME)
    return mlcrr

class MLCRRWorker(pl.multiprocessing.PersistentWorker):

    def __init__(self, Xs, ys, G, h, n_asvs, n_perts, total):
        self.Xs = Xs
        self.ys = ys
        self.G = G
        self.h = h
        self.n_asvs = n_asvs
        self.n_perts = n_perts
        self.total = total

    def sim(self, gp, ip, pp, leave_out,i):
        '''Set the penalty for each of the parameters and specify which replicate to
        leave out

        Parameters
        ----------
        gp : float
            growth penalty
        ip : float
            interaction penalty (includes self-interactions)
        pp : float
            perturbation penalty
        leave_out : int
            Which replicate index to leave out

        Returns
        -------
        (leave_out, (gp, ip, pp), loss)
        '''

        try:
            iiii = i

            Xs = None
            ys = None
            if len(self.Xs) == 1:
                X = self.Xs[0]
                y = self.ys[0]
            else:
                for i in range(len(self.Xs)):
                    if i == leave_out:
                        continue
                    if Xs is None:
                        Xs = (self.Xs[i],)
                        ys = (self.ys[i],)
                    else:
                        Xs += (self.Xs[i],)
                        ys += (self.ys[i],)

                y = np.vstack(ys)
                X = np.vstack(Xs)
            
            D = np.append(np.full(self.n_asvs, gp), np.full(self.n_asvs**2, ip))
            if self.n_perts is not None:
                D = np.append(D, np.full(self.n_asvs*self.n_perts, pp))
            D = np.diag(D)
            
            theta = pl.inference.MLRR.run_MLCRR(X=X, y=y, D=D, G=self.G, 
                h=self.h, A=None, b=None).reshape(-1,1)

            loss = y - X @ theta
            rmse_loss = np.sqrt(np.sum(np.square(loss)))
            print('{}/{}: {:.3E}'.format(iiii, self.total, rmse_loss))

        except Exception as e:
            print(e)
            rmse_loss = np.nan

        return (gp, ip, pp, leave_out, rmse_loss)


if __name__ == '__main__':
    '''Build the parameters, model, and run the cross validation
    '''
    args = parse_args()
    ONLY_PLOT = True

    # Generate parameters
    synparams = config.SimulationConfig(times=args.n_times, n_replicates=args.n_replicates,
        n_asvs=args.n_asvs, healthy=args.healthy, 
        process_variance_level=args.process_variance_level,
        measurement_noise_level=args.measurement_noise_level, 
        uniform_sampling_timepoints=args.uniform_sampling)
    lparams = config.LoggingConfig()
    params = config.MLCRRConfig(output_basepath=args.basepath, data_seed=args.data_seed,
        init_seed=args.init_seed, n_cpus=args.n_cpus, data_path=args.data_path)

    logging.info('Init settings')
    logging.info('Data seed: {}'.format(args.data_seed))
    logging.info('Init seed: {}'.format(args.init_seed))
    logging.info('measurement noise level: {}'.format(args.measurement_noise_level))
    logging.info('process variance level: {}'.format(args.process_variance_level))
    logging.info('basepath: {}'.format(args.basepath))
    logging.info('n_asvs: {}'.format(args.n_asvs))

    pl.seed(params.DATA_SEED)
    graph_name = 'graph' + params.suffix() + synparams.suffix()
    basepath = params.OUTPUT_BASEPATH + graph_name + '/'
    os.makedirs(basepath, exist_ok=True)

    subjset_filename = basepath + config.SUBJSET_FILENAME
    validate_subjset_filename = basepath + config.VALIDATION_SUBJSET_FILENAME
    mlcrr_filename = basepath + config.MLCRR_RESULTS_FILENAME
    graph_filename = basepath + config.GRAPH_FILENAME
    params_filename = basepath + config.PARAMS_FILENAME
    fparams_filename = basepath + config.FPARAMS_FILENAME
    syndata_filename = basepath + config.SYNDATA_FILENAME
    synparams_filename = basepath + config.SYNPARAMS_FILENAME

    if not ONLY_PLOT:
        params.save(params_filename)
        synparams.save(synparams_filename)

        # Get the steady states of the real data
        syndata = synthetic.SyntheticData.load(config.make_syndata_base_name(
            basepath=params.DATA_PATH, dset='icml', ds=params.DATA_SEED))
        subjset = pl.base.SubjectSet.load(config.make_subj_name(
            basepath=params.DATA_PATH, data_seed=params.DATA_SEED, n_asvs=args.n_asvs, 
            process_variance=args.process_variance_level, measurement_noise=args.measurement_noise_level,
            n_replicates=args.n_replicates, n_times=args.n_times))

        subjset.save(subjset_filename)
        syndata.save(syndata_filename)

        # Plot
        for i, subj in enumerate(subjset):
            ax = pl.visualization.abundance_over_time(subj=subj, dtype='abs', legend=True,
                taxlevel=None, set_0_to_nan=True, yscale_log=True, lca=False,
                color_code_clusters=True, clustering=syndata.dynamics.clustering)
            plt.savefig(basepath + 'data{}.pdf'.format(i))
        pl.visualization.abundance_over_time(subj=subjset, dtype='qpcr', taxlevel=None, 
            set_0_to_nan=True, yscale_log=True, clustering=syndata.dynamics.clustering)
        plt.savefig(basepath + 'qpcr.pdf')
        pl.visualization.abundance_over_time(subj=subjset, dtype='read-depth', taxlevel=None, 
            set_0_to_nan=True, yscale_log=True, clustering=syndata.dynamics.clustering)
        plt.savefig(basepath + 'read_depth.pdf')

        for pert in syndata.dynamics.perturbations:
            print(pert)
        # sys.exit()

        # Rescale if necessary
        if params.QPCR_NORMALIZATION_MAX_VALUE is not None:
            subjset.normalize_qpcr(max_value=params.QPCR_NORMALIZATION_MAX_VALUE)
            logging.info('Normalizing qPCR values. Normalization constant: {:.3E}'.format(
                subjset.qpcr_normalization_factor))

        # Run the model
        results = runCV(params=params, subjset=subjset_filename, graph_name=graph_name)
        results.save(mlcrr_filename)
        params.save(params_filename)

    params = config.MLCRRConfig.load(params_filename)
    mlcrr_result = pl.inference.MLRR.load(mlcrr_filename)
    
    noise_subjset = pl.base.SubjectSet.load(config.make_val_subjset_name(
        basepath=params.DATA_PATH, ds=params.DATA_SEED, 
        pv=args.process_variance_level, mn=args.measurement_noise_level,
        nt=args.n_times, uniform_sampling_timepoints=args.uniform_sampling))
    exact_subjset = pl.base.SubjectSet.load(config.make_val_subjset_name(
        basepath=params.DATA_PATH, ds=params.DATA_SEED, 
        pv=args.process_variance_level))

    noise_subjset.save(basepath + config.VALIDATION_SUBJSET_FILENAME)
    comparison = make_comparison(syndata_filename, exact_subjset=exact_subjset)

    main_base.validate(
        src_basepath=basepath, model=mlcrr_result, 
        forward_sims=['sim-full'],
        yscale_log=True, run_on_copy=True,
        asv_prefix_formatter='%(index)s: (%(name)s)',
        yticklabels='(%(name)s): %(index)s',
        perturbations_additive=True,
        uniform_sample_timepoints=bool(args.uniform_sampling),
        mp=params.N_CPUS, comparison=comparison,
        traj_error_metric=pl.metrics.logPE,
        pert_error_metric=pl.metrics.RMSE,
        interaction_error_metric=pl.metrics.RMSE,
        growth_error_metric=pl.metrics.logPE,
        si_error_metric=pl.metrics.logPE,
        traj_fillvalue=params.C_M/2,
        clus_error_metric=None)

    

