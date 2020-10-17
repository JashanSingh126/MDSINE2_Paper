'''This module is used to validate the models that we run. Can run with either Maximum likelihood or
bayesian models from pylab.
'''

import logging
import random
import time
import sys
import os
import shutil
import h5py
import warnings
import pickle
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import pylab as pl
import synthetic
import model as model_module
import preprocess_filtering as filtering

from names import STRNAMES

class Metrics(pl.Saveable):
    '''A validation suit. Given a model (either maximum likelihod or bayesian), we can
    run simulation validation and store other metrics.

    Parameters
    ----------
    model : pylab.inference.BaseModel
        Model we are using for the forward simulation
    simulation_dt : float, optional
        This is the :math:`\Delta_t` we use for the forward simulation. It is the step
        size (in days) we use when forward simulating.
    output_dt : float, optional
        This is the sampling rate that we return the simulated trajectory
    limit_of_detection : float
        This is the noise threshold we use
    traj_fillvalue : float
        If not None, replace 0s in the trajectories with this number as to not have any errors
    log_integration : bool
        If True, we use log integration. If false we use first order Euler integration.
    perturbations_additive : bool
        If True, we set the perturbation effects to be additive. If False, they are multiplicative
        to the growth rates.
    mp : None, int
        If not None, this sets how many cpus we can use for the forward integration multiprocessing.
    sim_max : float
        max simulation limit
    '''

    def __init__(self, model, perturbations_additive, limit_of_detection=1e5, traj_fillvalue=None,
        simulation_dt=0.001, output_dt=1/8, log_integration=True, mp=None, sim_max=1e20):
        # Check the parameters
        if not pl.ismodel(model):
            raise TypeError('`model` ({}) must be a subtype of pylab.inference.BaseModel'.format( 
                type(model)))
        if not pl.isnumeric(simulation_dt):
            raise TypeError('`simulation_dt` ({}) must be a numeric'.format(
                type(simulation_dt)))
        if simulation_dt <= 0:
            raise ValueError('`simulation_dt` ({}) must be > 0'.format(
                simulation_dt))
        if not pl.isnumeric(output_dt):
            raise TypeError('`output_dt` ({}) must be a numeric'.format(
                type(output_dt)))
        if output_dt <= 0:
            raise ValueError('`output_dt` ({}) must be > 0'.format(
                output_dt))
        if not pl.isnumeric(limit_of_detection):
            raise TypeError('`limit_of_detection` ({}) must be a numeric'.format(
                type(limit_of_detection)))
        if limit_of_detection <= 0:
            raise ValueError('`limit_of_detection` ({}) must be > 0'.format(
                limit_of_detection))
        if not pl.isbool(log_integration):
            raise TypeError('`log_integration` ({}) must be a bool'.format(
                type(log_integration)))
        if not pl.isbool(perturbations_additive):
            raise TypeError('`perturbations_additive` ({}) must be a bool'.format(
                type(perturbations_additive)))
        if mp is not None:
            if not pl.isint(mp):
                raise TypeError('`mp` ({}) must be an int or None'.format(
                    type(mp)))
            if mp <= 0:
                raise ValueError('`mp` ({}) must be >= 1'.format(mp))
    
        self.traj_fillvalue = traj_fillvalue
        self.model = model
        self.simulation_dt = simulation_dt
        self.output_dt = output_dt
        self.limit_of_detection = limit_of_detection
        self.log_integration = log_integration
        self.perturbations_additive = perturbations_additive
        self.mp = mp
        self.sim_max = sim_max

        # Set inference section
        if pl.isMCMC(self.model):
            # if self.model.sample_iter < 100:
            #     raise ValueError('Not enough samples in chain for forward simulation')
            if self.model.sample_iter <= self.model.burnin:
                self._INF_SECTION = 'burnin'
            else:
                self._INF_SECTION = 'posterior'

        self.results = {}
        self.time_lookaheads = {}
        self.truth_metrics = None

    def add_truth_metrics(self, d):
        '''Add metrics about the system about how it performed relative to a ground truth.
        
        Parameters
        ----------
        d : dict
            Dictionary of metrics
        '''
        if not pl.isdict(d):
            raise TypeError('`d` ({}) must be a dict'.format(type(d)))
        self.truth_metrics = d

    def readify(self, f, asv_format):
        '''Return a representation. This is used when we are writting to a text file

        Parameters
        ----------
        f : writable file (_io.TextIOWrapper)
            File we're writing to
        
        Returns
        -------
        file
        '''
        for subjname in self.results:
            for simtype in self.results[subjname]:
                f.write('\n\n====================================================\n')
                f.write('====================================================\n')
                f.write('Graph: {}\n'.format(self.model.graph.name))
                f.write('Subject {}\n'.format(subjname))
                if len(simtype) == 1:
                    f.write('Full Simulation\n')
                else:
                    f.write('Partial Simulation\n\tStart: {}\n\tNumber of days: {}\n'.format(
                        simtype[1], simtype[2]))
                temp = self.results[subjname][simtype]

                error_name = temp['error-metric'].__name__.replace('_', ' ')

                if pl.isMCMC(self.model):
                    f.write('Mean total {} Error trajectories: '.format(error_name))
                    if temp['error-metric'] == pl.metrics.PE:
                        f.write('{:.3}\n'.format(np.nanmean(temp['error-total']*100)))
                        lll = '%'
                    else:
                        f.write('{:.5E}\n'.format(np.nanmean(temp['error-total'])))
                        if temp['dtype'] == 'abs':
                            lll = 'CFUs/g'
                        elif temp['dtype'] == 'raw':
                            lll = 'counts'
                        else:
                            lll = ''
                else:
                    f.write('Total {} Error trajectories: '.format(error_name))
                    if temp['error-metric'] == pl.metrics.PE:
                        f.write('{:.3f}\n'.format(temp['error-total']*100))
                        lll = '%'
                    else:
                        f.write('{:.5E}\n'.format(temp['error-total']))
                        if temp['dtype'] == 'abs':
                            lll = 'CFUs/g'
                        elif temp['dtype'] == 'raw':
                            lll = 'counts'
                        else:
                            lll = ''
                subject = temp['subject']
                f.write('Mean {} Error ASVs\n'.format(error_name))
                for oidx in range(len(subject.asvs)):
                    if pl.isMCMC(self.model):
                        err = np.nanmean(temp['error-asvs'][:,oidx])
                    else:
                        err = temp['error-asvs'][oidx]
                    if temp['error-metric'] == pl.metrics.PE:
                        f.write('\t{:.3f} '.format(err*100))
                    else:
                        f.write('\t{:.5E} '.format(err))
                    f.write('{} : {}\n'.format( lll, pl.asvname_formatter(
                            asv_format, asv=oidx, asvs=subject.asvs)))
        return f
        
    def sim_full(self, subject, dtype='abs', percentile=25, truth=None, error_metric='pe'):
        '''Simulate the whole trajectory of the subject. Starts at day 0
        and simulates until the end.

        If `truth` is not None, then `truth` is the ground truth and `subject` are the 
        noisy measurements around this ground truth.

        Parameters
        ----------
        subjset : pylab.base.Subject
            Subject we are validating on
        dtype : str
            type of data we are simulating on. Options:
                'abs' : Abundance of the trajectories
                'raw' : Counts of the trajectories
                'rel' : Relative abundance of the trajectories
        percentile : float
            This is the percentile to get the high (100-`percentile`) and low (`percentile`) 
            trajectories. This is only used if `self.model` is a bayesian model.
        truth : pylab.base.Subject
            This is the ground truth
        error_metric : callable, str
            This is the error metric to use to calculate the difference in the trajectories
            Options:
                callable(a1, a2) -> float
                    A callable object that we pass the parameters through
                'pe'
                    Percent error (pylab.metrics.PE)
                'rmse'
                    Root mean squared error (pylab.metrics.RMSE)
        '''
        return self._sim(subject=subject, dtype=dtype, start=0, n_days=None, simtype='sim-full', 
            percentile=percentile, truth=truth, error_metric=error_metric, save_at_finish=True,
            skip_if_start_dne=False)

    def sim_days(self, n_days, start, subject, dtype='abs', percentile=25, truth=None, 
        error_metric='pe'):
        '''Simulate for `n_days` starting at time `start`. Calculate error compared to
        subject `subject`.

        If `truth` is not None, then `truth` is the ground truth and `subject` are the 
        noisy measurements around this ground truth.

        Parameters
        ----------
        n_days : float
            Number of days to forward simulate for. number must be a mutliple of
            `simulation_dt`.
        start : float, str, array(float)
            float: Time to start. 
            If array, these are an array of times to start.
            If str, options:
                'perts-start'
                    For each perturbation, start to simulate at the start of each of 
                    the perturbation for `n_days`.
                'perts-end'
                    For each perturbation, start to simulate at the end of each of the 
                    perturbations for `n_days`
        subjset : pylab.base.Subject
            Subject we are validating on
        dtype : str
            type of data we are simulating on. Options:
                'abs' : Abundance of the trajectories
                'raw' : Counts of the trajectories
                'rel' : Relative abundance of the trajectories
        percentile : float
            This is the percentile to get the high (100-`percentile`) and low (`percentile`) 
            trajectories. This is only used if `self.model` is a bayesian model.
        truth : pylab.base.Subject
            This is the ground truth
        error_metric : callable, str
            This is the error metric to use to calculate the difference in the trajectories
            Options:
                callable(a1, a2) -> float
                    A callable object that we pass the parameters through
                'pe'
                    Percent error (pylab.metrics.PE)
                'rmse'
                    Root mean squared error (pylab.metrics.RMSE)

        Raises
        ------
        ValueError
            `start` is not in `subject.times`
        '''
        if pl.isstr(start):
            arr = []
            if subject.perturbations is None:
                raise ValueError('No perturbations')
            if start == 'perts-start':
                for perturbation in subject.perturbations:
                    arr.append(perturbation.start)
            elif start == 'perts-end':
                for perturbation in subject.perturbations:
                    arr.append(perturbation.end)
            else:
                raise ValueError('`start` ({}) not recognized'.format(start))
            start = arr

        elif pl.isnumeric(start):
            start = [start]
        elif not pl.isarray(start):
            raise TypeError('`start` ({}) must be a str, array, or a numeric'.format(type(start)))
        for s in start:
            self._sim(subject=subject, dtype=dtype, start=s, n_days=n_days, simtype='sim-days',
                percentile=percentile, truth=truth, error_metric=error_metric, save_at_finish=True,
                skip_if_start_dne=False)

    def _sim(self, subject, dtype, start, n_days, simtype, percentile, truth, error_metric, 
        save_at_finish, skip_if_start_dne):
        '''Inner function for simulating

        If `truth` is not None, then the `error_metric` is against that, not subject. If it is
        an MCMC chain, then we calculate the metrics over all of the iterations, not
        just the expected.
        
        Parameters
        ----------
        subject : pl.base.Subject
        dtype : str
        start : int
        n_days : int, None
        simtype : str
        percentile : float
        truth : pl.base.Subject
        error_metric : str, callable
        save_at_finish : bool
        '''
        START_START = start
        # Get the data and set the data 0 if necessary
        if not pl.issubject(subject):
            raise TypeError('`subject` ({}) must be of type pylab.base.Subject'.format(type(subject)))
        M = subject.matrix()[dtype]
        M_ = M
        M_[M_ == 0] = self.traj_fillvalue

        if not callable(error_metric):
            if not pl.isstr(error_metric):
                raise ValueError('`error_metric` ({}) must either be callable or a str'.format(
                    type(error_metric)))
            if error_metric == 'pe':
                error_metric = pl.metrics.PE
            elif error_metric == 'rmse':
                error_metric = pl.metrics.RMSE
            else:
                raise ValueError('`error_metric` ({}) not recognized'.format(error_metric))

        start_idx = np.searchsorted(subject.times, start)
        try:
            _start = subject.times[start_idx]
        except:
            logging.critical('`start` ({}) out of bounds in times ({}). returning None'.format(
                start, subject.times))
            return None
        if truth is not None:
            if not pl.issubject(truth):
                raise ValueError('`truth` ({}) must be a pylab.base.Subject'.format(type(truth)))
            M_truth = truth.matrix()[dtype]
            M_truth_ = M_truth
            M_truth_[M_truth_ == 0] = self.traj_fillvalue
            times_truth = truth.times

            start_idx_truth = np.searchsorted(truth.times, start)
            _start = times_truth[start_idx_truth]
            initial_conditions = M_truth[:, start_idx_truth]

            if _start != START_START and skip_if_start_dne:
                logging.warning('Start ({}) not found in times ({}) skipping'.format(_start, times_truth))
                return
        else:
            initial_conditions = M[:, start_idx]
            if _start != START_START and skip_if_start_dne:
                logging.warning('Start ({}) not found in times ({}) skipping'.format(_start, subject.times))
                return
        for i,val in enumerate(initial_conditions):
            if val == 0:
                initial_conditions[i] = pl.random.truncnormal.sample(mean=self.limit_of_detection,
                    std=1e-2, low=0)
        initial_conditions = initial_conditions.reshape(-1,1)

        if n_days is None:
            given_times = subject.times[start_idx:]
            n_days = given_times[-1] - _start
            _end_subj_times = given_times
            M = M[:, start_idx:]
            M_ = M_[:, start_idx:]
            if truth is not None:
                M_truth = M_truth[:, start_idx_truth:]
                given_times = truth.times[start_idx_truth:]
                _end_truth_times = given_times
                n_days = _end_truth_times[-1] - _start
        else:
            end = start + n_days
            end_idx = np.searchsorted(subject.times, end)
            given_times = subject.times[start_idx:end_idx+1]
            _end_subj_times = given_times
            M = M[:, start_idx:end_idx+1]
            M_ = M_[:, start_idx:end_idx+1]
            if truth is not None:
                end_idx_truth = np.searchsorted(truth.times, end)
                M_truth = M_truth[:, start_idx_truth:end_idx_truth+1]
                given_times = truth.times[start_idx_truth:end_idx_truth+1]
                _end_truth_times = given_times

        n_days_total = given_times[-1] - given_times[0]
        end = given_times[-1]
        sim_times = np.arange(0, n_days_total + self.output_dt, step=self.output_dt)
        sim_times = np.sort(np.unique(np.append(sim_times, given_times)))

        # Get the parameters
        if pl.isML(self.model):
            # Maximum likelihood point estimate
            growth = self.model.graph[STRNAMES.GROWTH_VALUE].value.reshape(-1,1)
            self_interactions = self.model.graph[STRNAMES.SELF_INTERACTION_VALUE].value.reshape(-1,1)
            interactions = self.model.graph[STRNAMES.CLUSTER_INTERACTION_VALUE].value.reshape(
                len(subject.asvs), len(subject.asvs))
            perturbations = None
            if self.model.graph.perturbations is not None:
                perturbations = []
                for pert in self.model.graph.perturbations:
                    perturbations.append((pert.start, pert.end, pert.magnitude.value))
            pred_shape = (len(subject.asvs), len(sim_times))
        else:
            # MCMC chain
            growth = self.model.graph[STRNAMES.GROWTH_VALUE].get_trace_from_disk(
                section=self._INF_SECTION)
            self_interactions = self.model.graph[STRNAMES.SELF_INTERACTION_VALUE].get_trace_from_disk(
                section=self._INF_SECTION)
            interactions = self.model.graph[STRNAMES.INTERACTIONS_OBJ].get_trace_from_disk(
                section=self._INF_SECTION)
            perturbations = None
            if self.model.graph.perturbations is not None:
                perturbations = []
                for pert in self.model.graph.perturbations:
                    perturbations.append((pert.start, pert.end, 
                        pert.get_trace_from_disk(section=self._INF_SECTION)))
            pred_shape = (growth.shape[0], len(subject.asvs), len(sim_times))

        X_pred = np.zeros(shape=pred_shape)

        if pl.isML(self.model):
            # Using a maximum likelihood model, generate a point estimate
            pred_dyn = model_module.gLVDynamicsSingleClustering(asvs=subject.asvs, log_dynamics=self.log_integration,
                perturbations_additive=True, sim_max=self.sim_max, start_day=_start)
            pred_dyn.growth = growth
            pred_dyn.self_interactions = self_interactions
            pred_dyn.interactions = interactions
            pred_dyn.perturbations = perturbations

            _d = pl.dynamics.integrate(dynamics=pred_dyn, 
                initial_conditions=initial_conditions,
                dt=self.simulation_dt, n_days=sim_times[-1]+self.output_dt,
                subsample=True, times=sim_times, log_every=None)
            _d['X'][np.isnan(_d['X'])] = sys.float_info.max
            pred_traj = _d['X']

            # undo offset for simulation times
            sim_times = sim_times + given_times[0]

            # Calculate errors and record
            colidxs = []
            for t in given_times:
                for i,_t in enumerate(sim_times):
                    if t == _t:
                        colidxs.append(i)
                        break
            try:
                given_times_pred_traj = pred_traj[:,colidxs]
            except:
                colidxs[-1] -= -1
                given_times_pred_traj = pred_traj[:,colidxs]

            given_times_pred_traj_ = given_times_pred_traj
            given_times_pred_traj_[given_times_pred_traj_==0] = self.traj_fillvalue

            # If truth is supplied, then use that for the error metric
            if truth is not None:
                if error_metric.__name__ == 'spearmanr':
                    error_total = np.mean(error_metric(M_truth_, given_times_pred_traj_, axis=1)[0])
                else:
                    error_total = np.mean(error_metric(M_truth_, given_times_pred_traj_, axis=1))
                error_ASVs = []
                for oidx in range(len(subject.asvs)):
                    if error_metric.__name__ == 'spearmanr':
                        error_ASVs.append(error_metric(M_truth_[oidx,:], given_times_pred_traj_[oidx,:])[0])
                    else:
                        error_ASVs.append(error_metric(M_truth_[oidx,:], given_times_pred_traj_[oidx,:]))

            else:
                if error_metric.__name__ == 'spearmanr':
                    error_total = np.mean(error_metric(M_, given_times_pred_traj_,axis=1)[0])
                else:
                    error_total = np.mean(error_metric(M_, given_times_pred_traj_,axis=1))
                error_ASVs = []
                for oidx in range(len(subject.asvs)):
                    if error_metric.__name__ == 'spearmanr':
                        error_ASVs.append(error_metric(M_[oidx,:], given_times_pred_traj_[oidx,:])[0])
                    else:
                        error_ASVs.append(error_metric(M_[oidx,:], given_times_pred_traj_[oidx,:]))

        else:
            # Using a bayesian model, integrate over the posterior
            if self.mp is not None:
                pool = pl.multiprocessing.PersistentPool(ptype='dasw')
                for i in range(self.mp):
                    pool.add_worker(_ForwardSimWorker(asvs=subject.asvs,
                        initial_conditions=initial_conditions, start_day=_start,
                        sim_dt=self.simulation_dt, n_days=sim_times[-1]+self.simulation_dt, name=i,
                        log_integration=self.log_integration, 
                        perturbations_additive=self.perturbations_additive,
                        sim_times=sim_times))
                pool.staged_map_start(func='integrate')

            for i in range(growth.shape[0]):
                if self.mp is None:
                    logging.info('integrated {}/{} simulations'.format(i, growth.shape[0]))
                    pred_dyn = model_module.gLVDynamicsSingleClustering(asvs=subject.asvs, 
                        log_dynamics=self.log_integration, 
                        perturbations_additive=self.perturbations_additive, 
                        sim_max=self.sim_max, start_day=_start)
                    pred_dyn.growth = growth[i]
                    pred_dyn.self_interactions = self_interactions[i]
                    pred_dyn.interactions = interactions[i]
                    if perturbations is not None:
                        pred_dyn.perturbations = []
                        for start,end,pert_trace in perturbations:
                            pred_dyn.perturbations.append((start,end,pert_trace[i]))
                    
                    _d = pl.dynamics.integrate(dynamics=pred_dyn, 
                        initial_conditions=initial_conditions,
                        dt=self.simulation_dt, n_days=sim_times[-1]+self.output_dt, 
                        subsample=True, times=sim_times, log_every=None)
                    _d['X'][np.isnan(_d['X'])] = sys.float_info.max
                    X_pred[i] = _d['X']
                else:
                    kwargs = {
                        'i': i,
                        'growth': growth[i],
                        'self_interactions': self_interactions[i],
                        'interactions': interactions[i]}
                    kwargs['perturbations'] = None
                    if perturbations is not None:
                        pert = []
                        for start,end,pert_trace in perturbations:
                            pert.append((start, end, pert_trace[i]))
                        kwargs['perturbations'] = pert
                    pool.staged_map_put(kwargs)
            
            if self.mp is not None:
                ret = pool.staged_map_get()
                pool.kill()
                X_pred = np.asarray(ret, dtype=float)

            # print('\n\n\n\nX_pred shape')
            # print(X_pred.shape)
            # # sys.exit()

            # Calculate error and record
            # --------------------------
            # If it is an mcmc model, we predict the errors over the whole posterior
            pred_traj = np.nanpercentile(a=X_pred, q=50, axis=0)
            # print('pred_traj shape', pred_traj.shape)
            pred_traj_high = np.nanpercentile(a=X_pred, q=100-percentile, axis=0)
            # print('pred_traj_high shape', pred_traj_high.shape)
            pred_traj_low = np.nanpercentile(a=X_pred, q=percentile, axis=0)
            # print('pred_traj_low shape', pred_traj_low.shape)

            # undo offset for simulation times
            sim_times = sim_times + given_times[0]

            # print('sim times')
            # print(len(sim_times))
            # # print(sim_times)
            # print('given times')
            # print(len(given_times))
            # print(given_times)

            colidxs = []
            for t in given_times:
                found = False
                for i,_t in enumerate(sim_times):
                    if t == _t:
                        colidxs.append(i)
                        found = True
                        break
                if not found:
                    raise ValueError('`time {} not found in {}'.format(t, sim_times))
            # print('colidxs')
            # print(colidxs)
            try:
                given_times_pred_traj = X_pred[:,:,colidxs]
            except:
                colidxs[-1] -= -1
                given_times_pred_traj = X_pred[:,:,colidxs]

            given_times_pred_traj_ = given_times_pred_traj
            given_times_pred_traj_[given_times_pred_traj_==0] = self.traj_fillvalue

            error_total = np.zeros(X_pred.shape[0], dtype=float)
            error_ASVs = np.zeros(shape=(X_pred.shape[0],X_pred.shape[1]), dtype=float)

            # print('error_total', error_total.shape)
            # print('error_asvs', error_ASVs.shape)
            # print('truth.shape', M_truth_.shape)
            # print('given_times_pred_traj_', given_times_pred_traj_.shape)
            # print('\n', given_times)
            # print('\n', sim_times)


            for i in range(len(error_total)):
                if truth is not None:
                    if error_metric.__name__ == 'spearmanr':
                        error_total[i] = np.nanmean(error_metric(M_truth_, given_times_pred_traj_[i], axis=1)[0])
                    else:
                        error_total[i] = np.nanmean(error_metric(M_truth_, given_times_pred_traj_[i], axis=1))
                    for oidx in range(len(subject.asvs)):
                        if error_metric.__name__ == 'spearmanr':
                            error_ASVs[i,oidx] = error_metric(M_truth_[oidx,:], given_times_pred_traj_[i,oidx,:])[0]
                        else:
                            error_ASVs[i,oidx] = error_metric(M_truth_[oidx,:], given_times_pred_traj_[i,oidx,:])
                else:
                    if error_metric.__name__ == 'spearmanr':
                        error_total[i] = np.nanmean(error_metric(M_, given_times_pred_traj_[i], axis=1)[0])
                    else:
                        error_total[i] = np.nanmean(error_metric(M_, given_times_pred_traj_[i], axis=1))
                    for oidx in range(len(subject.asvs)):
                        if error_metric.__name__ == 'spearmanr':
                            error_ASVs[i,oidx] = error_metric(M_[oidx,:], given_times_pred_traj_[i,oidx,:])[0]
                            print('\t{} error_ASVs[{},{}]'.format(oidx,i,oidx), error_ASVs[i,oidx])
                            if np.isnan(error_ASVs[i,oidx]):
                                print('NAN')
                                print(M_[oidx,:])
                                print(given_times_pred_traj_[i,oidx,:])
                        else:
                            error_ASVs[i,oidx] = error_metric(M_[oidx,:], given_times_pred_traj_[i,oidx,:])

        ret = {}
        if simtype == 'sim-days':
            a = (simtype, START_START, n_days)
        else:
            a = (simtype,)

        ret['subject'] = subject
        ret['dtype'] = dtype
        ret['subj-traj'] = M
        ret['subj-times'] = _end_subj_times
        ret['initial-conditions'] = initial_conditions
        ret['pred-traj'] = pred_traj
        ret['pred-times'] = sim_times
        ret['error-total'] = error_total
        ret['error-asvs'] = error_ASVs
        ret['error-metric'] = error_metric

        if pl.isMCMC(self.model):
            ret['pred-high'] = pred_traj_high
            ret['pred-low'] = pred_traj_low
        if truth is not None:
            ret['truth-traj'] = M_truth
            ret['truth-times'] = _end_truth_times

        # for k,v in ret.items():
        #     print('{}:{}'.format(k,v))

        if save_at_finish:
            if subject.name not in self.results:
                self.results[subject.name] = {}
            if a in self.results[subject.name]:
                logging.info('Simulation type {} already in results, overwritting'.format(a))
            self.results[subject.name][a] = ret
        else:
            return ret
            
    def plot(self, basepath, title_formatter=None, yscale='log', legend=True):
        '''Plot the trajectories that we forward simulated individually. Iterates through the
        `results` dictionary. If there are recorded truth then we plot those as well.

        Title formatter
        ---------------
        Use these formats to substitute with each ASV title.
        Options:
            Simulation options:
                '%(error)s'   : Error
                '%(subjname)s': Subject Name
            ASV options:
                '%(name)s'    : Name of the ASV (pylab.base.ASV.name)
                '%(id)s'      : ID of the ASV (pylab.base.ASV.id)
                '%(index)s'   : The order that this appears in the ASVSet
                '%(genus)s'   : `'genus'` taxonomic classification of the ASV
                '%(family)s'  : `'family'` taxonomic classification of the ASV
                '%(class)s'   : `'class'` taxonomic classification of the ASV
                '%(order)s'   : `'order'` taxonomic classification of the ASV
                '%(phylum)s'  : `'phylum'` taxonomic classification of the ASV
                '%(kingdom)s' : `'kingdom'` taxonomic classification of the ASV
        Example:
                asv.name  = 'ASV_3'
                asv.idx   = 1
                asv.genus = 'Parabacteroides'
                RMSE(asv) = 1.5179e+09
                title_formatter = '%(index)s: (%(name)s): %(genus)s\nRMSE: %(error)s CFUs/g'
                output:
                    '1: (ASV_3) Parabacteroides\n RMSE: 1.5179e+09'

        Parameters
        ----------
        basepath : str
            Folder where you want to save the plots
        title_formatter : str, optional
            Format for the title of the plots. If nothing is provided then only use the 
            asv name.
        yscale : str
            See `matplotlib.pyplot.Axes.set_yscale` for options
        legend : bool
            If True add a legend. If False do not add a legend
        '''
        if not pl.isstr(basepath):
            raise TypeError('`basepath` ({}) must be a str'.format(type(basepath)))
        if basepath[-1] not in ['/']:
            basepath += '/'

        BLUE = sns.color_palette('deep')[0]

        for subjname in self.time_lookaheads:
            fig = plt.figure()
            ax = fig.add_subplot(111)

            n_colors = len(self.time_lookaheads[subjname]['lookaheads'])
            colors = sns.color_palette(n_colors=n_colors)

            subject = self.time_lookaheads[subjname]['subject']

            for tla_idx, tla in enumerate(self.time_lookaheads[subjname]['lookaheads']):

                xs = self.time_lookaheads[subjname]['lookaheads'][tla]['times']
                ys = self.time_lookaheads[subjname]['lookaheads'][tla]['pred']
                error_metric = self.time_lookaheads[subjname]['lookaheads'][tla]['metric']

                idxs = ~np.isnan(ys)
                xs = xs[idxs]
                ys = ys[idxs]
                if pl.isMCMC(self.model):
                    high_ys = self.time_lookaheads[subjname]['lookaheads'][tla]['high']
                    low_ys = self.time_lookaheads[subjname]['lookaheads'][tla]['low']

                    high_ys = high_ys[idxs]
                    low_ys = low_ys[idxs]

                    ax.fill_between(xs, y1=high_ys, y2=low_ys,
                            alpha=0.25, color=colors[tla_idx])
                ax.plot(xs, ys, color=colors[tla_idx], label='{} Days'.format(tla))

                print('printing ')
                print(xs)

            ax.legend()
            ax.set_xlabel('Days')

            pl.visualization.shade_in_perturbations(ax=ax, 
                perturbations=subject.perturbations, textcolor='grey',
                textsize=9)

            if error_metric == pl.metrics.PE:
                ylabel = 'Percent Error'
                ysc = 'linear'
            elif error_metric == pl.metrics.RMSE:
                ylabel = 'Root Mean Squared Error'
                ysc = 'log'
            else:
                ylabel = error_metric.__name__
                ysc = 'log'
            ax.set_ylabel(ylabel)
            ax.set_title('Time lookahead Errors')
            ax.set_yscale(ysc)

            plt.savefig(basepath + 'lookahead.pdf')
            plt.close()

        # Iterate through the simulation types and subjects
        for subject_name in self.results:
            for simtype in self.results[subject_name]:
                if len(simtype) == 1:
                    sim_name = simtype[0]
                else:
                    sim_name = '-'.join([str(s) for s in simtype])
                
                simpath = basepath + '{}-'.format(subject_name) + sim_name + '/'
                os.makedirs(simpath, exist_ok=True)

                # timelookaheadpath = simpath + 'lookahead/'
                # os.makedirs(timelookaheadpath, exist_ok=True)

                # Get data
                M = self.results[subject_name][simtype]['subj-traj']
                subj_times = self.results[subject_name][simtype]['subj-times']
                sim_traj = self.results[subject_name][simtype]['pred-traj']
                sim_times = self.results[subject_name][simtype]['pred-times']
                error_asvs = self.results[subject_name][simtype]['error-asvs']
                subject = self.results[subject_name][simtype]['subject']
                dtype = self.results[subject_name][simtype]['dtype']
                error_metric = self.results[subject_name][simtype]['error-metric']
                if pl.isMCMC(self.model):
                    pred_traj_high = self.results[subject_name][simtype]['pred-high']
                    pred_traj_low = self.results[subject_name][simtype]['pred-low']
                M_truth = None
                if 'truth-traj' in self.results[subject_name][simtype]:
                    M_truth = self.results[subject_name][simtype]['truth-traj']
                    truth_times = self.results[subject_name][simtype]['truth-times']

                # if pl.isMCMC(self.model):
                #     errorpath = simpath + 'errors/'
                #     os.makedirs(errorpath, exist_ok=True)

                # For each ASV, plot
                for oidx in range(len(subject.asvs)):

                    # Plot the trajectory
                    fig = plt.figure()
                    ax = fig.add_subplot(111)

                    if len(simtype) == 3: 
                        ax.set_xlim(left= (simtype[1]-0.05*simtype[2]), right=(simtype[1] + 1.05*simtype[2]))
                    
                    if subject.perturbations is not None:
                        pl.visualization.shade_in_perturbations(ax=ax, 
                            perturbations=subject.perturbations, textcolor='grey',
                            textsize=9)
                    ax.plot(subj_times, M[oidx,:], label='Data', 
                        color='black', marker='x', linestyle=':')
                    if M_truth is not None:
                        ax.plot(truth_times, M_truth[oidx, :], label='Truth', color='darkviolet', 
                            linestyle=':')
                    if pl.isMCMC(self.model):
                        ax.fill_between(sim_times, y1=pred_traj_low[oidx,:], y2=pred_traj_high[oidx,:],
                            alpha=0.25, color=BLUE)
                    ax.plot(sim_times, sim_traj[oidx, :], color=BLUE, label='Predicted')

                    ax.set_yscale(yscale)
                    if dtype == 'raw':
                        ax.set_ylabel('Counts')
                    elif dtype == 'rel':
                        ax.set_ylabel('Relative abundance')
                    else:
                        ax.set_ylabel('CFUs/g')
                    ax.set_xlabel('Days')

                    if pl.isMCMC(self.model):
                        err = np.nanmean(error_asvs[:, oidx])
                    else:
                        err = error_asvs[oidx]

                    title = title_formatter.replace('%(error)s', '{:.6E}'.format(err))
                    title = title.replace('%(subjname)s', subject.name)
                    title = pl.asvname_formatter(format=title, asv=oidx, asvs=subject.asvs)
                    ax.set_title(title)

                    # Set y limits. Set x limits if we are doing 'sim-days'
                    new_low = None
                    new_high = None
                    low,high = ax.get_ylim()
                    if low < 1e4:
                        new_low = 1e4
                    if high > 1e13:
                        new_high = 1e13
                    ax.set_ylim(bottom=new_low, top=new_high)

                    if legend:
                        ax.legend()
                    
                    fig.tight_layout()
                    plt.savefig(simpath + '{}.pdf'.format(subject.asvs[oidx].name))
                    plt.close()

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
