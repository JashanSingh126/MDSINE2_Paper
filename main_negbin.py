'''Learn the Negative Binomial Dispersion parameters offline of the main inference.
'''

import numpy as np
import logging
import sys
import time
import pandas as pd
import os
import os.path
import argparse

import numpy.random as npr
import random
import math
import numba

import matplotlib.pyplot as plt
import seaborn as sns

import pylab as pl
import config
from names import REPRNAMES, STRNAMES
import util_negbin as util

# import dirichlet # THIS IS FOR LEARNING THE DIRICHLET CONCENTRATION (dirichlet.mle(Data))

@numba.jit(nopython=True, fastmath=True, cache=True)
def negbin_loglikelihood(k,m,dispersion):
    '''Loglikelihood - with parameterization in [1]
    
    Parameters
    ----------
    k : numeric
        Observed counts
    m : numeric
        Mean
    phi : float
        Dispersion
    
    Returns
    -------
    float
        Negative Binomial Log Likelihood
    
    References
    ----------
    [1] TE Gibson, GK Gerber. Robust and Scalable Models of Microbiome Dynamics. ICML (2018)
    '''
    r = 1/dispersion
    return math.lgamma(k+r) - math.lgamma(k+1) - math.lgamma(r) \
            + r * (math.log(r) - math.log(r+m)) + k * (math.log(m) - math.log(r+m))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--synth', '-s', type=int,
        help='1 if run synthetic, 0 if run real data',
        dest='synthetic', default=0)
    parser.add_argument('--data-seed', '-d', type=int,
        help='Seed to initialize the data', default=0,
        dest='data_seed')
    parser.add_argument('--basepath', '-b', type=str,
        help='Folder to save the output', default='output_negbin/',
        dest='basepath')
    parser.add_argument('--n-samples', '-ns', type=int,
        help='Total number of Gibbs steps to do',
        dest='n_samples', default=1000)
    parser.add_argument('--burnin', '-nb', type=int,
        help='Total number of burnin steps',
        dest='burnin', default=500)

    args = parser.parse_args()
    return args


class Data(pl.graph.DataNode):
    '''This is the raw data that we are regressing over

    Parameters
    ----------
    asvs : pl.base.ASVSet
        ASVSet that we use to index the data
    subjects : pl.base.SubjectSet
            - These are a list of the subjects that we are going to get
              data from
    '''

    def __init__(self, subjects, **kwargs):
        kwargs['name'] = 'Data'
        pl.graph.DataNode.__init__(self, **kwargs)
        if not pl.issubjectset(subjects):
            raise ValueError('`subjects` ({}) must be a pylab SubjectSet'.format(
                type(subjects)))
        
        self.asvs = subjects.asvs
        self.subjects = subjects
        self.n_asvs = len(self.asvs)

        self.data = []
        self.read_depths = []
        self.qpcr = []
        for subject in self.subjects:
            d = subject.matrix()['raw']
            self.data.append(d)
            self.read_depths.append(np.sum(d, axis=0))
            self.qpcr.append(subject.qpcr[0])

        self.n_replicates = len(self.data)

    def __len__(self):
        return self.n_replicates


class NegBinDispersionParam(pl.variables.Uniform):
    '''These are for learning the a0 and a1 parameters - updated with 
    Metropolis-Hastings

    We assume these are uniform and have a uniform prior [1]

    Proposal distribution is a truncated normal distribution with truncation
    set to the same high and lows as the prior.

    References
    ----------
    [1] Bucci, Vanni, et al. "MDSINE: Microbial Dynamical Systems INference 
        Engine for microbiome time-series analyses." Genome biology 17.1 (2016): 121.
    '''

    def __init__(self, name, **kwargs):
        pl.variables.Uniform.__init__(
            self, dtype=float, name=name, **kwargs)
        self.proposal = pl.variables.TruncatedNormal(mean=None, var=None, value=None)

    def __str__(self):
        try:
            s = 'Value: {}, Acceptance rate: {}'.format(
                self.value, np.mean(self.acceptances[
                    np.max([self.sample_iter-50, 0]):self.sample_iter]))
        except:
            s = str(self.value)
        return s

    def initialize(self, value, truncation_settings, proposal_option,
        target_acceptance_rate, tune, end_tune, proposal_var=None, delay=0):
        '''Initialize the negative binomial dispersion parameter

        Parameters
        ----------
        value : numeric
            This is the initial value
        truncation_settings: str, tuple
            How to set the truncation parameters. The proposal trucation will
            be set the same way.
                tuple - (low,high)
                    These are the truncation parameters
                'auto'
                    (0, 1e5)
        proposal_option : str
            How to initialize the proposal variance:
                'auto'
                    initial_value**2 / 100
                'manual'
                    `proposal_var` must also be supplied
        target_acceptance_rate : str, float
            If float, this is the target acceptance rate
            If str: 
                'optimal', 'auto': 0.44
        tune : str, int
            How often to tune the proposal. If str:
                'auto': 50
        end_tune : str, int
            When to stop tuning the proposal. If str:
                'auto', 'half-burnin': Half of burnin
        '''
        if not pl.isint(delay):
            raise TypeError('`delay` ({}) must be an int'.format(type(delay)))
        if delay < 0:
            raise ValueError('`delay` ({}) must be >= 0'.format(delay))
        self.delay = delay

        # Set the propsal parameters
        if pl.isstr(target_acceptance_rate):
            if target_acceptance_rate in ['optimal', 'auto']:
                target_acceptance_rate = 0.44
            else:
                raise ValueError('`target_acceptance_rate` ({}) not recognized'.format(
                    target_acceptance_rate))
        elif pl.isfloat(target_acceptance_rate):
            if target_acceptance_rate < 0 or target_acceptance_rate > 1:
                raise ValueError('`target_acceptance_rate` ({}) out of range'.format(
                    target_acceptance_rate))
        else:
            raise TypeError('`target_acceptance_rate` ({}) type not recognized'.format(
                type(target_acceptance_rate)))
        self.target_acceptance_rate = target_acceptance_rate

        if pl.isstr(tune):
            if tune in ['auto']:
                tune = 50
            else:
                raise ValueError('`tune` ({}) not recognized'.format(tune))
        elif pl.isint(tune):
            if tune < 0:
                raise ValueError('`tune` ({}) must be > 0'.format(
                    tune))
        else:
            raise TypeError('`tune` ({}) type not recognized'.format(type(tune)))
        self.tune = tune

        if pl.isstr(end_tune):
            if end_tune in ['auto', 'half-burnin']:
                end_tune = int(self.G.inference.burnin/2)
            else:
                raise ValueError('`tune` ({}) not recognized'.format(end_tune))
        elif pl.isint(end_tune):
            if end_tune < 0 or end_tune > self.G.inference.burnin:
                raise ValueError('`end_tune` ({}) out of range (0, {})'.format(
                    end_tune, self.G.inference.burnin))
        else:
            raise TypeError('`end_tune` ({}) type not recognized'.format(type(end_tune)))
        self.end_tune = end_tune

        # Set the truncation settings
        if pl.isstr(truncation_settings):
            if truncation_settings == 'auto':
                self.low = 0.
                self.high = 1e5
            else:
                raise ValueError('`truncation_settings` ({}) not recognized'.format(
                    truncation_settings))
        elif pl.istuple(truncation_settings):
            if len(truncation_settings) != 2:
                raise ValueError('If `truncation_settings` is a tuple, it must have a ' \
                    'length of 2 ({})'.format(len(truncation_settings)))
            l,h = truncation_settings

            if (not pl.isnumeric(l)) or (not pl.isnumeric(h)):
                raise TypeError('`low` ({}) and `high` ({}) must be numerics'.format(
                    type(l), type(h)))
            if l < 0 or h < 0:
                raise ValueError('`low` ({}) and `high` ({}) must be >= 0'.format(l,h))
            if h <= l:
                raise ValueError('`low` ({}) must be strictly less than high ({})'.format(l,h))
            self.high.value = h
            self.low.value = l
        else:
            raise TypeError('`truncation_settings` ({}) type not recognized')
        self.proposal.high = self.high
        self.proposal.low = self.low

        # Set the value
        if not pl.isnumeric(value):
            raise TypeError('`value` ({}) must be a numeric'.format(type(value)))
        if value <= self.low or value >= self.high:
            raise ValueError('`value` ({}) out of range ({})'.format(
                value, (self.low, self.high)))
        self.value = value

        # Set the proposal variance
        if not pl.isstr(proposal_option):
            raise TypeError('`proposal_option` ({}) must be a str'.format(
                type(proposal_option)))
        elif proposal_option == 'manual':
            if not pl.isnumeric(proposal_var):
                raise TypeError('`proposal_var` ({}) must be a numeric'.format(
                    type(proposal_var)))
            if proposal_var <= 0:
                raise ValueError('`proposal_var` ({}) not proper'.format(proposal_var))
        elif proposal_option in ['auto']:
            proposal_var = (self.value ** 2)/10
        else:
            raise ValueError('`proposal_option` ({}) not recognized'.format(
                proposal_option))
        self.proposal.var.value = proposal_var

    def _update_proposal_variance(self):
        '''Update the proposal variance
        '''
        if self.sample_iter == 0:
            self.temp_acceptances = 0
            self.acceptances = np.zeros(self.G.inference.n_samples, dtype=bool)
        
        elif self.sample_iter > self.end_tune:
            # Don't do any more updates
            return
        
        elif self.sample_iter % self.tune == 0:
            # Update var
            acceptance_rate = self.temp_acceptances / self.tune
            if acceptance_rate > self.target_acceptance_rate:
                self.proposal.var.value *= 1.5
            else:
                self.proposal.var.value /= 1.5
            self.temp_acceptances = 0

    def update(self):
        '''Do a metropolis update
        '''
        # Update proposal variance if necessary
        if self.sample_iter < self.delay:
            return
        self._update_proposal_variance()
        proposal_std = np.sqrt(self.proposal.var.value)

        # Get the current likelihood
        old_loglik = self.data_likelihood()
        prev_value = self.value

        # Propose a new value and get the likelihood
        self.value = pl.random.truncnormal.sample(
            mean=self.value, std=proposal_std,
            low=self.proposal.low, high=self.proposal.high)
        new_loglik = self.data_likelihood()

        # reverse jump probabilities
        jump_to_new = pl.random.truncnormal.logpdf(value=self.value, 
            mean=prev_value, std=proposal_std, 
            low=self.proposal.low, high=self.proposal.high)
        jump_to_old = pl.random.truncnormal.logpdf(value=prev_value, 
            mean=self.value, std=proposal_std, 
            low=self.proposal.low, high=self.proposal.high)
        

        r = (new_loglik + jump_to_old) - (old_loglik + jump_to_new)
        u = np.log(pl.random.misc.fast_sample_standard_uniform())
        if r > u:
            self.acceptances[self.sample_iter] = True
            self.temp_acceptances += 1
        else:
            self.value = prev_value

    def data_likelihood(self):
        '''Calculate the current log likelihood
        '''
        a0 = self.G[REPRNAMES.NEGBIN_A0].value
        a1 = self.G[REPRNAMES.NEGBIN_A1].value
        latents = [v.value for v in self.G[REPRNAMES.FILTERING].value]
        datas = [v.data for v in self.G[REPRNAMES.FILTERING].value]
        read_depths = [v.read_depths for v in self.G[REPRNAMES.FILTERING].value]
        
        cumm = 0
        for ridx in range(len(latents)):
            data=datas[ridx]
            latent = latents[ridx]
            read_depth = read_depths[ridx]
            total_abund = np.sum(latent)
            rel_abund = latent / total_abund

            cumm += NegBinDispersionParam._data_likelihood(a0=a0, a1=a1, latent=latent, 
                data=data, read_depth=read_depth, rel_abund=rel_abund)
        return cumm
    
    @staticmethod
    @numba.jit(nopython=True)
    def _data_likelihood(a0, a1, latent, data, read_depth, rel_abund):
        cumm = 0

        # For each asv
        for oidx in range(data.shape[0]):
            # For each replicate
            for k in range(data.shape[1]):
                y = data[oidx, k]
                mean = read_depth[k] * rel_abund[oidx]
                dispersion = a0/rel_abund[k] + a1

                # This is the negative binomial loglikelihood
                r = 1/dispersion
                # try:
                cumm += math.lgamma(y+r) - math.lgamma(y+1) - math.lgamma(r) \
                    + r * (math.log(r) - math.log(r+mean)) + y * (math.log(mean) - math.log(r+mean))

                #     raise
        return cumm


class TrajectorySet(pl.variables.Variable):
    '''This aggregates a set of trajectories from each set
    '''
    def __init__(self, ridx, **kwargs):
        kwargs['name'] = STRNAMES.LATENT_TRAJECTORY + '_{}'.format(ridx)
        pl.variables.Variable.__init__(self, **kwargs)
        n_asvs = len(self.G.data.asvs)
        self.set_value_shape(shape=(n_asvs,))
        self.ridx = ridx
        self.value = np.zeros(n_asvs, dtype=float)
        self.data = self.G.data.data[self.ridx]
        self.read_depths = self.G.data.read_depths[self.ridx]
        self.qpcr_measurement = self.G.data.qpcr[self.ridx]
    
        prior = pl.variables.Normal(
            mean=pl.variables.Constant(name=self.name+'_prior_mean', value=None, G=self.G),
            var=pl.variables.Constant(name=self.name+'_prior_var', value=None, G=self.G),
            name=self.name+'_prior', G=self.G)
        self.add_prior(prior)

    def __getitem__(self, ridx):
        return self.value[ridx]

    def initialize(self):
        '''Initialize the value
        '''
        # Get the mean relative abundance
        rel = np.sum(self.data, axis=1)
        rel = rel / np.sum(rel)
        value = rel * self.qpcr_measurement.mean()

        self.value = np.zeros(len(value))
        for i in range(len(value)):
            self.value[i] = pl.random.truncnormal.sample(mean=value[i], std=1e-2, 
                low=0, high=float('inf'))

        self.prior.mean.override_value(self.value)
        self.prior.var.override_value(100 * np.var(self.value))


class FilteringMP(pl.graph.Node):
    '''This handles multiprocessing of the latent state

    Parallelization Modes
    ---------------------
    'debug'
        If this is selected, then we dont actually parallelize, but we go in
        order of the objects in sequential order. We would do this if we want
        to benchmark within each processor or do easier print statements
    'full'
        This is where each subject gets their own process

    This assumes that we are using the log model for the dynamics
    '''
    def __init__(self, mp, **kwargs):
        kwargs['name'] = STRNAMES.FILTERING
        pl.graph.Node.__init__(self, **kwargs)
        self.value = []
        for ridx in range(len(self.G.data.data)):
            self.value.append(TrajectorySet(G=self.G, ridx=ridx))
        
        self.print_vals = False
        self._strr = 'parallel'
        self.mp = mp

    def __str__(self):
        return self._strr

    @property
    def sample_iter(self):
        # It doesnt matter if we chose q or x because they are both the same
        return self.value[0].sample_iter

    def initialize(self, tune, end_tune, target_acceptance_rate, 
        qpcr_variance_inflation, delay=0):
        '''Initialize the latent state

        Parameters
        ----------
        value_option : str
            'tight-coupling'
                Set the value to the empirical mean of the trajectory with a small variance
            'small-bias'
                Add 1e-10 to all of the latent states
        target_acceptance_rate : str, float
            If float, this is the target acceptance rate
            If str: 
                'optimal', 'auto': 0.44
        tune : str, int
            How often to tune the proposal. If str:
                'auto': 50
        end_tune : str, int
            When to stop tuning the proposal. If str:
                'auto', 'half-burnin': Half of burnin
        proposal_option : str
            How to initialize the proposal variance:
                'auto'
                    initial_value**2 / 100
                'manual'
                    `proposal_var` must also be supplied
        qpcr_variance_inflation : float
            This is the factor to inflate the qPCR variance
        delay : int
            How many Gibb stepps to delay
        '''
        if not pl.isint(delay):
            raise TypeError('`delay` ({}) must be an int'.format(type(delay)))
        if delay < 0:
            raise ValueError('`delay` ({}) must be >= 0'.format(delay))
        self.delay = delay

        # Set the propsal parameters
        if pl.isstr(target_acceptance_rate):
            if target_acceptance_rate in ['optimal', 'auto']:
                target_acceptance_rate = 0.44
            else:
                raise ValueError('`target_acceptance_rate` ({}) not recognized'.format(
                    target_acceptance_rate))
        elif pl.isfloat(target_acceptance_rate):
            if target_acceptance_rate < 0 or target_acceptance_rate > 1:
                raise ValueError('`target_acceptance_rate` ({}) out of range'.format(
                    target_acceptance_rate))
        else:
            raise TypeError('`target_acceptance_rate` ({}) type not recognized'.format(
                type(target_acceptance_rate)))
        self.target_acceptance_rate = target_acceptance_rate

        if pl.isstr(tune):
            if tune in ['auto']:
                tune = 50
            else:
                raise ValueError('`tune` ({}) not recognized'.format(tune))
        elif pl.isint(tune):
            if tune < 0:
                raise ValueError('`tune` ({}) must be > 0'.format(
                    tune))
        else:
            raise TypeError('`tune` ({}) type not recognized'.format(type(tune)))
        self.tune = tune

        if pl.isstr(end_tune):
            if end_tune in ['auto', 'half-burnin']:
                end_tune = int(self.G.inference.burnin/2)
            else:
                raise ValueError('`tune` ({}) not recognized'.format(end_tune))
        elif pl.isint(end_tune):
            if end_tune < 0 or end_tune > self.G.inference.burnin:
                raise ValueError('`end_tune` ({}) out of range (0, {})'.format(
                    end_tune, self.G.inference.burnin))
        else:
            raise TypeError('`end_tune` ({}) type not recognized'.format(type(end_tune)))
        self.end_tune = end_tune

        # Initialize the trajectory sets
        self.value = []
        for ridx in range(self.G.data.n_replicates):
            self.value.append(TrajectorySet(ridx=ridx, G=self.G))
            self.value[ridx].initialize()

        if self.mp == 'full':
            self.pool = pl.multiprocessing.PersistentPool(G=self.G, ptype='sadw')
        elif self.mp == 'debug':
            self.pool = []
        else:
            raise ValueError('Filtering mutliprocessing argument ({}) not recognized'.format(
                self.mp))

        for ridx in range(len(self.value)):
            worker = _LatentWorker()
            worker.initialize(
                reads=self.value[ridx].data,
                qpcr_loc=self.value[ridx].qpcr_measurement.loc,
                qpcr_scale=np.sqrt(qpcr_variance_inflation) *self.value[ridx].qpcr_measurement.scale,
                proposal_std=np.log(1.5),
                prior_loc=self.value[ridx].prior.mean.value,
                prior_scale=np.sqrt(self.value[ridx].prior.var.value),
                tune=tune, end_tune=end_tune,
                target_acceptance_rate=target_acceptance_rate,
                value=self.value[ridx].value,
                delay=delay,
                ridx=ridx)
            if self.mp == 'full':
                self.pool.add_worker(worker)
            else:
                self.pool.append(worker)

        self.total_n_datapoints = len(self.G.data.asvs) * len(self.G.data)

    def update(self):
        start_time = time.time()
        a0 = self.G[REPRNAMES.NEGBIN_A0].value
        a1 = self.G[REPRNAMES.NEGBIN_A1].value

        kwargs={'a0': a0, 'a1':a1}
        str_acc = [None]*self.G.data.n_replicates
        if self.mp == 'debug':
            for ridx in range(len(self.value)):
                _, x, acc_rate = self.pool[ridx].update(**kwargs)
                self.value[ridx].value = x
                str_acc[ridx] = '{:.3f}'.format(acc_rate)
        else:
            ret = self.pool.map(func='update', args=kwargs)
            for ridx, x, acc_rate in ret:
                self.value[ridx].value = x
                str_acc[ridx] = '{:.3f}'.format(acc_rate)

        t = time.time() - start_time
        try:
            self._strr = 'Time: {:.4f}, Acc: {}, data/sec: {:.2f}'.format(t,
                str(str_acc).replace("'",''), self.total_n_datapoints/t)
        except:
            self._strr = 'NA'

    def add_trace(self):
        for x in self.value:
            x.add_trace()

    def set_trace(self, *args, **kwargs):
        for x in self.value:
            x.set_trace(*args, **kwargs)

    def add_init_value(self):
        for x in self.value:
            x.add_init_value()
    
    def kill(self):
        if pl.ispersistentpool(self.pool):
            self.pool.kill()


class _LatentWorker(pl.multiprocessing.PersistentWorker):
    '''Worker class for multiprocessing. Everything is in log scale
    '''
    def __init__(self):
        return

    def initialize(self, reads, qpcr_loc, qpcr_scale, prior_loc, prior_scale,
        proposal_std, tune, end_tune, target_acceptance_rate, value, delay,
        ridx):
        '''Initialize the values

        reads : np.ndarray((n_asvs x n_reps))
        qpcr_mean : float
        qpcr_std : float
        prior_mean : float
        prior_std : float
        proposal_std : float
        tune : int
        end_tune : int
        target_acceptance_rate :float
        value : np.ndarray((n_asvs,))
        ridx : int
        '''
        self.reads = reads
        self.read_depths = np.sum(self.reads, axis=0)
        self.qpcr_loc = qpcr_loc
        self.qpcr_scale = qpcr_scale
        self.prior_loc = prior_loc
        self.prior_scale = prior_scale
        self.proposal_std = proposal_std
        self.tune = tune
        self.end_tune = end_tune
        self.target_acceptance_rate = target_acceptance_rate
        self.value = value
        self.ridx = ridx

        self.sumq = np.sum(self.value)
        self.log_sumq = np.log(self.sumq)

        self.sample_iter = 0
        self.acceptances = 0
        self.total_acceptances = 0

    def update_proposal_std(self):
        if self.sample_iter > self.end_tune:
            return
        if self.sample_iter == 0:
            return
        if self.sample_iter % self.tune == 0:
            # Adjust
            acc_rate = self.acceptances/self.n_props_total
            if acc_rate < 0.1:
                logging.debug('Very low acceptance rate, scaling down past covariance')
                self.proposal_std *= 0.01
            elif acc_rate < self.target_acceptance_rate:
                self.proposal_std /= np.sqrt(1.5)
            else:
                self.proposal_std *= np.sqrt(1.5)
            
            self.acceptances = 0
            self.n_props_local = 0

    def update(self, a0, a1):
        '''Update the latent state with the updated negative binomial
        dispersion parameters
        '''
        self.a0 = a0
        self.a1 = a1

        self.update_proposal_std()
        self.n_props_local = 0
        self.n_props_total = 0
        self.n_accepted_iter = 0

        oidxs = npr.permutation(self.reads.shape[0])
        for oidx in oidxs:
            self.update_single(oidx=oidx)

        return self.ridx, self.value, self.n_accepted_iter/len(self.value)

    def update_single(self, oidx):
        old_log_value = np.log(self.value[oidx])
        old_value = self.value[oidx]
        self.oidx = oidx
        self.curr_log_val = old_log_value

        aaa = self.prior_ll()
        bbb = self.qpcr_ll()
        ccc = self.negbin_ll()

        old_ll = aaa + bbb + ccc

        # propose new value
        log_new = pl.random.normal.sample(mean=old_log_value, 
            std=self.proposal_std)
        self.value[oidx] = np.exp(log_new)

        self.sumq = self.sumq - old_value + self.value[oidx]
        self.log_sumq = np.log(self.sumq)

        aaa = self.prior_ll()
        bbb = self.qpcr_ll()
        ccc = self.negbin_ll()

        new_ll = aaa + bbb + ccc

        r_accept = new_ll - old_ll
        r = pl.random.misc.fast_sample_standard_uniform()
        if math.log(r) > r_accept:
            # Reject
            self.sumq = self.sumq + old_value - self.value[oidx]
            self.log_sumq = np.log(self.sumq)
            self.value[self.oidx] = old_value
        else:
            self.acceptances += 1
            self.total_acceptances += 1
            self.n_accepted_iter += 1

        self.n_props_local += 1
        self.n_props_total += 1

    def prior_ll(self):
        return pl.random.normal.logpdf(value=self.curr_log_val, 
            mean=self.prior_loc[self.oidx], std=self.prior_scale)

    def qpcr_ll(self):
        return pl.random.normal.logpdf(value=self.log_sumq, 
            mean=self.qpcr_loc, std=self.qpcr_scale)

    def negbin_ll(self):
        cumm = 0
        rel = self.value[self.oidx]/self.sumq
        for k in range(self.reads.shape[1]):
            cumm += negbin_loglikelihood(
                k=self.reads[self.oidx, k],
                m=self.read_depths[k] * rel,
                dispersion=self.a0/rel + self.a1)
        return cumm


def run(params, graph_name, subjset_filename, graph_filename, tracer_filename, 
    hdf5_filename, mcmc_filename):
    '''This is the method used to run the model with the data located at
    `subjset_filename` with the parameters `params`.

    Parameters
    ----------
    params : config.NegBinConfig
        This class specifies all of the parameters of the model
    graph_name : str
        Name of the graph
    subjset_filename : str
        Location of the `pylab.base.SubjectSet` object containing the data
    graph_filename : str
        Location to save the graph
    tracer_filename : str
        Loction to store the Tracer object
    hdf5_filename : str
        Location to store the HDF5 file object
    mcmc_filename : str
        Location to store the MCMC chain object
    
    Returns
    -------
    pl.inference.BaseMCMC
        Inference chain
    '''

    GRAPH = pl.graph.Graph(name=graph_name, seed=params.SEED)
    GRAPH.as_default()
    GRAPH.set_save_location(graph_filename)

    # Make the asvset and subjectset
    subjset = pl.SubjectSet.load(subjset_filename)
    d = Data(subjects=subjset, G=GRAPH)

    # Make variables
    x = FilteringMP(mp=params.MP_FILTERING, G=GRAPH, name=STRNAMES.FILTERING)
    a0 = NegBinDispersionParam(name=STRNAMES.NEGBIN_A0, G=GRAPH, low=0, high=1e5)
    a1 = NegBinDispersionParam(name=STRNAMES.NEGBIN_A1, G=GRAPH, low=0, high=1e5)

    # Make mcmc object
    mcmc = pl.inference.BaseMCMC(burnin=params.BURNIN, n_samples=params.N_SAMPLES, 
        graph=GRAPH)
    
    # Set the inference order
    inference_order = []
    for name in params.INFERENCE_ORDER:
        if params.LEARN[name]:
            inference_order.append(name)
    mcmc.set_inference_order(inference_order)
    mcmc.set_save_location(mcmc_filename)

    REPRNAMES.set(G=GRAPH)

    for name in params.INITIALIZATION_ORDER:
        try:
            GRAPH[name].initialize(**params.INITIALIZATION_KWARGS[name])
        except:
            logging.critical('Failed in {}'.format(name))
            raise

    # Setup tracer - dont need to checkpoint
    mcmc.set_tracer(filename=hdf5_filename, ckpt=params.CKPT)
    mcmc.tracer.set_save_location(tracer_filename)

    # Run
    mcmc = mcmc.run(log_every=1)
    x.kill()

    return mcmc

@numba.jit(nopython=True, fastmath=True, cache=True)
def _single_calc_mean_var(means, variances, a0, a1, rels, read_depths):
    i = 0
    for col in range(rels.shape[1]):
        for oidx in range(rels.shape[0]):
            mean = rels[oidx, col] * read_depths[col]
            disp = a0 / mean + a1
            variances[i] = mean + disp * (mean**2)
            means[i] = mean

            i += 1
    return means, variances
   
def visualize_posterior(chain, basepath, true_subjset=None, true_a0=None, true_a1=None,
    asv_fmt='%(name)s: %(genus)s %(species)s'):
    '''Visualize:
        - latent abundances
        - a0 and a1 parameters
        - a0 and a1 posterior with the data

    Parameters
    ----------
    chain : pl.inference.BaseMCMC, str
        MCMC chain that we need to plot. If it is a str then it is the save location
        of the chain.
    basepath : str
        This is the base folder to plot in
    true_subjset : pylab.base.SubjectSet, str
        If provided, these are the true values for the latent state
    true_a0 : numeric
        If provided, this is the true value for a0
    true_a1 : numeric
        If provided, this is the true value for a1
    '''
    if pl.isstr(chain):
        chain = pl.inference.BaseMCMC.load(chain)
    if not pl.isMCMC(chain):
        raise TypeError('`chain` ({}) must be a chain'.format(type(chain)))
    if not pl.isstr(basepath):
        raise TypeError('`basepath` ({}) must be a str'.format(type(basepath)))

    if pl.isstr(true_subjset):
        true_subjset = pl.SubjectSet.load(true_subjset)
    if true_subjset is not None:
        if not pl.issubjectset(true_subjset):
            raise TypeError('`true_subjset` ({}) must be a pylab.SubjectSet'.format( 
                type(true_subjset)))

    if basepath[-1] != '/':
        basepath += '/'
    os.makedirs(basepath, exist_ok=True)
    main_output_filename = basepath + 'output.txt'

    f = open(main_output_filename, 'w')
    f.close()

    # Plot a0
    for aname in [STRNAMES.NEGBIN_A0, STRNAMES.NEGBIN_A1]:
        # Determine if there is a true value or not
        if aname == STRNAMES.NEGBIN_A0 and true_a0 is not None:
            trueval = true_a0
        elif aname == STRNAMES.NEGBIN_A1 and true_a1 is not None:
            trueval = true_a1
        else:
            trueval = None
        if chain.tracer.is_being_traced(aname):
            a = chain.graph[aname]
            f = open(main_output_filename, 'a')

            summ = pl.variables.summary(a, section='posterior')
            f.write('Negative Binomial Dispersion Parameter: {}\n'.format(a.name))
            f.write('-------------------------------------------\n')
            for k,v in summ.items():
                f.write('\t{}: {}\n'.format(k,v))

            axleft, axright = pl.visualization.render_trace(a, plt_type='both', 
                include_burnin=True, rasterized=True, log_scale=aname==STRNAMES.NEGBIN_A0)

            # Plot the acceptance rate on the right hand side
            ax2 = axright.twinx()
            ax2 = pl.visualization.render_acceptance_rate_trace(a, ax=ax2, 
                label='Acceptance Rate', color='red', scatter=False, rasterized=True)

            # Plot the true value if it is passed in
            if trueval is not None:
                axleft.axvline(trueval, color='green', linestyle='--', alpha=0.75)
                axright.axhline(trueval, color='green', linestyle='--', alpha=0.75)

            ax2.legend()

            fig = plt.gcf()
            fig.suptitle(a.name)
            fig.tight_layout()
            plt.savefig(basepath + '{}.pdf'.format(aname))
            plt.close()
            f.close()

    if chain.tracer.is_being_traced(STRNAMES.NEGBIN_A0) or chain.tracer.is_being_traced(STRNAMES.NEGBIN_A1):
        # Plot the learned negative binomial model

        reads= []
        for subj in subjset:
            reads.append(subj.matrix()['raw'])
        reads = np.hstack(reads)
        read_depths = np.sum(reads, axis=0)
        rels = reads / read_depths + 1e-10

        a0 = chain.graph[STRNAMES.NEGBIN_A0]
        a1 = chain.graph[STRNAMES.NEGBIN_A1]

        # Calculate the mean and variance using the noise model
        if chain.tracer.is_being_traced(STRNAMES.NEGBIN_A0):
            a0s = a0.get_trace_from_disk(section='posterior')
        else:
            a0s = a0.value * np.ones(chain.n_samples - chain.burnin)
        if chain.tracer.is_being_traced(STRNAMES.NEGBIN_A0):
            a1s = a1.get_trace_from_disk(section='posterior')
        else:
            a1s = a1.value * np.ones(chain.n_samples - chain.burnin)

        means = np.zeros(shape=(a0s.shape[0], rels.size), dtype=float)
        variances = np.zeros(shape=(a0s.shape[0], rels.size), dtype=float)

        for i in range(len(a0s)):
            _single_calc_mean_var(
                means=means[i,:],
                variances=variances[i,:],
                a0=a0s[i], a1=a1s[i], rels=rels, 
                read_depths=read_depths)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        # plot the data
        colors = sns.color_palette()
        for sidx, subj in enumerate(subjset):
            reads_subj = subj.matrix()['raw']

            x = np.mean(reads_subj, axis=1)
            y = np.var(reads_subj, axis=1)

            idxs = x > 0
            x = x[idxs]
            y = y[idxs]

            ax.scatter(
                x=x, y=y, alpha=0.5,
                c=colors[sidx], rasterized=False, 
                label='Subject {}'.format(subj.name))

        # Still need to get the 75th percentile, 25th percentile and the median

        summ_m = pl.variables.summary(means)
        summ_v = pl.variables.summary(variances)

        med_m = summ_m['median']

        med_v = summ_v['median']
        low_v = np.nanpercentile(variances, 2.5, axis=0)
        high_v = np.nanpercentile(variances, 97.5, axis=0)

        idxs = np.argsort(med_m)

        med_m = med_m[idxs]

        med_v = med_v[idxs]
        low_v = low_v[idxs]
        high_v = high_v[idxs]


        ax.plot(med_m, med_v, color='black', label='Fitted NegBin Model', rasterized=False)
        ax.fill_between(x=med_m, y1=low_v, y2=high_v, color='black', alpha=0.3, label='95th percentile')

        if true_subjset is not None:
            means_true = np.zeros(rels.size, dtype=float)
            vars_true = np.zeros(rels.size, dtype=float)
            _single_calc_mean_var(
                means=means_true,
                variances=vars_true,
                a0=true_a0, a1=true_a1, rels=rels, 
                read_depths=read_depths)
            idxs = np.argsort(means_true)
            means_true = means_true[idxs]
            vars_true = vars_true[idxs]
            ax.plot(means_true, vars_true, color='red', label='True NegBin Model', rasterized=False)

        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlabel('Mean (counts)')
        ax.set_ylabel('Variance (counts)')
        ax.set_title('Empirical mean vs variance of counts')
        ax.set_xlim(left=0.5)
        ax.set_ylim(bottom=0.5)
        ax.legend()
        plt.savefig(basepath + 'model_fit.pdf')
        plt.close()


    # Plot the latent trajectory
    asvs = chain.graph.data.asvs
    if chain.is_in_inference_order(STRNAMES.FILTERING):
        asvnames = asvs.names.order

        # For each replicate, make a 
        for ridx in range(chain.graph.data.n_replicates):

            if true_subjset is not None:
                true_ridxname = str(ridx)
                Mtruth = true_subjset[true_ridxname].matrix()['abs']
            else:
                true_ridxname = None


            subj_basepath = basepath + 'subject{}/'.format(ridx)
            subj = chain.graph.data.subjects.iloc(ridx)
            M_subj = subj.matrix()['abs']
            os.makedirs(subj_basepath, exist_ok=True)
            latent_name = STRNAMES.LATENT_TRAJECTORY + '_{}'.format(ridx)
            
            latent = chain.graph[latent_name]
            summ = pl.variables.summary(latent, section='posterior')

            M = subjset[str(ridx)].matrix()['abs']
            f = open(subj_basepath + 'output.txt', 'w')
            f.write('Subject {} output\n'.format(ridx))
            f.write('==================\n')

            for aidx, aname in enumerate(asvnames):
                f.write('\n\nASV {}: {}\n'.format(aidx, pl.asvname_formatter(format=asv_fmt,
                    asv=aname, asvs=asvs)))
                f.write('-----------------\n')

                # Write true value if true
                if true_ridxname is not None:
                    true_idx = true_subjset.asvs[aname].idx
                    f.write('True value: {:.4E}\n'.format(Mtruth[true_idx,0]))

                # Write what the data is
                f.write('Data: ')
                row = M[aidx, :]
                for ele in row:
                    f.write('{:.4E}  '.format(ele))
                f.write('\n')

                f.write('Learned Values:\n')
                for k,v in summ.items():
                    f.write('\t{}: {:.4E}\n'.format(k,v[aidx]))

                # plot the variable
                axpost, axtrace = pl.visualization.render_trace(latent, idx=aidx, plt_type='both', 
                    rasterized=True, log_scale=True)
                for idx in range(M_subj.shape[1]):
                    if idx == 0:
                        label = 'data'
                    else:
                        label = None
                    axpost.axvline(x=M_subj[aidx, idx], color='green', label=label)
                    axtrace.axhline(y=M_subj[aidx, idx], color='green', label=label)
                fig = plt.gcf()
                fig.suptitle( pl.asvname_formatter(format=asv_fmt,
                    asv=aname, asvs=asvs))
                plt.savefig(subj_basepath + '{}.pdf'.format(aname))
                plt.close()

            f.close()

def plot_linear_model(a0,a1):
    '''Plots the data with the parameters of the noise model
    to see if they fit each other.

    Parameters
    ----------
    a0, a1 : float
        These are the parameters learned from the inference
    '''
    reads = pd.read_csv(RAW_COUNTS_FILENAME, sep="\t", header=0)
    reads = reads.set_index('otuName')
    reads = reads.drop(columns=DADA_ASVSET_COL_NAMES)
    reads = reads.drop(columns=DADA_SEUQUENCE_COL_NAME)

    reads = reads.to_numpy()
    rels = reads / np.sum(reads, axis=0)
    read_depths = np.sum(reads, axis=0)

    # Calculate the mean and variance using the noise model
    means = []
    variances = []
    colors = []
    for col in range(rels.shape[1]):
        for oidx in range(rels.shape[0]):
            if col < 6:
                colors.append('blue')
            elif col < 12:
                colors.append('green')
            else:
                colors.append('brown')
            mean = rels[oidx, col] * read_depths[col]
            disp = a0 / mean + a1
            means.append(mean)
            variances.append(mean + (disp * (mean**2)))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    i = 0

    idxs = np.argsort(means)
    means = np.array(means)[idxs]
    variances = np.array(variances)[idxs]

    for color, day in [('red', 'Day 10'), ('blue', 'Day 8'), ('green', 'Day 9')]:
        colstart = i*6
        colend = (i+1)*6
        ax.scatter(
            x=np.mean(reads[:, colstart:colend], axis=1), 
            y=np.var(reads[:, colstart:colend], axis=1), 
            alpha=0.3,
            c=color, rasterized=True,
            label=day)
        i += 1
    ax.plot(means, variances, color='black', label='Fitted NegBin Model', rasterized=True)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Mean (counts)')
    ax.set_ylabel('Variance (counts)')
    ax.set_title('Empirical mean vs variance of counts')
    ax.legend()#bbox_to_anchor=(1,1))
    plt.savefig(OUTPUT_BASE_BASEPATH + 'model_fit.pdf')
    plt.show()
    plt.close()

def fit_dirichlet():
    '''Fits a dirichlet distribution to the samples
    '''
    reads = pd.read_csv(FILENAME, sep="\t", header=0)
    reads = reads.set_index('asvName')
    reads = reads.drop(columns=DADA_ASVSET_COL_NAMES)
    data = reads.to_numpy()

    # Make the relative abundance
    data = data / np.sum(data, axis=0)
    data = data.T

    D10 = data[:6,:100]
    idxs = []
    for col in range(D10.shape[1]):
        if np.all(D10[:,col] == 0):
            continue
        idxs.append(col)
    D10 = D10[:, idxs]
    print('D10.shape', D10.shape)
    a10 = dirichlet.mle(D=D10, maxiter=1000000)
    print(np.sum(a10))
    sys.exit()

    # print(D10[:,:10])

    # D8 = data[6:12, :]
    # idxs = []
    # for col in range(D8.shape[1]):
    #     if np.all(D8[:,col] == 0):
    #         continue
    #     idxs.append(col)
    # D8 = D8[:, idxs]
    # print('D8.shape', D8.shape)

    # D9 = data[12:, :]
    # idxs = []
    # for col in range(D9.shape[1]):
    #     if np.all(D9[:,col] == 0):
    #         continue
    #     idxs.append(col)
    # D9 = D9[:, idxs]
    # print('D9.shape', D9.shape)

    # # print(reads)
    # a10 = dirichlet.mle(D=D10, maxiter=1000000)
    # a9 = dirichlet.mle(D=D9, maxiter=1000000)
    # a8 = dirichlet.mle(D=D8, maxiter=1000000)
    # print(np.sum(a10))
    # print(np.sum(a9))
    # print(np.sum(a8))

if __name__ == '__main__':
    # Functionality
    LOAD_IF_POSSIBLE = False
    ONLY_PLOT = False

    config.LoggingConfig()

    args = parse_args()
    params = config.NegBinConfig(seed=args.data_seed, burnin=args.burnin, 
        n_samples=args.n_samples, synth=False, basepath=args.basepath)
    
    graph_name = 'graph_' + params.suffix()
    basepath = params.OUTPUT_BASEPATH + graph_name + '/'
    os.makedirs(basepath, exist_ok=True)

    chain_result_filename = basepath + config.MCMC_FILENAME
    subjset_filename = basepath + config.SUBJSET_FILENAME
    graph_filename = basepath + config.GRAPH_FILENAME
    hdf5_filename = basepath + config.HDF5_FILENAME
    tracer_filename = basepath + config.TRACER_FILENAME
    params_filename = basepath + config.PARAMS_FILENAME
    val_subjset_filename = basepath + config.VALIDATION_SUBJSET_FILENAME

    params.save(params_filename)

    # Build the data object if needed
    if not os.path.isfile(subjset_filename) or not LOAD_IF_POSSIBLE:
        logging.info('SubjectSet file does not exist in this directory. Build it')
        if params.SYNTHETIC:
            subjset, subjset_true = util.build_synthetic_subjset(params=params)
            subjset_true.save(val_subjset_filename)
        else:
            subjset = util.build_real_subjset(params=params)

        # Filter the subjectset
        util.filter_out_zero_asvs(subjset)
        subjset.save(subjset_filename)
    else:
        subjset = pl.base.SubjectSet.load(subjset_filename)

    if not ONLY_PLOT:
        logging.info('Run the model')

        # Visualize the data before hand
        logging.info('Run the model')
        chain_result = run(
            params=params,
            graph_name=graph_name,
            subjset_filename=subjset_filename,
            graph_filename=graph_filename,
            tracer_filename=tracer_filename,
            hdf5_filename=hdf5_filename,
            mcmc_filename=chain_result_filename)

        chain_result.save(chain_result_filename)
        params.save(params_filename)

    params = config.NegBinConfig.load(params_filename)
    chain_result = pl.inference.BaseMCMC.load(chain_result_filename)
    
    # Get the true synthetic parameters if possible
    if params.SYNTHETIC:
        val_subjset = val_subjset_filename
        true_a0 = params.SYNTHETIC_A0
        true_a1 = params.SYNTHETIC_A1
    else:
        val_subjset = None
        true_a0 = None
        true_a1 = None

    visualize_posterior(
        chain=chain_result_filename, 
        basepath=basepath+'posterior/',
        true_subjset=val_subjset, 
        true_a0=true_a0,
        true_a1=true_a1)