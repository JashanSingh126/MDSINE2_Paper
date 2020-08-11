import logging
import time
import collections
import itertools
import sys
import os
import h5py
import copy
import psutil
# import ray

import numpy as np
import numba
import scipy.sparse

import numpy.random as npr
import scipy.stats
import scipy
import math
import random

import matplotlib.pyplot as plt
import seaborn as sns

import pylab as pl
from names import STRNAMES, REPRNAMES

logging.basicConfig(
    format='%(levelname)s:%(module)s.%(lineno)s: %(message)s', 
    level=logging.INFO)

##############################################################################
# Global parameters
##############################################################################
ONAMES_TO_KEEP = ['ASV_1', 'ASV_2', 'ASV_3']
N_ASVS = len(ONAMES_TO_KEEP)
N_SAMPLES = 2000
BURNIN = 1000
MAX_DAY = 20
SEED = 100
BASEPATH = 'output_real/not_as_dumb_test/test{}_{}_{}_{}/'.format(BURNIN, N_SAMPLES, SEED, MAX_DAY)
os.makedirs(BASEPATH, exist_ok=True)

GROWTH_LOW = 0.1
GROWTH_HIGH = 2 * math.log(10)
SELF_INTERACTIONS_LOW = float('-inf')
SELF_INTERACTIONS_HIGH = 0

PV_V1 = 0.2 ** 2
C_M = np.log(5e6)

##############################################################################
# Inference and data classes
##############################################################################
def simulate(initial_conditions, growth, self_interactions, dt, times):
    '''Simulate with the given dynamics up to n_days with `dt` timesteps
    '''
    n_days = times[-1]
    n_time_steps = int(n_days/dt)+1
    M = np.zeros(shape=(len(growth), n_time_steps))
    M[:,0] = initial_conditions

    growth = growth.ravel()
    self_interactions.ravel()

    for i in range(1,n_time_steps):
        xt = (M[:,i-1])

        xtp1 = xt + dt * xt * (growth + self_interactions * xt)
        M[:,i] = xtp1

    # Subsample the timepoints to keep
    ts = np.arange(n_days+dt,step=dt)
    idxs = np.searchsorted(ts,times)
    
    return M[:,idxs]

class GrowthDM(pl.graph.Node):
    def __init__(self, **kwargs):
        kwargs['name'] = STRNAMES.GROWTH_VALUE + '_design_matrix'
        pl.graph.Node.__init__(self, **kwargs)
        self.G.data.design_matrices[STRNAMES.GROWTH_VALUE] = self
        self.matrix_w_perturbations = None
        self.matrix_wo_perturbations = None

    def build_without_perturbations(self):
        '''Only thing in here is the abundance
        '''
        n_dts = 0
        for times in self.G.data.times:
            n_dts += len(times)-1
        self.matrix_wo_perturbations = np.zeros(shape=(int(n_dts * self.G.data.n_asvs), self.G.data.n_asvs),dtype=float)
        for i, (ridx, tidx, oidx) in enumerate(self.G.data.iterate()):
            d = self.G.data.data[ridx]
            self.matrix_wo_perturbations[i,oidx] = d[oidx, tidx]

    def set_to_lhs(self):
        '''wp :: with perturbations
        '''
        b = self.G[STRNAMES.GROWTH_VALUE].value.reshape(-1,1)
        m = self.matrix_wo_perturbations
        return m @ b

    def set_to_rhs(self):
        return self.matrix_wo_perturbations

    def build(self):
        # self.build_with_perturbations()
        self.build_without_perturbations()


class SelfInteractionsDM(pl.graph.Node):
    def __init__(self, **kwargs):
        kwargs['name'] = STRNAMES.SELF_INTERACTION_VALUE + '_design_matrix'
        pl.graph.Node.__init__(self, **kwargs)
        self.G.data.design_matrices[STRNAMES.SELF_INTERACTION_VALUE] = self
        self.matrix = None

    def build(self):
        n_dts = 0
        for times in self.G.data.times:
            n_dts += len(times)-1
        self.matrix = np.zeros(shape=(int(n_dts * self.G.data.n_asvs), self.G.data.n_asvs),dtype=float)
        for i, (ridx, tidx, oidx) in enumerate(self.G.data.iterate()):
            d = self.G.data.data[ridx]
            self.matrix[i,oidx] = d[oidx, tidx]**2

    def set_to_lhs(self):
        b = self.G[STRNAMES.SELF_INTERACTION_VALUE].value.reshape(-1,1)
        return self.matrix @ b

    def set_to_rhs(self):
        return self.matrix


class LHSVector(pl.graph.Node):
    def __init__(self, **kwargs):
        pl.graph.Node.__init__(self, **kwargs)
        self.G.data.lhs = self
        self.value = None

    def build(self):
        n_dts = 0
        for times in self.G.data.times:
            n_dts += len(times)-1
        self.value = np.zeros(int(n_dts * self.G.data.n_asvs), dtype=float) * np.nan
        for i, (ridx, tidx, oidx) in enumerate(self.G.data.iterate()):
            d = self.G.data.data[ridx]
            times = self.G.data.times[ridx]
            dt = times[tidx+1] - times[tidx]
            self.value[i] = (d[oidx,tidx+1] - d[oidx,tidx])/dt

        self.value = self.value.reshape(-1,1)


class Data(pl.graph.DataNode):
    def __init__(self, subjset, **kwargs):
        kwargs['name'] = 'data'
        pl.graph.DataNode.__init__(self, **kwargs)

        self.asvs = subjset.asvs
        self.data = [np.log(subj.matrix()['abs']) for subj in subjset]
        for i in range(len(self.data)):
            self.data[i][np.isinf(self.data[i])] = C_M/2

        self.times = [subj.times for subj in subjset]
        self.subjset = subjset
        self.n_replicates = len(subjset)
        self.n_asvs = len(self.asvs)

        self.design_matrices = {}
        self.lhs = None

        self.timepoint2index = []
        for ridx in range(self.n_replicates):
            temp = {}
            for tidx, t in enumerate(self.times[ridx]):
                temp[t] = tidx
            self.timepoint2index.append(temp)

    def iterate(self):
        for ridx in range(self.n_replicates):
            for tidx in range(len(self.times[ridx])-1):
                for oidx in range(len(self.asvs)):
                    yield ridx, tidx, oidx

    def construct_lhs(self, keys=[], kwargs={}):
        y = self.lhs.value
        for key in keys:
            if key in kwargs:
                k = kwargs[key]
            else:
                k = {}
            b = self.design_matrices[key].set_to_lhs(**k)
            y = y-b
        return y

    def construct_rhs(self, keys, kwargs={}):
        v = []
        for key in keys:
            if key in kwargs:
                k = kwargs[key]
            else:
                k = {}
            X = self.design_matrices[key].set_to_rhs(**k)
            v.append(X)
        return np.hstack(v)


class Growth(pl.variables.TruncatedNormal):
    def __init__(self, prior, **kwargs):
        kwargs['name'] = STRNAMES.GROWTH_VALUE
        kwargs['dtype'] = float
        pl.variables.TruncatedNormal.__init__(self, mean=None, var=None, **kwargs)
        self.set_value_shape(shape=(len(self.G.data.asvs),))
        self.add_prior(prior)

    def __str__(self):
        return str(self.value)

    def initialize(self):
        return

    def update(self):
        # Create rhs and lhs
        lhs = [STRNAMES.SELF_INTERACTION_VALUE]
        rhs = [STRNAMES.GROWTH_VALUE]
        X = self.G.data.construct_rhs(keys=rhs)
        y = self.G.data.construct_lhs(keys=lhs)
        process_prec = self.G[STRNAMES.PROCESSVAR].precision
        prior_prec = self.prior.var.value

        prec = X.T @ process_prec @ X + prior_prec
        # prec = X.T @ X
        cov = np.linalg.pinv(prec)
        mean = np.asarray(cov @ X.T @ process_prec @ y).ravel()
        # mean = np.asarray(cov @ X.T @ y).ravel()

        self.mean.value = mean
        self.var.value = np.diag(cov)

        # print('--------------\ngrowth')
        # print(self.mean.value)
        # print(self.var.value)


        self.sample()


class SelfInteractions(pl.variables.TruncatedNormal):
    def __init__(self, prior, **kwargs):
        kwargs['name'] = STRNAMES.SELF_INTERACTION_VALUE
        kwargs['dtype'] = float
        pl.variables.TruncatedNormal.__init__(self, mean=None, var=None, **kwargs)
        self.set_value_shape(shape=(len(self.G.data.asvs),))
        self.add_prior(prior)

    def __str__(self):
        return str(self.value)

    def initialize(self):
        return

    def update(self):
        # Create rhs and lhs
        lhs = [STRNAMES.GROWTH_VALUE]
        rhs = [STRNAMES.SELF_INTERACTION_VALUE]
        X = self.G.data.construct_rhs(keys=rhs)
        y = self.G.data.construct_lhs(keys=lhs)
        process_prec = self.G[STRNAMES.PROCESSVAR].precision
        prior_prec = self.prior.var.value

        prec = X.T @ process_prec @ X + prior_prec
        # prec = X.T @ X
        cov = np.linalg.pinv(prec)
        mean = np.asarray(cov @ X.T @ process_prec @ y).ravel()
        # mean = np.asarray(cov @ X.T @ y).ravel()

        self.mean.value = mean
        self.var.value = np.diag(cov)

        # print('--------------\nself-interactions')
        # print(self.mean.value)
        # print(self.var.value)

        self.sample()


class Process(pl.Variable):
    def __init__(self, **kwargs):
        kwargs['name'] = STRNAMES.PROCESSVAR
        kwargs['dtype'] = float
        pl.Variable.__init__(self, **kwargs)
        self.c_m = None
        self.v1 = None

    def make(self):
        '''make the process variance and precision
        '''
        n_dts = 0
        for times in self.G.data.times:
            n_dts += len(times)-1
        variance = np.zeros(self.G.data.n_asvs * n_dts)*np.nan
        for i, (ridx, tidx, oidx) in enumerate(self.G.data.iterate()):
            M = self.G.data.data[ridx]
            # We use +1 because it is at the current timestep
            variance[i] = 1/(self.v1 * (M[oidx,tidx+1]**2) + (self.c_m**2))
        self.variance = np.diag(variance)
        self.precision = np.diag(1/variance)


##############################################################################
# Run
##############################################################################
# Load the data
logging.info('Loading the data')
subjset = pl.SubjectSet.load('pickles/real_subjectset.pkl')
subjset.pop_subject(['2','3','4','5'])

# Delete unecessary data
oids = []
for oname in subjset.asvs.names:
    if oname not in ONAMES_TO_KEEP:
        oids.append(oname)
subjset.pop_asvs(oids)
ts_to_delete = np.arange(MAX_DAY,70,step=0.5)
subjset.pop_times(ts_to_delete, sids='all')

# Set validation 
v = subjset.pop_subject('7')
val_subject = v['7']

for i, subj in enumerate(subjset):
    pl.visualization.abundance_over_time(subj=subj, dtype='abs', legend=True,
        taxlevel='genus', set_0_to_nan=True, yscale_log=True)
    plt.savefig(BASEPATH + 'data{}.pdf'.format(subj.name))
    plt.close()

GRAPH = pl.graph.Graph(name='graph_mer', seed=SEED)
GRAPH.as_default()

data = Data(subjset=subjset, G=GRAPH)

logging.info('Making growth and self interactions')
# Make growth and self-itneractions
growth_prior = pl.variables.TruncatedNormal(mean=0, var=10, low=GROWTH_LOW,
    high=GROWTH_HIGH, G=GRAPH)
growth_prior.var.value = np.diag(np.ones(GRAPH.data.n_asvs)/10)
growth = Growth(prior=growth_prior, low=GROWTH_LOW, high=GROWTH_HIGH, G=GRAPH)
growth.value = np.ones(data.n_asvs)

self_interactions_prior = pl.variables.TruncatedNormal(mean=0, var = 1,
    low=SELF_INTERACTIONS_LOW, high=SELF_INTERACTIONS_HIGH, G=GRAPH)
self_interactions_prior.var.value = np.diag(1/((
    np.quantile(np.hstack(GRAPH.data.data),q=.75, axis=1) ** 2)))
self_interactions = SelfInteractions(prior=self_interactions_prior, G=GRAPH)
self_interactions.value = -1/np.quantile(np.hstack(GRAPH.data.data),q=.75, axis=1)
process = Process(G=GRAPH)
process.v1 = PV_V1
process.c_m = C_M
process.make()

# Build the design matrices
logging.info('Building the design matrices')
growthDM = GrowthDM(G=GRAPH)
growthDM.build()
self_interactionsDM = SelfInteractionsDM(G=GRAPH)
self_interactionsDM.build()
y = LHSVector(G=GRAPH)
y.build()

# print('process precision')
# print(np.diag(process.variance))
# sys.exit()

# Start the inference
logging.info('Starting the inference')
mcmc = pl.inference.BaseMCMC(burnin=BURNIN, n_samples=N_SAMPLES, graph=GRAPH)
mcmc.set_inference_order(order=[STRNAMES.SELF_INTERACTION_VALUE, STRNAMES.GROWTH_VALUE])

mcmc.set_tracer(filename=BASEPATH+'tracer.hdf5', ckpt=200)
mcmc.tracer.set_save_location(BASEPATH+'tracer.pkl')
mcmc.set_save_location(BASEPATH+'mcmc.pkl')
GRAPH.set_save_location(BASEPATH+'graph.pkl')

mcmc = mcmc.run(log_every=1)

##############################################################################
# Plot the posterior
##############################################################################
for oidx in range(N_ASVS):
    pl.visualization.render_trace(var=growth, idx=oidx, rasterized=True)
    fig = plt.gcf()
    fig.suptitle('ASV {} growth'.format(oidx))
    plt.savefig(BASEPATH + 'growth{}.pdf'.format(oidx))
    plt.close()

    pl.visualization.render_trace(var=self_interactions, idx=oidx, log_scale=True, rasterized=True)
    fig = plt.gcf()
    fig.suptitle('ASV {} self_interactions'.format(oidx))
    plt.savefig(BASEPATH + 'self_interactions{}.pdf'.format(oidx))
    plt.close()

##############################################################################
# Forward Simulate
##############################################################################
val_m = np.log(val_subject.matrix()['abs'])
print('val_m.shape', val_m.shape)
initial_conditions = val_m[:,0]
n_timepoints = int(MAX_DAY / .001)
output = np.zeros(shape=(N_SAMPLES - BURNIN, N_ASVS, len(val_subject.times)))

growth_trace = growth.get_trace_from_disk()
self_interactions_trace = (self_interactions.get_trace_from_disk())

for i in range(BURNIN, N_SAMPLES):
    print('{}/{}'.format(i-BURNIN,N_SAMPLES-BURNIN))
    output[i-BURNIN] = simulate(
        initial_conditions=initial_conditions, 
        growth=growth_trace[i-BURNIN], 
        self_interactions=self_interactions_trace[i-BURNIN], 
        times=val_subject.times, dt=0.001)

# colors = sns.color_palette(n_colors=N_ASVS)
times = val_subject.times
pred_low = np.nanpercentile(a=output, q=5, axis=0)
pred_high = np.nanpercentile(a=output, q=95, axis=0)
pred_med = np.nanpercentile(a=output, q=50, axis=0)

mean_growth = np.mean(growth_trace, axis=0)
mean_si = np.mean(self_interactions_trace, axis=0)
for oidx in range(N_ASVS):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.fill_between(times, y1=pred_low[oidx], y2=pred_high[oidx], color='blue', alpha=0.15)
    ax.plot(times, pred_med[oidx], label='Predicted', color='blue', marker='o')
    ax.plot(times, val_m[oidx,:], color='black', linestyle=':', marker='x', label='data')
    ax.axhline(y=-mean_growth[oidx]/mean_si[oidx], label='steady-state', color='orange', alpha=0.5)

    ax.set_yscale('log')
    ax.set_title('RMSE: {:.3E}'.format(pl.metrics.RMSE(pred_med[oidx], val_m[oidx,:])))

    plt.savefig(BASEPATH + 'val{}.pdf'.format(oidx))
    plt.close()




