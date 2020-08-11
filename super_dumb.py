'''How to learn growths in the real data

Model
-----
x_{k+1,i} - x_{k,i} = x_{k,i} * (a_{1,i} - x_{k,i} * a_{2,i}) * \delta_{t}, where
    - k \in N_{>0} indexes time as t_k
    - \delta_{t} = t_{k+1} - t_{k}
    - i indexes the ASV index

a_{1,i}, a_{2,i} ~ Normal(\mu_{2}, \Sigma_{2})
(x_{k+1,i} - x_{k,i}) / \delta_{t} ~ Normal( x_{k,i} * a_{1,i} + x_{k,i}^{2} * a_{2,i} , \Sigma_{1} ), where
    - Sigma_{1,k,i} = v_{1} * x_{k,i}^2 + c_{m}^2
        - v_{1} = 0.04, c_{m} = 5e6

Bayesian Linear Regression
--------------------------
Posterior mean and variance of the parameters can be calculated as follows:

\Sigma_3 = (X.T @ \Sigma_{1}^{-1} @ X + \Sigma_{2}^{-1})^{-1}
mean = \Sigma_3 @ (X.T @ \Sigma_{1}^{-1} @ y + \Sigma_{2}^{-1} @ \mu_{2})


'''
import logging
import numpy as np
import sys
import os
import math
import os.path
import pickle
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import pylab as pl
# import config
# import posterior
# import main_base
# import diversity.alpha
# from names import STRNAMES
# import filtering

# Global parameters
N_SAMPLES = 750
N_BURNIN = 700
DIVISOR = 1e10 #1e12
OIDXS = [0,1,2,20]
N_ASVS = len(OIDXS)
MAX_DAY = 20

GROWTH_LOW = .1
GROWTH_HIGH = 3 * np.log(10) #float('inf') #4 * np.log(10)
LEARN_GROWTH = True

SELF_INTERACTION_LOW = 0 
SELF_INTERACTION_HIGH = float('inf')
LEARN_SI = True

SUBJECTED_FOR_VAL = '10'

BASEPATH = 'output_real/extremely_dumb/'
os.makedirs(BASEPATH, exist_ok=True)

C_M = (5e6/DIVISOR)
PV_V1 = .2 ** 2
REAL = True

LAMBDA_GROWTH = 1e-1
LAMBDA_SELF_INTERACTIONS = 1e-10

#######################################
# Inference functions
class Growth:

    def __init__(self):
        self.low = GROWTH_LOW
        self.high = GROWTH_HIGH
        self.mean = np.zeros(N_ASVS) * np.nan
        self.var = np.zeros(N_ASVS) * np.nan
        self.value = None

        self.prior_mean = None
        self.prior_var = None

        self.trace = np.zeros(shape=(N_SAMPLES, N_ASVS)) * np.nan
        self.trace_i = 0

    def update(self, self_interactions, data):
        '''
        data : data object
        self_interactions : self-interactions object
        '''
        if LEARN_GROWTH:
            # Update the growths one at a time

            X = data.growth_X

            process_prec = (data.process_prec)
            y = data.y - (data.self_interactions_X @ (self_interactions.value.reshape(-1,1)))
            prior_prec = np.diag(1/self.prior_var)


            prior_mean = self.prior_mean.reshape(-1,1)

            prec = X.T @ process_prec @ X + prior_prec
            cov = np.linalg.pinv(prec)
            mean = cov @ (X.T @ process_prec @ y + (prior_prec @ prior_mean))

            # prec = X.T @ X + np.diag(np.ones(N_ASVS))*LAMBDA_GROWTH
            # cov = np.linalg.pinv(prec)
            # mean = cov @ (X.T @ y)

            self.mean = np.asarray(mean).ravel()
            self.var = np.diag(cov)

            # sys.exit()

    def sample(self):
        self.value = pl.random.truncnormal.sample(mean=self.mean, std=np.sqrt(self.var),
            low=self.low, high=self.high)
        if not pl.isarray(self.value):
            self.value = np.array([self.value])

    def set_trace(self):
        self.trace[self.trace_i] = self.value
        self.trace_i += 1

    def redo(self):
        self.trace = np.zeros(shape=(N_SAMPLES, N_ASVS)) * np.nan
        self.trace_i = 0


class Self_interactions:
    
    def __init__(self):
        self.low = SELF_INTERACTION_LOW
        self.high = SELF_INTERACTION_HIGH
        self.mean = np.zeros(N_ASVS) * np.nan
        self.var = np.zeros(N_ASVS) * np.nan
        self.value = None

        self.prior_mean = None
        self.prior_var = None

        self.trace = np.zeros(shape=(N_SAMPLES, N_ASVS)) * np.nan
        self.trace_i = 0

    def update(self, growth, data):
        '''
        data : data object
        growth : growth object
        '''
        if LEARN_SI:
            X = data.self_interactions_X
            process_prec = data.process_prec

            mer = data.growth_X @ (growth.value.reshape(-1,1))

            y = data.y - mer
            prior_prec = np.diag(1/self.prior_var)
            prior_mean = self.prior_mean.reshape(-1,1)

            prec = X.T @ process_prec @ X + prior_prec
            cov = np.linalg.pinv(prec)
            mean = cov @ (X.T @ process_prec @ y + prior_prec @ prior_mean)

            # prec = X.T @ X + np.diag(np.ones(N_ASVS))*LAMBDA_SELF_INTERACTIONS
            # cov = np.linalg.pinv(prec)
            # mean = cov @ X.T @ y

            self.mean = np.asarray(mean).ravel()
            self.var = np.diag(cov)

    def sample(self):
        self.value = pl.random.truncnormal.sample(mean=self.mean, std=np.sqrt(self.var),
            low=self.low, high=self.high)
        if not pl.isarray(self.value):
            self.value = np.array([self.value])

    def set_trace(self):
        self.trace[self.trace_i] = self.value
        self.trace_i += 1

    def redo(self):
        self.trace = np.zeros(shape=(N_SAMPLES, N_ASVS)) * np.nan
        self.trace_i = 0


class Data:
    def __init__(self, Ms, times, n_asvs):
        self.matrices = Ms
        self.times = times
        self.n_replicates = len(self.matrices)
        self.n_asvs = n_asvs

        self.pv_v1 = None
        self.pv_c_m = None

        self.total_n_dts = 0
        for ts in self.times:
            self.total_n_dts += len(ts)-1

    def __len__(self):
        n_times = 0
        for ridx in range(self.n_replicates):
            n_times += len(self.times[ridx])-1
        return n_times

    def __iter__(self):
        for ridx in range(self.n_replicates):
            for tidx in range(len(self.times[ridx])-1):
                for oidx in range(N_ASVS):
                    yield oidx, tidx, ridx

    def build(self):
        self.build_y()
        self.build_pv()
        self.build_growth()
        self.build_self_interactions()

    def build_y(self):
        self.y = []
        for oidx, tidx, ridx in self:
            d = self.matrices[ridx]
            times = self.times[ridx]
            dt = times[tidx+1] - times[tidx]
            self.y.append((d[oidx, tidx+1]-d[oidx, tidx])/dt)

        self.y = np.array(self.y).reshape(-1,1)

    def build_pv(self):
        self.process_var = []
        v1 = self.pv_v1
        cm = self.pv_c_m
        for oidx, tidx, ridx in self:
            # q = (self.matrices[ridx][oidx, tidx] + self.matrices[ridx][oidx, tidx+1])/2 
            # q = (self.matrices[ridx][oidx, tidx] * self.matrices[ridx][oidx, tidx+1])**.5
            q = self.matrices[ridx][oidx, tidx]
            self.process_var.append(v1*(q**2) + (cm**2))
        self.process_var = np.array(self.process_var)
        self.process_prec = 1/self.process_var
        self.process_var = np.diag(self.process_var)
        self.process_prec = np.diag(self.process_prec)

        # self.process_var = np.append(
        #     np.append(1e-4 * np.ones(N_ASVS), 1e-2 * np.ones(N_ASVS)), 
        #     # 1e-5 * np.ones(N_ASVS),
        #     np.ones(self.total_n_dts*N_ASVS - 2*N_ASVS))
        # self.process_prec = 1/self.process_var

        # self.process_var = np.diag(self.process_var)
        # self.process_prec = np.diag(self.process_prec)
            


    def build_growth(self):
        self.growth_X = np.zeros(shape=(N_ASVS * self.total_n_dts, N_ASVS))
        for i, (oidx, tidx, ridx) in enumerate(self):
            d = self.matrices[ridx]
            self.growth_X[i, oidx] = d[oidx, tidx]
            

    def build_self_interactions(self):
        self.self_interactions_X = np.zeros(shape=(N_ASVS * self.total_n_dts, N_ASVS))
        for i, (oidx, tidx, ridx) in enumerate(self):
            d = self.matrices[ridx]
            self.self_interactions_X[i, oidx] = - d[oidx, tidx] ** 2


def simulate(initial_conditions, growth, self_interactions, dt, times, pv=False):
    '''Simulate with the given dynamics up to n_days with `dt` timesteps

    if pv is True, add process variance in the integration
    '''
    n_days = times[-1]
    n_time_steps = int(n_days/dt)+1
    M = np.zeros(shape=(len(growth), n_time_steps))
    M[:,0] = initial_conditions

    growth = growth.ravel()
    self_interactions.ravel()

    for i in range(1,n_time_steps):
        xt = (M[:,i-1])

        xtp1 = xt + dt * xt * (growth - self_interactions * xt)
        if pv:
            std = np.sqrt((PV_V1 * (xtp1**2) + C_M**2)*dt)
            xtp1 = pl.random.truncnormal.sample(xtp1, std=std, low=0)
        M[:,i] = xtp1

    # Subsample the timepoints to keep
    ts = np.arange(n_days+dt,step=dt)
    idxs = np.searchsorted(ts,times)
    
    return M[:,idxs]

def sample(growth, self_interactions, data):
    '''Sample the growth and self-interactions jointly
    '''
    X = np.hstack((data.growth_X, data.self_interactions_X))
    y = data.y
    process_prec = data.process_prec

    prior_prec = np.diag(np.append(1/growth.prior_var, 1/self_interactions.prior_var))
    prior_mean = np.append(growth.prior_mean, self_interactions.prior_mean).reshape(-1,1)

    prec = X.T @ process_prec @ X + prior_prec
    cov = np.linalg.pinv(prec)
    mean = cov @ (X.T @ process_prec @ y + prior_prec @ prior_mean)
    # prec = X.T @ X
    # cov = np.linalg.pinv(prec)
    # mean = cov @ X.T @ y

    mean = np.asarray(mean).ravel()
    var = np.diag(cov)

    # print(mean)
    # print(var)

    growth.mean = mean[:N_ASVS]
    self_interactions.mean = mean[N_ASVS:]
    growth.var = var[:N_ASVS]
    self_interactions.var = var[N_ASVS:]


#######################################
# Load and make the data
# Get times from data
subjset = pl.SubjectSet.load('pickles/real_subjectset.pkl')
# Delete all the asvs that are not in `KEEP`
subjset.pop_subject(['2','3','4','5'])
# Delete all the timepoints after time MAX_DAY
ts_to_delete = np.arange(MAX_DAY,70,step=0.5)
subjset.pop_times(ts_to_delete, sids='all')
# subjset.pop_times([0], sids='all')

val_subject = subjset.pop_subject(SUBJECTED_FOR_VAL)[SUBJECTED_FOR_VAL]

if REAL:
    val_m = (val_subject.matrix()['abs'][OIDXS, :])/DIVISOR
    val_times = val_subject.times

    Ms = [val_m]
    ts = [val_times]

    # Ms = []
    # ts = []
    # for i, subj in enumerate(subjset):
    #     if True:
    #         d = (subj.matrix()['abs'][OIDXS, :])/DIVISOR
            
    #         # for tidx in range(d.shape[1]-1):
    #         #     if np.any(d[:,tidx+1] < d[:,tidx]):
    #         #         d[:, tidx+1] = d[:, tidx]
    #         Ms.append(d)
    #         ts.append(subj.times)    
else:
    # Need Ms, ts, val stuff

    ts = []
    for subj in subjset:
        ts.append(subj.times)

    # We hae the ts, now lets make the data
    growths = np.array([3.5]) #2.2, 3.5, 2.6])
    self_interactions = np.array([-2e-12*DIVISOR, -1e-11*DIVISOR, -5e-10*DIVISOR])

    Ms = []
    for subj in subjset:
        init_conditions = subj.matrix()['abs'][:,0]/DIVISOR
        Ms.append(simulate(init_conditions, growth=growths, 
            self_interactions=self_interactions, dt=0.001, 
            times=subj.times, pv=True))

    val_times = val_subject.times
    val_m = simulate(initial_conditions=val_subject.matrix()['abs'][:,0]/DIVISOR, 
        growth=growths, 
        self_interactions=self_interactions, dt=0.001, 
        times=subj.times, pv=True)

data = Data(Ms, ts, N_ASVS)
data.pv_v1 = PV_V1
data.pv_c_m = C_M
data.build()


#######################################
# Set up
GROWTH = Growth()
GROWTH.value = 4*np.ones(N_ASVS) #np.array([2.4406, 2.2602, 2.6904]) #np.ones(N_ASVS, dtype=float)*3
GROWTH.prior_mean = np.ones(N_ASVS, dtype=float) * 4
GROWTH.prior_var = np.ones(N_ASVS, dtype=float)*1

SELF_INTERACTIONS = Self_interactions()
SELF_INTERACTIONS.value = np.ones(N_ASVS)* 10 #np.array([5.2535e-11, 4.6119e-11]) #1/np.quantile(np.hstack(Ms),q=.75, axis=1) #np.array([3.01915e-11, 5.2535e-11, 4.6119e-11])  #
SELF_INTERACTIONS.prior_mean = np.ones(N_ASVS, dtype=float) * 1 #1/np.quantile(np.hstack(Ms),q=.75, axis=1)
SELF_INTERACTIONS.prior_var = np.ones(N_ASVS, dtype=float) * 1 #100/(np.quantile(np.hstack(Ms),q=.75, axis=1)**2)

for ridx in range(len(Ms)):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = sns.color_palette(n_colors=N_ASVS)
    for oidx in range(N_ASVS):

        lbda_growth = []
        lbda_self_interactions = []
        for tidx in range(len(ts[ridx]) - 1):
            lbda_growth.append((PV_V1*(Ms[ridx][oidx,tidx]**2) + C_M**2)/GROWTH.prior_var[oidx])
            lbda_self_interactions.append((PV_V1*(Ms[ridx][oidx,tidx]**2) + C_M**2)/SELF_INTERACTIONS.prior_var[oidx])

        ax.plot(ts[ridx], Ms[ridx][oidx,:], label='{}'.format(oidx), marker='o', 
            color=colors[oidx])
        ax.plot(ts[ridx][1:], lbda_growth, label='{}-growth'.format(oidx), color=colors[oidx], marker='o',
            linestyle=':')
        ax.plot(ts[ridx][1:], lbda_self_interactions, label='{}-self-interactions'.format(oidx), color=colors[oidx], marker='o',
            linestyle='-.')
        
    ax.set_yscale('log')
    ax.legend()
    if not REAL:
        for oidx in range(N_ASVS):
            ax.axhline(y=(-growths[oidx]/self_interactions[oidx]), 
                color=colors[oidx], alpha=0.5, linestyle=':')
    plt.savefig(BASEPATH + 'Basedata{}.pdf'.format(ridx))
    plt.close()

# print('SELF_INTERACTIONS init')
# print(SELF_INTERACTIONS.value)
# print('GROWTH init')
# print(GROWTH.value)

# print(np.diag(data.process_prec))
# sys.exit()

#######################################
# Run inference
for a in [0]: #, 1e10, 1e20, 1e22, 1.5e22, 2e22, 5e22, 8e22, 1e24, 1e26, 1e28, 1e30]:
    # LAMBDA_GROWTH = a
    # LAMBDA_SELF_INTERACTIONS = a
    print('-------------------\nlambda',a)
    GROWTH.redo()
    SELF_INTERACTIONS.redo()

    for i in range(N_SAMPLES):
        # print('\n\n\n\n{}------------'.format(i))

        SELF_INTERACTIONS.update(GROWTH, data)
        GROWTH.update(SELF_INTERACTIONS, data)
        # sample(GROWTH, SELF_INTERACTIONS, data)

        if LEARN_GROWTH:
            GROWTH.sample()
        if LEARN_SI:
            SELF_INTERACTIONS.sample()
        GROWTH.set_trace()
        SELF_INTERACTIONS.set_trace()

    print('Expected values for growth', np.mean(GROWTH.trace, axis=0))
    print('Expected values for self-interactions', np.mean(SELF_INTERACTIONS.trace, axis=0))

# sys.exit()
    

#######################################
# Plot runs
os.makedirs(BASEPATH, exist_ok=True)
for oidx in range(N_ASVS):
    pl.visualization.render_trace(var=GROWTH.trace, idx=oidx, n_burnin=N_BURNIN)
    fig = plt.gcf()
    fig.suptitle('ASV {} growth'.format(oidx))
    plt.savefig(BASEPATH + 'growth{}.pdf'.format(oidx))
    plt.close()

    pl.visualization.render_trace(var=SELF_INTERACTIONS.trace, idx=oidx, n_burnin=N_BURNIN, log_scale=True)
    fig = plt.gcf()
    fig.suptitle('ASV {} self_interactions'.format(oidx))
    plt.savefig(BASEPATH + 'self_interactions{}.pdf'.format(oidx))
    plt.close()

#######################################
# Forward Simulate

initial_conditions = val_m[:,0]

n_timepoints = int(MAX_DAY / .001)
output = np.zeros(shape=(N_SAMPLES - N_BURNIN, N_ASVS, len(val_times)))
for i in range(N_BURNIN, N_SAMPLES):
    if i % 50 == 0:
        print('{}/{}'.format(i-N_BURNIN,N_SAMPLES-N_BURNIN))
    output[i-N_BURNIN] = simulate(initial_conditions=initial_conditions, growth=GROWTH.trace[i], 
        self_interactions=SELF_INTERACTIONS.trace[i], times=val_times, dt=0.001)

output = output


colors = sns.color_palette(n_colors=N_ASVS)
times = val_times
pred_low = np.nanpercentile(a=output, q=5, axis=0)
pred_high = np.nanpercentile(a=output, q=95, axis=0)
pred_med = np.nanpercentile(a=output, q=50, axis=0)

mean_growth = np.mean(GROWTH.trace[N_BURNIN:], axis=0)
mean_si = np.mean(SELF_INTERACTIONS.trace[N_BURNIN:], axis=0)

for oidx in range(N_ASVS):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    color = colors[oidx]

    ax.fill_between(times, y1=pred_low[oidx], y2=pred_high[oidx], color=color, alpha=0.15)
    ax.plot(times, pred_med[oidx], label='Predicted', color=color, marker='o')
    ax.plot(times, (val_m[oidx,:]), color=color, linestyle=':', marker='x', label='data')

    # for ridx, M in enumerate(data.matrices):
    #     ax.plot(data.times[ridx], M[oidx, :], color=color, linestyle=':', marker='o', alpha=0.5)


    ax.axhline(y=(-mean_growth[oidx]/mean_si[oidx]), label='steady-state', color=color, alpha=0.5)

    ax.set_yscale('log')

    plt.savefig(BASEPATH + 'val{}.pdf'.format(oidx), transparent=True)
    plt.close()
    






