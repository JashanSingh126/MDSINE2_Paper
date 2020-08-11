import logging
import numpy as np
import sys
import os
import math
import os.path
import pickle
import pandas as pd

import sklearn
import scipy


import matplotlib.pyplot as plt
import seaborn as sns

import pylab as pl

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

        xtp1 = xt + dt * xt * (growth + self_interactions * xt)
        if pv:
            std = np.sqrt((PV_V1 * (xtp1**2) + C_M**2)*dt)
            xtp1 = pl.random.truncnormal.sample(xtp1, std=std, low=0)
        M[:,i] = xtp1

    # Subsample the timepoints to keep
    ts = np.arange(n_days+dt,step=dt)
    idxs = np.searchsorted(ts,times)
    
    return M[:,idxs]


subjset = pl.SubjectSet.load('pickles/real_subjectset.pkl')
subjset.pop_subject(['2','3','4','5']) #,'6','7','8','9'])
subjset.pop_times(np.arange(20,70,step=0.5), sids='all')

# Ms = []
# master_times = []
# for subj in subjset:
#     y = subj.matrix()['abs'][1,:]/1e10
#     x = subj.times

#     f1 = scipy.interpolate.interp1d(x,y,kind='linear')
#     x = np.arange(x[0], x[-1], step=0.25)

#     y = f1(x)
#     Ms.append(y)
#     master_times.append(x)

Ms = []
Ms = [subjset['10'].matrix()['abs'][0,:]/1e10 ]
master_times = [subjset['10'].times]

total_times = 0
for times in master_times:
    total_times += len(times) - 1

# for i, M in enumerate(Ms):
#     print('\n\nSubject', i)
#     print('data\n',M)
#     print('times\n', master_times[i])

# Construct y
y = np.zeros(total_times)
i = 0
for ridx, M in enumerate(Ms):
    for tidx in range(len(master_times[ridx]) - 1):
        times = master_times[ridx]
        a = ((M[tidx+1]) - (M[tidx]))/(times[tidx+1]-times[tidx])
        y[i] = a
        i += 1

# Construct pv
process_prec = np.diag(np.append(
    1e5,
    np.ones((total_times)-1)))
prior_prec = np.diag(np.ones(2))

y = np.array(y).reshape(-1,1)
# print('y')
# print(y)

# Construct X
X = np.zeros(shape=(total_times,2))
i = 0
for ridx, M in enumerate(Ms):
    for tidx in range(len(master_times[ridx]) - 1):
        X[i,0] = M[tidx]
        X[i,1] = M[tidx] ** 2
        i += 1
# print('X')
# print(X)

# do regression
prec = X.T @ process_prec @ X + prior_prec

cov = np.linalg.pinv(prec)
value = cov @ ( X.T @ process_prec @ y )

growth = np.asarray([value[0]])
si = np.asarray([value[1]])
ic = np.array([Ms[0][0]])

print('ic', ic)
print('growth', growth)
print('si', si)

M = simulate(initial_conditions=ic, growth=growth, 
    self_interactions=si, dt=0.001, times=np.arange(18), pv=False)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(np.arange(18), M.ravel())
ax.plot(master_times[0], Ms[0])
# ax.axhline(y=-growth/si)
ax.set_yscale('log')


# # SKlearn
# clf = sklearn.linear_model.Ridge(alpha=[0,0])
# clf.fit(X,y)
# print(clf.get_params())



plt.show()
print(value)

