'''Memory and computationally efficient script for time lookahead for MDSINE2. Not
designed to be flexible, just efficient and readable.
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
import argparse

import matplotlib.pyplot as plt
import seaborn as sns

import pylab as pl

logging.basicConfig(level=logging.INFO)

pred_format = 'MDSINE2-subj{subjname}-tla{n_days}-start{start}-pred.npy'
truth_format = 'MDSINE2-subj{subjname}-tla{n_days}-start{start}-truth.npy'

subject_filename = 'subject.pkl'
growth_filename = 'growth.npy'
self_interactions_filename = 'self_interactions.npy'
interactions_filename = 'interactions.npy'
perturbationX_filename = 'perturbation{pidx}.npy'



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--n-days', type=float, dest='n_days',
        help='Maximum number of days to look ahead')
    parser.add_argument('--times-to-start-on', type=float, dest='times_to_start_on', nargs='+',
        help='comma separated times to start the forward simulation on')
    parser.add_argument('--input-basepaths', type=str, dest='input_basepath',
        help='The path to find the growth, interaction, and perturbation traces as wll as the subject')
    parser.add_argument('--validation-subject', type=str, dest='input',
        help='Pylab.base.Subject obejct we want to do the inference over')
    parser.add_argument('--output-basepath', type=str, dest='output_basepath',
        help='Where to save the outputs')
    parser.add_argument('--simulation-dt', type=float, dest='simulation_dt',
        help='Timesteps we go in during forward simulation', default=0.01)
    parser.add_argument('--max-posterior', type=int, dest='max_posterior',
        help='TESTING USE ONLY', default=None)

    args = parser.parse_args()
    return args

class EfficientgLVDynamics(pl.BaseDynamics):

    def __init__(self, asvs, sim_max, growth, interactions, perturbations=None,
        perturbation_starts=None, perturbation_ends=None):
        '''
        '''
        pl.BaseDynamics.__init__(self, asvs=asvs)

        self.sim_max = sim_max

        self.growth = growth
        self.interactions = interactions
        self.perturbations = perturbations
        self.perturbation_starts = perturbation_starts
        self.perturbation_ends = perturbation_ends
        self._pert_intervals = None
        self._adjusted_growth = None

    def init_integration(self):

        self.growth = self.growth.reshape(-1,1)
        if self.perturbations is not None:
            self._adjust_growth = []
            for pert in self.perturbations:
                pert = pert.reshape(-1,1)
                self._adjust_growth.append(self.growth * (1 + pert))

    def integrate_single_timestep(self, x, t, dt):
        '''Integrate over a single step

        Parameters
        ----------
        x : np.ndarray((n,1))
            This is the abundance as a column vector for each ASV
        t : numeric
            This is the time point we are integrating to
        dt : numeric
            This is the amount of time from the previous time point we
            are integrating from
        '''
        growth = self.growth

        if self.perturbations is not None:

            # Initialize pert_intervals
            if self._pert_intervals is None:
                # float -> int
                # timepoint -> perturbation index

                self._pert_intervals = {}
                for pidx in range(len(self.perturbation_ends)):
                    start = self.perturbation_starts[pidx]
                    end = self.perturbation_ends[pidx]
                    rang = np.arange(start, end, step=dt)

                    for t in rang:
                        self._pert_intervals[t] = pidx
            
            if t-dt in self._pert_intervals:
                growth = self._adjust_growth[self._pert_intervals[t]]

        # Integrate
        logret = np.log(x) + (growth + self.interactions @ x) * dt
        ret = np.exp(logret).ravel()
        ret[ret >= self.sim_max] = self.sim_max
        return ret
    
    def finish_integration(self):
        self._pert_intervals = None
        self._adjusted_growth = None


def forward_simulate(growth, self_interactions, interactions, perturbations, dt, 
    subject, start, n_days, output_basepath, limit_of_detection=1e5):
    '''Forward simulate using the specified parameters

    Parameters
    ----------
    growth : np.ndarray (n_samples, n_asvs)
    self_interactions : np.ndarray (n_samples, n_asvs)
    interactions : np.ndarray (n_samples, n_asvs, n_asvs)
    perturbations : list(np.ndarray (n_samples, n_asvs))
    dt : float
        Simulation time step
    subject : pylab.base.Subject
        Data
    start : float
        Time to start at
    n_days : float
        Max number of days to simulate to
    output_basepath : str
        basepath to save the output
    '''
    # Set the data
    asvs = subject.asvs

    if start not in subject.times:
        raise ValueError('`start` ({}) not found'.format(start))
    
    startidx = None
    endidx = None
    for tidx, t in enumerate(subject.times):
        if t == start:
            startidx = tidx
        if t > start + n_days:
            break
        endidx = tidx + 1 # Add 1 so that it is included

    time_look_aheads = subject.times[startidx:endidx] - start

    # Make reverse index matrix for subject
    t2tidx = {}
    for i, t in enumerate(subject.times):
        t2tidx[t] = i
    M = subject.matrix()['abs']

    # Prepare the variables for forward integration
    self_interactions = -np.absolute(self_interactions)
    for i in range(len(asvs)):
        interactions[:, i, i] = self_interactions[:, i]
    perturbation_starts = [pert.start - start for pert in subject.perturbations]
    perturbation_ends = [pert.end - start for pert in subject.perturbations]

    n_samples = growth.shape[0]
    n_asvs = len(asvs)
    n_times = len(time_look_aheads)
    pred_matrix = np.zeros(shape=(n_samples, n_asvs, n_times), dtype=float)

    n_days = np.max(time_look_aheads)
    initial_conditions = M[:, [t2tidx[start]]]
    initial_conditions[initial_conditions == 0] = limit_of_detection
    initial_conditions = initial_conditions.reshape(-1,1)

    # Perform inference
    for gibb_step in range(n_samples):
        dyn = EfficientgLVDynamics(
            asvs=asvs, 
            growth=growth[gibb_step], 
            interactions=interactions[gibb_step], 
            perturbations=[pert[gibb_step] for pert in perturbations], 
            perturbation_starts=perturbation_starts,
            perturbation_ends=perturbation_ends, sim_max=1e20)
    
        output = pl.dynamics.integrate(dynamics=dyn, initial_conditions=initial_conditions,
            dt=dt, n_days=n_days+dt, subsample=True, times=time_look_aheads)
        pred_matrix[gibb_step] = output['X']
    
    # Save the predictions
    os.makedirs(output_basepath, exist_ok=True)
    if output_basepath[-1] != '/':
        output_basepath += '/'
    
    for col, t in enumerate(time_look_aheads):
        if t == 0:
            continue

        # check if the time exists in subject
        if t + start not in subject.times:
            continue
        
        # Save the prediction
        arr = pred_matrix[:,:,col]
        path = output_basepath + pred_format.format(subjname=subject.name, n_days=t, start=start)
        np.save(path, arr)

        # Save the truth
        tidx = t2tidx[t+start]
        arr = M[:, tidx]
        path = output_basepath + truth_format.format(subjname=subject.name, n_days=t, start=start)
        np.save(path, arr)
        
if __name__ == '__main__':

    args = parse_args()

    # Load the inference parameters
    basepath = args.input_basepaths
    if basepath[-1] != '/':
        basepath += '/'

    subject = pl.Subject.load(basepath + subject_filename)
    growth = np.load(basepath + growth_filename)
    self_interactions = np.load(basepath + self_interactions_filename)
    interactions = np.load(basepath + interactions_filename)
    perturbation0 = np.load(basepath + perturbationX_filename.format(0))
    perturbation1 = np.load(basepath + perturbationX_filename.format(1))
    perturbation2 = np.load(basepath + perturbationX_filename.format(2))
    perturbations = [perturbation0, perturbation1, perturbation2]

    for t in args.times_to_start_on:

        if t not in subject.times:
            raise ValueError('t ({}) not in subject times ({})'.format(t, subject.times))

        forward_simulate(
            growth=growth, self_interactions=self_interactions, interactions=interactions,
            perturbations=perturbations, dt=args.simulation_dt, subject=subject, 
            start=t, n_days=args.n_days, output_basepath=args.output_basepath, 
            limit_of_detection=1e5)
    



