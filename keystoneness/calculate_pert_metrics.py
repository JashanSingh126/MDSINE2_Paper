'''Calculate the metrics for the perturbation keystoneness:

Maximum deviation
-----------------
Maximum deviation is the maximum distance a trajectory deviates from the baseline trajectory 
in any time during the perturbation period. Note that every asv can have a different time.

When we report them, we report the mean maximum deviation over the gibb steps and over the 
ASV trajectories

Return Time to baseline
-----------------------
This is the amount of time it takes a trajectory to return within X% of the baseline

'''

import numpy as np
import pandas as pd
import logging
import sys
import os
import time
import argparse
import re

logging.basicConfig(level=logging.INFO)

def mean_absolute_maximum_deviation(trajectories, baseline, start_idx, end_idx):
    '''Return the mean maximum deviation from the baseline.
    We calculate this over each ASV trajectory in each gibb step and then we
    return the mean over all of those at the end. 

    Parameters
    ----------
    trajectories, baseline : np.ndarray(n_asvs, n_times)
    '''
    trajectories = trajectories[:, start_idx:end_idx]
    baseline = baseline[:, start_idx:end_idx]
    diff = np.absolute(trajectories-baseline)

    max_diff = np.nanmax(diff, axis=1)
    return np.nanmean(max_diff)

def time_return_to_baseline(baseline, end_pert_idx, trajectories, dt, thresh):
    '''Return the mean amount of time it takes for the trajectory to return to the baseline

    Parameters
    ----------
    baseline : np.ndarray(n_asvs)
        Steady state for each asv in each of the gibb steps
    end_pert_idx : int
        This is the index that is the end of the perturbation. We index 
        `end_pert_idx:`. This is the last axis in trajectories
    trajectories : np.ndarray(n_asvs, n_times)
        These are the trajectories for each on of the asvs
    dt : float
        This is the time step for each of the timepoints
    thresh : float
        threshold it should be within
        Example: 0.01 ~ must be within 99% of the baseline
    '''
    trajectories = trajectories[:,end_pert_idx:]
    baseline = baseline.reshape(baseline.shape[0], 1)

    diff = np.absolute(trajectories - baseline) / baseline
    low_thresh = 1 - thresh
    high_thresh = 1 + thresh

    ret = np.zeros(len(baseline))*np.nan
    # For each asv
    for i in range(diff.shape[0]):
        # for each timepoint
        if np.isnan(diff[i,0]):
            continue
        for j in range(diff.shape[1]):
            if diff[i,j] >= low_thresh and diff[i,j] <= high_thresh:
                break
        ret[i] = j*dt
    return np.nanmean(ret)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--basepath', type=str, dest='basepath',
        help='This is the path that holds the numpy arrays returned by ' \
        '`keystoneness.keystoneness_perturbation_single`.')
    parser.add_argument('--pert-start', type=float, dest='pert_start',
        help='This is the timepoint the perturbation started at')
    parser.add_argument('--pert-end', type=float, dest='pert_end',
        help='This is the timepoint the perturbation ended at')
    parser.add_argument('--dt', type=float, dest='dt',
        help='This is the time step in the vectors')
    parser.add_argument('--subjset', type=str, dest='subjset',
        help='Filename for the SubjectSet> This gets us our Perturbation Names')
    parser.add_argument('--leave-out-table', type=str, dest='leave_out_table',
        help='File containing which ASVs to remove at each iteration', default=None)

    args = parser.parse_args()

    columns = ['Perturbation', 'Mean Maximum Deviation', 'Return to Baseline (5%)', 
        'Return to Baseline (3%)', 'Return to Baseline (1%)']
    index = []

    basepath = args.basepath
    if basepath[-1] != '/':
        basepath += '/'
    subjset = pl.SubjectSet.load(args.subjset)
    pert_end = args.pert_end

    idx_start = int(args.pert_start/args.dt)
    idx_end = int(args.pert_end/args.dt)
    pre_pert_idx = idx_start-1

    baselines = []
    for pidx in range(len(subjset.perturbations)):
        temp_baseline_path = basepath+'base_{}'.format(pidx)
        arr = []
        nfiles = len(os.listdir(temp_baseline_path))
        for fn in range(nfiles):
            temp = np.load(temp_baseline_path + 'arr{}.npy'.format(fn))
            arr = np.append(arr, temp)
        baselines.append(arr)


    get_loidx = re.compile(r'^leave_out(\d+)_pert\d+')
    get_pidx = re.compile(r'^leave_out\d+_pert(\d+)')

    f = open(args.leave_out_tale, 'r')
    leave_out_table = f.read()
    f.close()
    leave_out_table = leave_out_table.split('\n')

    data = []
    onlydirs = [f for f in os.listdir(basepath) if os.path.isdir(basepath+f)]
    for dirname in onlydirs:
        if 'base' in dirname:
            continue

        loidx = get_loidx.findall(dirname)[0]
        pidx = get_pidx.findall(dirname)[0]

        left_out = tuple(leave_out_table[loidx].split(','))
        pert_name = subjset.pertubrations[pidx].name

        baseline = baselines[pidx]

        trajs = []
        temp_traj_basepath = basepath + dirname + '/'
        fn = len([f for f in os.listdir(temp_traj_basepath) if os.path.isfil(temp_traj_basepath+f)])
        for i in range(fn):
            temp = np.load(temp_traj_basepath + 'arr{}.npy'.format(i))
            trajs = np.append(trajs, temp)

        max_deviation = mean_absolute_maximum_deviation(
            trajectories=trajs, baseline=baseline, start_idx=idx_start, 
            end_idx=idx_end)

        retbase5 = time_return_to_baseline(
            baseline=baseline[:, pre_pert_idx], 
            end_pert_idx=idx_end, 
            trajectories=trajs, 
            dt=args.dt, thresh=0.05)

        retbase3 = time_return_to_baseline(
            baseline=baseline[:, pre_pert_idx], 
            end_pert_idx=idx_end, 
            trajectories=trajs, 
            dt=args.dt, thresh=0.03)

        retbase1 = time_return_to_baseline(
            baseline=baseline[:, pre_pert_idx], 
            end_pert_idx=idx_end, 
            trajectories=trajs, 
            dt=args.dt, thresh=0.01)

        data.append([pert_name, max_deviation, retbase5, retbase3, retbase1])
        index.append(left_out)

    df = pd.DataFrame(data, columns=columns, index=index)
    fname

    

        

        
        







