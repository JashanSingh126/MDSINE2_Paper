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

import pylab as pl

import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

def max_L2_deviation(trajectories, baseline, start_idx, end_idx):
    '''
    Mean_gibb(max_t(L2(stability, deviation)))

    Parameters
    ----------
    trajectories, baseline : np.ndarray(n_asvs, n_times)
    '''
    # print(trajectories.shape)
    # print(baseline.shape)

    baseline = baseline[:trajectories.shape[0], ...]

    trajectories = trajectories[:,:,start_idx:end_idx]
    baseline = baseline[:,:,start_idx:end_idx]

    # [gibb, time]
    maxval = 1e13
    keep = []
    for gibbstep in range(trajectories.shape[0]):
        nanmaxtraj = np.nanmax(trajectories[gibbstep])
        nanmaxbase = np.nanmax(baseline[gibbstep])
        if nanmaxtraj >= maxval or nanmaxbase >= maxval:
            continue
        keep.append(gibbstep)
    # print(keep)
    trajectories = trajectories[keep]
    baseline = baseline[keep]
    
    # s = 0
    # for i in range(inner.shape[1]):
    #     if np.nansum(inner[])
    L2_over_time = np.sqrt(np.nansum(np.square(trajectories - baseline), axis=1))
    maxdiff = np.nanmean(np.max(L2_over_time, axis=1))
    # mean_max_diff = np.nanmean(diff)

    return maxdiff

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
    # print('time return to baseline')
    baseline = baseline[:trajectories.shape[0], ...]
    trajectories = trajectories[:,:,end_pert_idx:]

    maxval = 1e13
    keep = []
    for gibbstep in range(trajectories.shape[0]):
        nanmaxtraj = np.nanmax(trajectories[gibbstep])
        nanmaxbase = np.nanmax(baseline[gibbstep])
        if nanmaxtraj >= maxval or nanmaxbase >= maxval:
            continue
        keep.append(gibbstep)
    # print(keep)
    trajectories = trajectories[keep]
    baseline = baseline[keep]
    baseline = baseline.reshape(trajectories.shape[0], trajectories.shape[1], 1)

    diff = np.absolute(trajectories-baseline)/baseline

    # print(diff)
    # print(diff.shape)

    # baseline = baseline.reshape(-1,1)

    # print(trajectories.shape)
    # print(baseline.shape)

    # diff = np.absolute(trajectories - baseline) / baseline
    # print('diff')
    # print(diff.shape)
    # print(diff)
    low_thresh = 1 - thresh
    high_thresh = 1 + thresh

    ret = np.zeros(diff.shape[0])*np.nan
    # for each Gibb step
    for gibbstep in range(diff.shape[0]):
        temp = np.zeros(diff.shape[1])*np.nan
        # For each asv
        for aidx in range(diff.shape[1]):
            # Get the max time

            if np.any(np.isnan(diff[gibbstep,aidx,0])):
                continue
            for j in range(diff.shape[2]):
                if diff[gibbstep, aidx,j] >= low_thresh and diff[gibbstep, aidx,j] <= high_thresh:
                    break
            temp[aidx] = j*dt
        ret[gibbstep] = np.nanmax(temp)

    a = np.nanmean(ret)
    print(a)
    return a
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--basepath', type=str, dest='basepath',
        help='This is the path that holds the numpy arrays returned by ' \
        '`keystoneness.keystoneness_perturbation_single`.')
    parser.add_argument('--outfile', type=str, dest='outfile',
        help='WHere to save the output')
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

    columns = ['Perturbation', 'Maximum L2 Deviation (CFUs/g)', 'Return to Baseline (5%) (days)', 
        'Return to Baseline (3%) (days)', 'Return to Baseline (1%) (days)']
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
    print('loading baseline')
    for pidx in range(len(subjset.perturbations)):
        temp_baseline_path = basepath+'base_{}'.format(pidx)
        arrs = []
        nfiles = len(os.listdir(temp_baseline_path))
        for fn in range(nfiles):
            temp = np.load(temp_baseline_path + '/arr{}.npy'.format(fn))
            arrs.append(temp)
        arr = np.asarray(arrs)
        baselines.append(arr)


    get_loidx = re.compile(r'^leave_out(\d+)_pert\d+')
    get_pidx = re.compile(r'^leave_out\d+_pert(\d+)')

    f = open(args.leave_out_table, 'r')
    leave_out_table = f.read()
    f.close()
    leave_out_table = leave_out_table.split('\n')

    data = []
    onlydirs = [f for f in os.listdir(basepath) if os.path.isdir(basepath+f)]
    for idir, dirname in enumerate(onlydirs):
        if 'base' in dirname:
            continue

        print('on', dirname)

        loidx = int(get_loidx.findall(dirname)[0])
        pidx = int(get_pidx.findall(dirname)[0])

        left_out = tuple(leave_out_table[loidx].split(','))
        pert_name = subjset.perturbations[pidx].name

        baseline = baselines[pidx]

        trajs = []
        temp_traj_basepath = basepath + dirname + '/'
        fn = len([f for f in os.listdir(temp_traj_basepath) if os.path.isfile(temp_traj_basepath+f)])
        for i in range(fn):
            temp = np.load(temp_traj_basepath + '/arr{}.npy'.format(i))
            trajs.append(temp)
        trajs = np.asarray(trajs)
        print('traj shape', trajs.shape)

        max_deviation = max_L2_deviation(
            trajectories=trajs, baseline=baseline, start_idx=idx_start, 
            end_idx=idx_end)

        print(max_deviation)

        retbase5 = time_return_to_baseline(
            baseline=baseline[:, :, pre_pert_idx], 
            end_pert_idx=idx_end, 
            trajectories=trajs, 
            dt=args.dt, thresh=0.05)

        retbase3 = time_return_to_baseline(
            baseline=baseline[:, :, pre_pert_idx], 
            end_pert_idx=idx_end, 
            trajectories=trajs, 
            dt=args.dt, thresh=0.03)

        retbase1 = time_return_to_baseline(
            baseline=baseline[:, :, pre_pert_idx], 
            end_pert_idx=idx_end, 
            trajectories=trajs, 
            dt=args.dt, thresh=0.01)

        data.append([pert_name, max_deviation, retbase5, retbase3, retbase1])
        index.append(left_out)

        # if idir == 30:
        #     break

    df = pd.DataFrame(data, columns=columns, index=index)
    print(df)
    df.to_csv(args.outfile, sep='\t')
    # fname

    

        

        
        







