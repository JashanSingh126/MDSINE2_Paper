'''Aggregate output from `MDSINE2/time_look_ahead/forward_simulate.py` and calculate the error.

Saves a TSV file indicating the error

Metrics
-------
These metrics are calculated per Gibb step per ASV trajectory. Options:

'RMSE' : Root mean square error
'logRMSE' : Root mean square error of the log abundance (we ignore points that have nans)
'relRMSE' : Root mean square error of the relative abundance
'mean-spearman' : Mean spearman correlation over each of teh trajectories
'percent-error' : Percent error

'''
import logging
import time
import sys
import os
import os.path
import numpy as np
import argparse
import time
import scipy.stats

logging.basicConfig(level=logging.INFO)

columns = ['Dataset', 'Model', 'Day', 'Lookahead', 'Error', 'Metric']

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--basepath', type=float, dest='basepath',
        help='Path to look for runs')
    parser.add_argument('--output', type=str, dest='output',
        help='Place to save the tsv output')
    parser.add_argument('--metric', type=float, dest='metric',
        help='Path to look for runs')
    parser.add_argument('--dataset', type=str, dest='dataset',
        help='Dataset')
    parser.add_argument('--model', type=str, dest='model',
        help='Model')

    args = parser.parse_args()
    return args

def _RMSE(pred, truth):
    '''Root Mean Square Error

    Parameters
    ----------
    pred : np.ndarray (n_asvs, n_times)
        Predicted tracjectory for each gibb sample
    truth : np.ndarray (n_saves, n_times)
        Truth trajectory

    Returns
    -------
    float
    '''
    return np.sqrt(np.mean(np.square(pred - truth)))

def _logRMSE(pred, truth):
    '''Root Mean Square Error of log trajectories.

    Parameters
    ----------
    pred : np.ndarray (n_asvs, n_times)
        Predicted tracjectory for each gibb sample
    truth : np.ndarray (n_saves, n_times)
        Truth trajectory

    Returns
    -------
    float
    '''
    return np.sqrt(np.nanmean(np.square(np.log(pred) - np.log(truth))))

def _relRMSE(pred, truth):
    '''Root Mean Square Error of the relative abundances

    Parameters
    ----------
    pred : np.ndarray (n_asvs, n_times)
        Predicted tracjectory for each gibb sample
    truth : np.ndarray (n_saves, n_times)
        Truth trajectory

    Returns
    -------
    float
    '''
    return np.sqrt(np.mean(np.square(pred.sum(axis=0) - truth.sum(axis=0))))

def _mean_spearman(pred, truth):
    '''Mean spearman correlation over the trajectories.

    Parameters
    ----------
    pred : np.ndarray (n_asvs, n_times)
        Predicted tracjectory for each gibb sample
    truth : np.ndarray (n_saves, n_times)
        Truth trajectory

    Returns
    -------
    float
    '''
    a = np.zeros(pred.shape[0])
    for i in range(len(a)):
        a[i] = scipy.stats.spearmanr(pred[i], truth[i])[0]
    return np.nanmean(a)

def calculate_error(pred, truth, metric, stat='mean'):
    '''Return the `stat` `metric` error between `pred` and `truth` over each
    Gibb step

    Parameters
    ----------
    pred : np.ndarray (n_samples, n_asvs, n_times)
        Predicted trajectories for each Gibb sample
    truth : np.ndarray (n_asvs, n_times)
        Ground truth
    metric : callable
        Metric
    stat : str
        Options:
            `mean`
            `median`
    '''
    ret = np.zeros(pred.shape[0])
    for i in range(len(ret)):
        ret[i] = metric(pred=pred[i], truth=truth)
    if stat == 'mean':
        ret = np.mean(ret)
    elif stat == 'median':
        ret = np.median(ret)
    else:
        raise ValueError('`stat` ({}) not recognized'.format(stat))
    return ret



if __name__ == '__main__':
    args = parse_args()

    basepath = args.basepath
    if basepath[-1] != '/':
        basepath += '/'
    
    onlyfiles = [f for f in os.listdir(basepath) if os.path.isfile(basepath+f)]
    print(onlyfiles)