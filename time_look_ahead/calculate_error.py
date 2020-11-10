'''Aggregate output from `MDSINE2/time_look_ahead/forward_simulate.py` and calculate the error.

Saves a TSV file indicating the error

Metrics
-------
These metrics are calculated per Gibb step per ASV trajectory. Options:

'RMSE' : Root mean square error
'logRMSE' : Root mean square error of the log abundance (we ignore points that have nans)
'relRMSE' : Root mean square error of the relative abundance
'mean-spearman' : Mean spearman correlation over each of teh trajectories

'''
import logging
import time
import sys
import os
import os.path
import numpy as np
import argparse
import time
import re
import scipy.stats
import pandas as pd

logging.basicConfig(level=logging.INFO)

columns = ['Dataset', 'Model', 'Day', 'Lookahead', 'Error', 'Metric']

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--basepath', type=str, dest='basepath',
        help='Path to look for runs')
    parser.add_argument('--output', type=str, dest='output',
        help='Place to save the tsv output')
    parser.add_argument('--metric', type=str, dest='metric',
        help='Path to look for runs')
    parser.add_argument('--dataset', type=str, dest='dataset',
        help='Dataset')
    parser.add_argument('--model', type=str, dest='model',
        help='Model')
    parser.add_argument('--stat', type=str, dest='stat',
        help='How to aggregate the errors. Options: mean, median')

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
    pred : np.ndarray (n_asvs)
        Predicted tracjectory for each gibb sample
    truth : np.ndarray (n_saves)
        Truth trajectory

    Returns
    -------
    float
    '''
    return np.sqrt(np.mean(np.square(pred/pred.sum() - truth/truth.sum())))

def _mean_spearman(pred, truth):
    '''Mean Root mean square error

    Parameters
    ----------
    pred : np.ndarray (n_asvs)
        Predicted tracjectory for each gibb sample
    truth : np.ndarray (n_saves)
        Truth trajectory

    Returns
    -------
    float
    '''
    err = []
    for aidx in range(pred.shape[0]):
        err.append(scipy.stats.spearmanr(pred[aidx], truth[aidx])[0])
    return np.nanmean(err)

def calculate_error(pred, truth, metric, stat='mean'):
    '''Return the `stat` `metric` error between `pred` and `truth` over each
    Gibb step

    Parameters
    ----------
    pred : np.ndarray (n_samples, n_asvs)
        Predicted trajectories for each Gibb sample
    truth : np.ndarray (n_asvs)
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

    if args.metric == 'RMSE':
        metric = _RMSE
    elif args.metric == 'logRMSE':
        metric = _logRMSE
    elif args.metric == 'relRMSE':
        metric = _relRMSE
    elif args.metric == 'mean-spearman':
        metric = _mean_spearman
    # elif args.metric == 'mean-spearman':
    #     metric = _mean_spearman
    else:
        raise ValueError('`metric` ({}) not recognized'.format(args.metric))

    find_tla_start = re.compile(r'.+-tla([\d.]+)-start([\d.]+).+')
    prefixes = set([])
    data = []
    for idx_fname, fname in enumerate(onlyfiles):

        if '-pred.npy' in fname:
            prefix = fname.replace('-pred.npy', '')
        elif '-truth.npy':
            prefix = fname.replace('-truth.npy', '')
        else:
            raise ValueError('fname ({}) does not follow standard'.format(fname))

        if prefix in prefixes:
            # Already calculated on, skip
            continue
        prefixes.add(prefix)

        pred = np.load(basepath + prefix + '-pred.npy')
        truth = np.load(basepath + prefix + '-truth.npy')

        error = calculate_error(pred=pred, truth=truth, metric=metric, stat=args.stat)

        tla, start = find_tla_start.findall(fname)[0]
        tla = float(tla)
        start = float(start)
        day = tla+start
        logging.info('{}/{}: tla{}-start{} {:.3E}'.format(idx_fname, len(onlyfiles),tla, start, error))

        data.append([args.dataset.replace('_', ' '), args.model, day, tla, error, args.metric])

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(args.output, index=False, header=True, sep='\t')

    
        
