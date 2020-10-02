'''Aggregate the different runs and make a dataframe for each one of them. Runs
on any dispatch run. The actual plotting of the semi synthetic runs is run in
`MDSINE2.main_figures.semi_synthetic_benchmark_figure`
'''

import os
import sys
import pickle
import numpy as np
import pandas as pd
import argparse

import pylab as pl

sys.path.append('..')
import metrics
import config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--basepaths', '-p', type=str, 
        help='Basepaths for all the runs to get plots from',
        dest='basepaths', default=[], nargs='+')
    parser.add_argument('--names', '-n', type=str, 
        help='Names of model relative to each set of runs',
        dest='names', default=[], nargs='+')
    parser.add_argument('--filename', '-f', type=str,
        help='Destination to save the dataframe as a tsv file',
        dest='filename', default='./df.tsv')
    args = parser.parse_args()
    return args

def get_data(data, basepath, modelname):
    
    folders = [d for d in os.listdir(basepath) if os.path.isdir(os.path.join(basepath, d))]

    if basepath[-1] != '/':
        basepath += '/'

    for fidx, fold in enumerate(folders):
        print('\t{}/{}'.format(fidx, len(folders)))
        path = basepath + fold + '/validation/results.pkl'
        paramspath = basepath + fold + '/' + config.SYNPARAMS_FILENAME
        if os.path.isfile(path):
            
            metric = metrics.Metrics.load(path)
            truth_metrics = metric.truth_metrics
            params = config.SimulationConfig.load(paramspath)

            for subjname in metric.results:
                for simtype in metric.results[subjname]:

                    data['Model'].append(modelname)
                    data['Subject Name'].append(subjname)
                    data['Simulation Type'].append(simtype)
                    data['Measurement Noise'].append(params.MEASUREMENT_NOISE_LEVEL)
                    data['Process Variance'].append(params.PROCESS_VARIANCE_LEVEL)
                    data['Number of Timepoints'].append(params.TIMES)
                    data['Uniform Samples'].append(params.UNIFORM_SAMPLING_TIMEPOINTS)
                    data['Number of Replicates'].append(params.N_REPLICATES)
                    if pl.isMCMC(metric.model):
                        data['Error Trajectories'].append(np.nanmean(metric.results[subjname][simtype]['error-total']))
                        data['Error Interactions'].append(np.nanmean(truth_metrics['error-interactions']))
                        data['Error Perturbations'].append(np.nanmean(truth_metrics['error-perturbations']))
                        data['Error Topology'].append(np.nanmean(truth_metrics['error-topology']))
                        data['Error Growth'].append(np.nanmean(truth_metrics['error-growth']))
                        data['Error Clustering'].append(np.nanmean(truth_metrics['error-clustering']))
                        
                    else:
                        data['Error Trajectories'].append(metric.results[subjname][simtype]['error-total'])
                        data['Error Interactions'].append(truth_metrics['error-interactions'])
                        data['Error Perturbations'].append(truth_metrics['error-perturbations'])
                        data['Error Topology'].append(truth_metrics['error-topology'])
                        data['Error Growth'].append(truth_metrics['error-growth'])
                        data['Error Clustering'].append(truth_metrics['error-clustering'])

    return data

if __name__ == '__main__':

    args = parse_args()
    if len(args.basepaths) != len(args.names):
        raise ValueError('`data paths` ({}) and `names` ({}) should be the same length'.format(
            len(args.basepaths), len(args.names)))
    
    data = {
        'Model': [],
        'Subject Name': [],
        'Simulation Type': [],
        'Error Trajectories': [], 
        'Error Interactions': [], 
        'Error Perturbations': [], 
        'Error Topology': [],
        'Error Growth': [],
        'Error Clustering': [],
        'Measurement Noise': [],
        'Process Variance': [],
        'Number of Timepoints': [],
        'Uniform Samples': [],
        'Number of Replicates': []}

    for idx in range(len(args.basepaths)):
        print('Models {}/{}'.format(idx, len(args.basepaths)))
        basepath = args.basepaths[idx]
        modelname = args.names[idx]
        data = get_data(data=data, basepath=basepath, modelname=modelname)

    # Make the dataframe and save
    df = pd.DataFrame(data=data)
    df.to_csv(args.filename, sep='\t', header=True, index=False)
