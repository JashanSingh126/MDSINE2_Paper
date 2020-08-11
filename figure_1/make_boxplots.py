'''Make boxplots to compare performances of different models
'''
import numpy as np
import logging
import sys
import pandas as pd
import h5py
import inspect
import random
import copy
import os
import shutil
import math
import argparse
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as plticker

import pylab as pl

sys.path.append('..')
import synthetic
import config
import main_base

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-paths', '-p', type=str, 
        help='Basepath for all the runs to get plots from',
        dest='data_paths', default=[], nargs='+')
    parser.add_argument('--names', '-n', type=str, 
        help='Basepath for all the runs of the L2 interence',
        dest='names', default=[], nargs='+')
    parser.add_argument('--dest-path', '-d', type=str,
        help='Destination path to save all of the plots',
        dest='dest_path', default='./')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    # Get the data
    config.LoggingConfig()
    args = parse_args()

    if len(args.data_paths) != len(args.names):
        raise ValueError('`data paths` ({}) and `names` ({}) should be the same length'.format(
            len(args.data_paths), len(args.names)))


    df = None
    for idx in range(len(args.data_paths)):
        print('Appending {}'.format(args.names[idx]))
        df = main_base.make_df(basepath=args.data_paths[idx], df=df, name=args.names[idx])

    if df is None:
        print('No paths given. Not plotting')
        sys.exit()

    dest_path = args.dest_path
    os.makedirs(dest_path, exist_ok=True)

    if dest_path[-1] != '/':
        dest_path += '/'

    with open(dest_path + 'df.pkl', 'wb') as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # with open(dest_path + 'df.pkl', 'rb') as handle:
    #     df = pickle.load(handle)

    # Measurement Noise
    # =================
    fig = plt.figure(figsize=(10,10))
    try:
        main_base.make_boxplots(df=df, x='Measurement Noise', y='Error-traj', hue='Model',
            only={'Replicates': 5, 'N Timepoints': 65}, yscale='linear', 
            title='Forward Simulation Error', ax=fig.add_subplot(2,2,1))
    except:
        pass

    try:
        main_base.make_boxplots(df=df, x='Measurement Noise', y='Error-interactions', hue='Model',
            only={'Replicates': 5, 'N Timepoints': 65}, yscale='log', 
            title='Predicted Interactions Error', ax=fig.add_subplot(2,2,2))
    except:
        pass
    
    try:
        main_base.make_boxplots(df=df, x='Measurement Noise', y='Error-clustering', hue='Model',
            only={'Replicates': 5, 'N Timepoints': 65}, yscale='linear', 
            title='Mean clustering error', ax=fig.add_subplot(2,2,3))
    except:
        pass
    
    try:
        main_base.make_boxplots(df=df, x='Measurement Noise', y='Error-growth', hue='Model',
            only={'Replicates': 5, 'N Timepoints': 65}, yscale='linear', 
            title='Predicted Growth Rate Errors', ax=fig.add_subplot(2,2,4))
    except:
        pass
    fig.suptitle('Measurement Noises')
    plt.savefig(dest_path + 'measurement_noise.pdf')
    plt.close()

    # Times
    # =====
    fig = plt.figure(figsize=(10,10))
    try:
        main_base.make_boxplots(df=df, x='N Timepoints', y='Error-traj', hue='Model',
            only={'Replicates': 4, 'Measurement Noise': 0.3}, yscale='linear', 
            title='Forward Simulation Error', ax=fig.add_subplot(2,2,1))
    except:
        pass
    
    try:
        main_base.make_boxplots(df=df, x='N Timepoints', y='Error-interactions', hue='Model',
            only={'Replicates': 4, 'Measurement Noise': 0.3}, yscale='log', 
            title='Predicted Interactions Error', ax=fig.add_subplot(2,2,2))
    except:
        pass

    try:    
        main_base.make_boxplots(df=df, x='N Timepoints', y='Error-clustering', hue='Model',
            only={'Replicates': 4, 'Measurement Noise': 0.3}, yscale='linear', 
            title='Mean clustering error', ax=fig.add_subplot(2,2,3))
    except:
        pass

    try:
        main_base.make_boxplots(df=df, x='N Timepoints', y='Error-growth', hue='Model',
            only={'Replicates': 4, 'Measurement Noise': 0.3}, yscale='linear', 
            title='Predicted Growth Rate Errors', ax=fig.add_subplot(2,2,4))
    except:
        pass
    fig.suptitle('Number of Time points')
    plt.savefig(dest_path + 'timepoints.pdf')
    plt.close()

    # Replicates
    # ==========
    fig = plt.figure(figsize=(10,10))
    try:
        main_base.make_boxplots(df=df, x='Replicates', y='Error-traj', hue='Model',
            only={'N Timepoints': 65, 'Measurement Noise': 0.2}, yscale='linear', 
            title='Forward Simulation Error', ax=fig.add_subplot(2,2,1))
    except:
        pass

    try: 
        main_base.make_boxplots(df=df, x='Replicates', y='Error-interactions', hue='Model',
            only={'N Timepoints': 65, 'Measurement Noise': 0.2}, yscale='log', 
            title='Predicted Interactions Error', ax=fig.add_subplot(2,2,2))
    except:
        pass
        
    try:
        main_base.make_boxplots(df=df, x='Replicates', y='Error-clustering', hue='Model',
            only={'N Timepoints': 65, 'Measurement Noise': 0.2}, yscale='linear', 
            title='Mean clustering error', ax=fig.add_subplot(2,2,3))
    except:
        pass

    try:
        main_base.make_boxplots(df=df, x='Replicates', y='Error-growth', hue='Model',
            only={'N Timepoints': 65, 'Measurement Noise': 0.2}, yscale='linear', 
            title='Predicted Growth Rate Errors', ax=fig.add_subplot(2,2,4))
    except:
        pass
    fig.suptitle('Number of Replicates')
    plt.savefig(dest_path + 'replicates.pdf')
    plt.close()





    