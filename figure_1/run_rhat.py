'''Run rhat for all (most) the variables
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

import pylab as pl

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

import config
sys.path.append('..')
import names

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--basepath', '-db', type=str, 
        help='Basepath for all the runs',
        dest='basepath')
    parser.add_argument('--output-path', '-ob', type=str, 
        help='Basepath for all the runs', dest='output_basepath')
    parser.add_argument('--data-seed', '-d', type=int,
        help='Dataseed they were run at', dest='data_seed')
    parser.add_argument('--init-seeds', '-is', type=int,
        help='Initialization seeds they were run at', 
        dest='init_seeds', nargs='+')
    parser.add_argument('--n-samples', '-ns', type=int,
        help='Total number of Gibbs steps to do',
        dest='n_samples')
    parser.add_argument('--burnin', '-nb', type=int,
        help='Total number of burnin steps',
        dest='burnin')
    parser.add_argument('--n-asvs', '-o', type=int,
        help='Number of ASVs', dest='n_asvs')
    parser.add_argument('--n-replicates', '-nr', type=int,
        help='Number of replicates to use for the data of inference',
        dest='n_replicates')
    parser.add_argument('--measurement-noise', '-m', type=float,
        help='Measurement noise it was run at',
        dest='measurement_noise')
    parser.add_argument('--process-variance', '-p', type=float,
        help='Process variance it was run at',
        dest='process_variance')
    parser.add_argument('--n-times', '-nt', type=int,
        help='Number of time points', dest='n_times')
    parser.add_argument('--clustering-ons', '-co', type=int,
        help='To run with clustering on and off', 
        dest='clustering_on')
    parser.add_argument('--healthy', '-hy', type=bool,
        help='Whether or not to use the healthy patients or not',
        default=False, dest='healthy')
    parser.add_argument('--percent-change-clustering', '-pcc', type=float,
        help='Percent of ASVs to update during clustering every time it runs',
        default=1.0, dest='percent_change_clustering')
    return parser.parse_args()


def compute_rhat():
    pass

if __name__ == '__main__':

    args = parse_args()
    chains = []
    STRNAMES = names.STRNAMES

    # Get the chains
    config.LoggingConfig()
    synparams = config.SimulationConfig(times=args.n_times, n_replicates=args.n_replicates,
        n_asvs=args.n_asvs, healthy=args.healthy, 
        process_variance_level=args.process_variance,
        measurement_noise_level=args.measurement_noise)
    for init_seed in args.init_seeds:
        params = config.ModelConfigICML(output_basepath=args.basepath, data_seed=args.data_seed,
            data_path=None, init_seed=init_seed, a0=synparams.NEGBIN_A0, a1=synparams.NEGBIN_A1,
            n_samples=args.n_samples, burnin=args.burnin, pcc=args.percent_change_clustering,
            clustering_on=bool(args.clustering_on))
        graph_name = 'graph'+ params.suffix() + synparams.suffix()
        basepath = params.OUTPUT_BASEPATH + graph_name + '/'

        chains.append(pl.inference.BaseMCMC.load(basepath + 'mcmc.pkl'))

    basepath = args.output_basepath
    os.makedirs(basepath, exist_ok=True)

    n_asvs = args.n_asvs

    f = open(basepath + 'rhat.txt', 'w')

    # Run growths
    f.write('\n\nPrior mean growths\n')
    f.write('--------------------\n')
    ret = pl.inference.r_hat(chains=chains, vname=STRNAMES.PRIOR_MEAN_GROWTH)
    f.write('{}'.format(ret))

    f.write('Growths\n')
    f.write('-------\n')
    ret = pl.inference.r_hat(chains=chains, vname=STRNAMES.GROWTH_VALUE)
    for oidx in range(n_asvs):
        f.write('\t{}: {}\n'.format(oidx, ret[oidx]))

    # Run self-interactions
    f.write('\n\nPrior mean self-interactions\n')
    f.write('------------------------------\n')
    ret = pl.inference.r_hat(chains=chains, vname=STRNAMES.PRIOR_MEAN_SELF_INTERACTIONS)
    f.write('{}'.format(ret))

    f.write('\n\nSelf-interactions\n')
    f.write('-----------------\n')
    ret = pl.inference.r_hat(chains=chains, vname=STRNAMES.SELF_INTERACTION_VALUE)
    for oidx in range(n_asvs):
        f.write('\t{}: {}\n'.format(oidx, ret[oidx]))

    # Run interactions
    f.write('\n\nPrior mean interactions\n')
    f.write('-------------------------\n')
    ret = pl.inference.r_hat(chains=chains, vname=STRNAMES.PRIOR_MEAN_INTERACTIONS)
    f.write('{}'.format(ret))

    f.write('\n\nInteractions\n')
    f.write('------------\n')
    ret = pl.inference.r_hat(chains=chains, vname=STRNAMES.INTERACTIONS_OBJ)
    f.write('{}'.format(ret))

    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111)
    ax = sns.heatmap(ret, annot=True, fmt='.2f', ax=ax)
    fig.suptitle(r'$\hat{R}$' + ' interactions', size=20)
    plt.savefig(basepath + 'interactions.pdf')
    plt.close()

    # # Run peturbations
    # f.write('\n\nPrior mean perturbations\n')
    # f.write('--------------------------\n')
    # ret = pl.inference.r_hat(chains=chains, vname=STRNAMES.PRIOR_MEAN_PERT)
    # f.write('{}'.format(ret))

    if STRNAMES.PERT_VALUE in chains[0].graph:
        for pidx in range(len(chains[0].graph.perturbations)):
            pname = 'pert{}'.format(pidx)
            f.write('\n\nPerturbation {}\n'.format(pidx))
            f.write('--------------\n')
            ret = pl.inference.r_hat(chains=chains, vname=pname)
            for oidx in range(n_asvs):
                f.write('\t{}: {}\n'.format(oidx, ret[oidx]))

    f.write('\n\nProcess Variance\n')
    f.write('------------------\n')
    ret = pl.inference.r_hat(chains=chains, vname=STRNAMES.PROCESSVAR)
    f.write('{}'.format(ret))

    f.write('\n\nConcentration\n')
    f.write('------------------\n')
    ret = pl.inference.r_hat(chains=chains, vname=STRNAMES.CONCENTRATION)
    f.write('{}'.format(ret))

    # Run filtering
    f.write('\n\nLatent state\n')
    f.write('-------------\n')
    for ridx in range(chains[0].graph.data.n_replicates):

        times = chains[0].graph.data.times[ridx]

        dname = 'x_ridx{}'.format(ridx)
        ret = pl.inference.r_hat(chains=chains, vname=dname)

        fig = plt.figure(figsize=(20,10))
        ax = fig.add_subplot(111)
        # pl.visualization.shade_in_perturbations(ax=ax, perturbations=chains[0].graph.perturbations)
        ax = sns.heatmap(ret, ax=ax, xticklabels=times)
        fig.suptitle(r'$\hat{R}$' + ' Latent {}'.format(ridx), size=40)

        ax.set_ylabel('ASV', size=20)
        ax.set_xlabel('Day', size=20)

        # loc = plticker.MultipleLocator(base=10)
        # ax.xaxis.set_major_locator(loc)

        plt.savefig(basepath + 'latent{}.pdf'.format(ridx))
        plt.close()



    

    f.close()

        

    
    

