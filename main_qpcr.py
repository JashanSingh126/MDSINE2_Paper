'''Learn the qPCR scaling parameter offline.
'''
import numpy as np
import logging
import sys
import time
import pandas as pd
import os
import argparse

import matplotlib.pyplot as plt
import seaborn as sns

import pylab as pl
import config

UNHEALTHY_SUBJECTS = ['2','3','4','5']
HEALTHY_SUBJECTS = ['6','7','8','9','10']

def parse_args():
    '''Parse the arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-filename', '-d', type=str,
        help='Location that the SubjectSet object is stored',
        dest='data_filename', default='./pickles/real_subjectset.pkl')
    parser.add_argument('--output-basepath', '-o', type=str,
        help='Location to save the output txt file',
        dest='output_basepath', default='./output_qpcr/')
    return parser.parse_args()

def plot_log_historgram(logdata, logdeviations, suptitle, nbins=25):
    '''Plot the data with a logscale plot

    Parameters
    ----------
    logdata : numpy.ndarray
        This is the data in log scale
    title : str
        This is the title of the figure
    n_bins : int
        This is the number of bins to make in the histogram
    '''
    colors = sns.color_palette()
    fig = plt.figure()

    # Data
    ax = fig.add_subplot(211)
    data = np.exp(np.asarray(logdata))
    _, bins = np.histogram(data, bins=nbins)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    ax.hist(x=data, bins=logbins, color=colors[0])
    ax.set_xlabel('CFUs/g')
    ax.set_xscale('log')
    ax.set_ylabel('Count')
    ax.set_title('Data')

    # Deviations
    ax = fig.add_subplot(212)
    data = np.exp(np.asarray(logdeviations))
    _, bins = np.histogram(data, bins=nbins)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    ax.hist(x=data, bins=logbins, color=colors[0])
    ax.set_xlabel('CFUs/g')
    ax.set_xscale('log')
    ax.set_ylabel('Count')
    ax.set_title('Deviation from triplicate mean')

    fig.suptitle(suptitle)
    fig.subplots_adjust(hspace=0.55)

    return ax

def learn_qpcr_parameter(subjset, basepath):
    '''Gather all of the deviations from the goemetric mean of the qPCR triplicates
    and then learn that variance.

    Plot and calculate the qPCR scale for each subject individually, for each consortia,
    and for all of the subjects together.

    Parameters
    ----------
    subjset : pylab.base.SubjectSet
        Where all of the qPCR measurements are.
    basepath : str
        Folder to save the file summarizing the learning.
    '''
    f = open(basepath + 'overview.txt', 'w')
    f.write('Learning qPCR scaling parameter\n')
    f.write('-------------------------------\n\n')

    # Get the data
    qpcr_log_data = {}
    for subj in subjset:
        qpcr_log_data[subj.name] = {}
        for t in subj.qpcr:
            qpcr_log_data[subj.name][t] = subj.qpcr[t].log_data

    # Learn for each subject and plot
    f.write('Subject specific data - logscale\n')
    f.write('--------------------------------\n')
    logdeviations_d = {}
    logdata_d = {}
    for subjname in qpcr_log_data:

        f.write('Subject {}\n'.format(subjname))
        logdata = []
        logdeviations = []
        for t,devs in qpcr_log_data[subjname].items():
            logdata = np.append(logdata, devs)
            logdeviations = np.append(logdeviations, devs - np.mean(devs))

        # Learn the scale parameter
        s = np.std(logdeviations)
        f.write('\tLearned lognormal scale parameter: {}\n'.format(s))

        _ = plot_log_historgram(logdata=logdata, logdeviations=logdeviations, 
            suptitle='Subject {}, lognormal scale = {:.3f}'.format(subjname, s),
            nbins=30)
        plt.savefig(basepath + 'subject_{}.pdf'.format(subjname))
        plt.close()

        logdeviations_d[subjname] = logdeviations
        logdata_d[subjname] = logdata

    # Learn for each consortium
    f.write('\n\nHealthy Consortium\n')
    f.write('------------------\n')
    logdevs = []
    logdata = []
    for subjname in HEALTHY_SUBJECTS:
        logdevs = np.append(logdevs, logdeviations_d[subjname])
        logdata = np.append(logdata, logdata_d[subjname])

    s = np.std(logdevs)
    f.write('Learned lognormal scale parameter: {}\n'.format(s))

    _ = plot_log_historgram(logdata=logdata, logdeviations=logdevs, 
        suptitle='Healthy Consortium, lognormal scale = {:.3f}'.format(s),
        nbins=40)
    plt.savefig(basepath + 'healthy_consortium.pdf')
    plt.close()

    f.write('\n\nUlcerative Colitis Consortium\n')
    f.write('-----------------------------\n')
    logdevs = []
    logdata = []
    for subjname in UNHEALTHY_SUBJECTS:
        logdevs = np.append(logdevs, logdeviations_d[subjname])
        logdata = np.append(logdata, logdata_d[subjname])

    s = np.std(logdevs)
    f.write('Learned lognormal scale parameter: {}\n'.format(s))

    _ = plot_log_historgram(logdata=logdata, logdeviations=logdevs, 
        suptitle='Ulcerative Colitis Consortium, lognormal scale = {:.3f}'.format(s),
        nbins=40)
    plt.savefig(basepath + 'uc_consortium.pdf')
    plt.close()

    # Learn for all the data
    f.write('\n\nAll data\n')
    f.write('--------\n')
    logdevs = []
    logdata = []
    for subjname in HEALTHY_SUBJECTS:
        logdevs = np.append(logdevs, logdeviations_d[subjname])
        logdata = np.append(logdata, logdata_d[subjname])
    for subjname in UNHEALTHY_SUBJECTS:
        logdevs = np.append(logdevs, logdeviations_d[subjname])
        logdata = np.append(logdata, logdata_d[subjname])

    s = np.std(logdevs)
    f.write('Learned lognormal scale parameter: {}\n'.format(s))

    _ = plot_log_historgram(logdata=logdata, logdeviations=logdevs, 
        suptitle='All data, lognormal scale = {:.3f}'.format(s),
        nbins=40)
    plt.savefig(basepath + 'all_data.pdf')
    plt.close()





    

        

        
        

    


if __name__ == '__main__':

    args = parse_args()
    os.makedirs(args.output_basepath, exist_ok=True)

    basepath = args.output_basepath
    if basepath[-1] != '/':
        basepath += '/'

    try:
        subjset = pl.base.SubjectSet.load(args.data_filename)
    except:
        raise FileNotFoundError('Location ({}) does not have a subjectset file.'.format(
            args.data_filename))

    learn_qpcr_parameter(subjset=subjset, basepath=basepath)

