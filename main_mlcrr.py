'''Implement the Maximum Likelihood Constrained Ridge Regression specified in
the original MDSINE [1] paper.

[1] B. Vanni, et al., "Mdsine: Microbial dynamical systems inference engine for 
    microbiome time-series analysis," Genome Biology, 17(1):121, 2016.
'''

import logging
import numpy as np
import sys
import os
import os.path
import pickle
import pandas
import argparse

import matplotlib.pyplot as plt
import seaborn

import pylab as pl
import config
import model
import synthetic
from names import STRNAMES, REPRNAMES
import preprocess_filtering as filtering
import data
import main_base
import mlcrr

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
        help='Dataset to do inference on',
        dest='dataset', default='mdsine2-real-data')
    parser.add_argument('--data-seed', '-d', type=int,
        help='Seed to initialize the data',
        dest='data_seed')
    parser.add_argument('--init_seed', '-i', type=int,
        help='Seed to initialize the inference',
        dest='init_seed')
    parser.add_argument('--basepath', '-b', type=str,
        help='Folder to save the output', default='output_mlcrr/',
        dest='basepath')
    parser.add_argument('--healthy', '-hy', type=bool,
        help='Whether or not to use the healthy patients or not',
        default=False, dest='healthy')
    parser.add_argument('--leave-out', '-l', type=int,
        help='Which subject/s to leave out of the inference and to use for' \
             ' testing the predictive accuracy.',
        default=-1, dest='leave_out')
    parser.add_argument('--cross-validate', '-cv', type=int,
        help='Whether or not to do a cross validation',
        default=0, dest='cross_validate')
    parser.add_argument('--use-bsub', '-cvbsub', type=int,
        help='Necessary if cross_validate is True. If True, use bsub to do the dispatching. ' \
            'Else just run the jobs sequentially.', default=0, dest='use_bsub')
    return parser.parse_args()

def make_comparison(syndata_filename, exact_subjset):
    '''Create a graph and data that has all the true values in it.

    Needs to all be in the same graph and needs to have the same variable names
    as in the inference graph.
    '''
    synth = synthetic.SyntheticData.load(syndata_filename)

    # Set the variables in the graph
    GRAPH = pl.graph.Graph(name='synthetic')
    d = data.Data(asvs=exact_subjset.asvs, subjects=exact_subjset, G=GRAPH, 
        data_logscale=True)
    n_asvs = d.n_asvs

    growth = pl.variables.Variable(G=GRAPH, name=STRNAMES.GROWTH_VALUE, 
        shape=(n_asvs,))
    growth.value = synth.dynamics.growth

    self_interactions = pl.variables.Variable(G=GRAPH, name=STRNAMES.SELF_INTERACTION_VALUE,
        shape=(n_asvs,))
    self_interactions.value = synth.dynamics.self_interactions

    interactions = pl.variables.Variable(G=GRAPH, name=STRNAMES.CLUSTER_INTERACTION_VALUE, 
        shape=(n_asvs, n_asvs))
    interactions.value = synth.dynamics.interactions.get_datalevel_value_matrix(
        set_neg_indicators_to_nan=False)

    if synth.dynamics.perturbations is not None:
        for synth_perturbation in synth.dynamics.perturbations:
            perturbation = pl.contrib.Perturbation( 
                start=synth_perturbation.start, 
                end=synth_perturbation.end,
                name=synth_perturbation.name,
                G=GRAPH, asvs=exact_subjset.asvs)
            indicators = synth_perturbation.indicator.item_bool_array()
            values = synth_perturbation.item_array(only_pos_ind=True)
            iii = 0
            for oidx in range(n_asvs):
                if indicators[oidx]:
                    perturbation.indicator.value[oidx] = True
                    perturbation.magnitude.value[oidx] = values[iii]
                    iii += 1
                else:
                    perturbation.indicator.value[oidx] = False
                    perturbation.magnitude.value[oidx] = 0
    
    # set the inference
    mlcrr = pl.inference.MLRR(constrain=True, graph=GRAPH)
    return mlcrr

if __name__ == '__main__':
    '''Build the parameters, model, and run the cross validation
    '''
    args = parse_args()
    if args.data_seed is None:
        args.data_seed = 9880035
    if args.init_seed is None:
        args.init_seed = 12114740

    ONLY_PLOT = False

    # Generate parameters
    params = config.MLCRRConfig(output_basepath=args.basepath, data_seed=args.data_seed,
        init_seed=args.init_seed, dataset=args.dataset, leave_out=args.leave_out, data_path=None)
    fparams = config.FilteringConfig(healthy=args.healthy, dataset=args.dataset)

    pl.seed(params.DATA_SEED)
    graph_name = 'graph' + params.suffix() + fparams.suffix()
    basepath = params.OUTPUT_BASEPATH + graph_name + '/'
    os.makedirs(basepath, exist_ok=True)
    lparams = config.LoggingConfig(basepath=basepath)

    subjset_filename = basepath + config.SUBJSET_FILENAME
    validate_subjset_filename = basepath + config.VALIDATION_SUBJSET_FILENAME
    mlcrr_filename = basepath + config.MLCRR_RESULTS_FILENAME
    graph_filename = basepath + config.GRAPH_FILENAME
    params_filename = basepath + config.PARAMS_FILENAME
    fparams_filename = basepath + config.FPARAMS_FILENAME
    syndata_filename = basepath + config.SYNDATA_FILENAME
    synparams_filename = basepath + config.SYNPARAMS_FILENAME

    subjset = pl.SubjectSet.load(params.DATA_FILENAME)
    if not ONLY_PLOT:
        params.save(params_filename)
        fparams.save(fparams_filename)

        # Get the steady states of the real data
        subjset_real = pl.SubjectSet.load(params.DATA_FILENAME)
        # Filtering and abundance normalization
        if fparams.DATASET == 'mdsine2-real-data':
            subjset = filtering.consistency(subjset, dtype=fparams.DTYPE,
                threshold=fparams.THRESHOLD, union_both_consortia=fparams.HEALTHY==-1,
                min_num_consecutive=fparams.MIN_NUM_CONSECUTIVE,
                colonization_time=fparams.COLONIZATION_TIME,
                min_num_subjects=fparams.MIN_NUM_SUBJECTS)
        elif fparams.DATASET == 'mdsine-cdiff':
            subjset = filtering.mdsine_cdiff_preprocess(subjset=subjset)
        else:
            raise ValueError('`dataset` ({}) not recognized'.format(fparams.DATASET))

        f = open(basepath + 'filtering_output.txt', 'w')
        for asv in subjset.asvs:
            f.write('Index: {}\n\tName: {}\n\tTaxonomy: {}\n\tSequence: {}\n'.format(asv.idx, asv.name, 
                pl.asvname_formatter(format='%(order)s %(family)s, %(genus)s, %(species)s',
                    asv=asv, asvs=subjset.asvs),asv.sequence))
        f.close()

        logging.info('num asvs after preprocess filtering {}'.format(len(subjset.asvs)))
        if params.DELETE_FIRST_TIMEPOINT:
            subjset.pop_times(0, sids='all')

        plotbasepath = basepath+'valid_asvs/'
        os.makedirs(plotbasepath, exist_ok=True) # Make the folder

        # Remove subjects if necessary
        if params.LEAVE_OUT != -1:
            validate_subjset = subjset.pop_subject(params.LEAVE_OUT)
            validate_subjset.save(validate_subjset_filename)
        else:
            validate_subjset = None

        matrixes = [subj.matrix()['abs'] for subj in subjset]
        read_depthses = [subj.read_depth() for subj in subjset]
        qpcrses = [np.sum(subj.matrix()['abs'], axis=0) for subj in subjset]

        # logging.info('Plotting Data')
        # for asv in subjset.asvs:
        #     logging.info('{}/{}'.format(asv.idx, len(subjset.asvs)))
        #     fig = plt.figure(figsize=(20,10))
        #     fig = filtering.plot_asv(
        #         subjset=subjset, asv=asv, fparams=fparams, fig=fig,
        #         legend=True, title_format='Subject %(sname)s',
        #         suptitle_format='%(index)s: %(name)s\n%(order)s, %(family)s, %(genus)s',
        #         yscale_log=True, matrixes=matrixes, read_depthses=read_depthses,
        #         qpcrses=qpcrses)
        #     plt.savefig(plotbasepath + '{}.pdf'.format(asv.name))
        #     plt.close()

        # Plot data
        subjs = [subj for subj in subjset]
        for i, subj in enumerate(subjs):
            pl.visualization.abundance_over_time(subj=subj, dtype='abs', legend=True,
                taxlevel='genus', set_0_to_nan=True, yscale_log=True)
            plt.savefig(basepath + 'data{}.pdf'.format(subj.name))
            plt.close()
        
        pl.visualization.abundance_over_time(subj=subjset, dtype='qpcr', include_errorbars=False, grid=True)
        fig = plt.gcf()
        fig.tight_layout()
        plt.savefig(basepath + 'qpcr_data.pdf')
        plt.close()

        pl.visualization.abundance_over_time(subj=subjset, dtype='read-depth', yscale_log=False, grid=True)
        plt.savefig(basepath + 'read_depths.pdf')
        plt.close()

        # Rescale if necessary
        if params.QPCR_NORMALIZATION_MAX_VALUE is not None:
            subjset.normalize_qpcr(max_value=params.QPCR_NORMALIZATION_MAX_VALUE)
            logging.info('Normalizing qPCR values. Normalization constant: {:.3E}'.format(
                subjset.qpcr_normalization_factor))
        subjset.save(subjset_filename)

        # Run the model
        results = mlcrr.runCV(params=params, subjset=subjset_filename, graph_name=graph_name)
        results.save(mlcrr_filename)
        params.save(params_filename)

    params = config.MLCRRConfig.load(params_filename)
    fparams = config.FilteringConfig.load(fparams_filename)
    mlcrr_result = pl.inference.MLRR.load(mlcrr_filename)

    if os.path.isfile(validate_subjset_filename):
        main_base.validate(
            src_basepath=basepath, model=mlcrr_result, 
            forward_sims=['sim-full'],
            yscale_log=True, run_on_copy=True,
            asv_prefix_formatter='%(index)s: (%(name)s) %(genus)s %(species2)s ',
            yticklabels='(%(name)s) %(genus)s %(species2)s: %(index)s',
            mp=5,output_dt=1/8, perturbations_additive=True,
            traj_error_metric=pl.metrics.RMSE,
            pert_error_metric=pl.metrics.RMSE,
            interaction_error_metric=pl.metrics.RMSE,
            growth_error_metric=pl.metrics.RMSE,
            si_error_metric=pl.metrics.PE,)

    

