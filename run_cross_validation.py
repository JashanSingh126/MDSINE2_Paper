'''Run leave-one-out cross validation.

Author: David Kaplan
Date: 11/30/20
MDSINE2 version: 4.0.4

This script runs inference for each cross validation sequentially only, but you
can still run multiprocessing for each fold of inference.

Parameters
----------
--dataset, -d : strto do cross validation with
    This is the dataset to do inference with
--cv-basepath, -b : str
    Folder location to save to. Note that this is the basepath for cross validation
    as a whole. Each fold has its own folder within this folder
--dset-basepath, -db : str
    This is the path to save the datasets that are created for each CV
--negbin-run : str
    This is the MCMC object that was run to learn a0 and a1
--seed, -s : int
    This is the seed to initialize the inference with
--burnin, -b : int
    How many burn-in Gibb steps for Markov Chain Monte Carlo (MCMC)
--n-samples, -n : int
    Total number Gibb steps to perform during MCMC inference
--checkpoint, -c : int
    How often to write the posterior to disk. Note that `--burnin` and
    `--n-samples` must be a multiple of `--checkpoint` (e.g. checkpoint = 100, 
    n_samples = 600, burnin = 300)
--multiprocessing, -mp : int
    If 1, run the inference with multiprocessing. Else run on a single process
'''
import mdsine2 as md2
import argparse
import logging
import os
import sys

command_fmt = 'python step_5_infer_mdsine2.py --input {dset} ' \
    '--negbin-run {negbin} ' \
    '--seed {seed} ' \
    '--burnin {burnin} ' \
    '--n-samples {n_samples} ' \
    '--checkpoint {ckpt} ' \
    '--basepath {basepah} ' \
    '--multiprocessing {mp}'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, dest='dataset',
        help='This is the Gibson dataset we want to do cross validation on')
    parser.add_argument('--cv-basepath', '-o', type=str, dest='output_basepath',
        help='This is the basepath to save the output')
    parser.add_argument('--dset-basepath', '-db', type=str, dest='input_basepath',
        help='This is the basepath to load and save the cv datasets')
    parser.add_argument('--negbin-run', type=str, dest='negbin',
        help='This is the MCMC object that was run to learn a0 and a1')
    parser.add_argument('--seed', '-s', type=int, dest='seed',
        help='This is the seed to initialize the inference with')
    parser.add_argument('--burnin', '-nb', type=int, dest='burnin',
        help='How many burn-in Gibb steps for Markov Chain Monte Carlo (MCMC)')
    parser.add_argument('--n-samples', '-ns', type=int, dest='n_samples',
        help='Total number Gibb steps to perform during MCMC inference')
    parser.add_argument('--checkpoint', '-c', type=int, dest='checkpoint',
        help='How often to write the posterior to disk. Note that `--burnin` and ' \
             '`--n-samples` must be a multiple of `--checkpoint` (e.g. checkpoint = 100, ' \
             'n_samples = 600, burnin = 300)')
    parser.add_argument('--basepath', '-b', type=str, dest='basepath',
        help='This is folder to save the output of inference')
    parser.add_argument('--multiprocessing', '-mp', type=int, dest='mp',
        help='If 1, run the inference with multiprocessing. Else run on a single process',
        default=0)
    
    args = parser.parse_args()

    md2.config.LoggingConfig(level=logging.INFO)

    input_basepath = args.input_basepath
    os.makedirs(input_basepath, exist_ok=True)
    os.makedirs(args.output_basepath, exist_ok=True)

    logging.info('Loading dataset {}'.format(args.dataset))
    study_master = md2.Study.load(args.dataset)

    for subj in study_master:
        logging.info('Leave out {}'.format(subj.name))
        study = md2.Study.load(args.dataset)
        val_study = study.pop_subject(subj.name)
        study.name = study.name + '-cv{}'.format(subj.name)

        # Save the datasets
        val_study.name = study.name + '-validate'
        study_fname = os.path.join(input_basepath, study.name + '.pkl')
        study.save(study_fname)
        val_study.save(os.path.join(input_basepath, val_study.name + '.pkl'))

        logging.info('Run inference')
        command = command_fmt.format(
            dset=study_fname, negbin=args.negbin, seed=args.seed, 
            burnin=args.burnin, n_samples=args.n_samples, ckpt=args.checkpoint,
            basepath=args.output_basepath, mp=args.mp)
        logging.info(command)
        os.system(command)
        
