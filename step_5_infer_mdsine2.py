'''Run MDSINE2 inference

Author: David Kaplan
Date: 11/30/20
MDSINE2 version: 4.0.4

Parameters
----------
--input, -i : str
    This is the dataset to do inference with
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
--basepath, -b : str
    Folder location to save to
--multiprocessing : int
    If 1, run the inference with multiprocessing. Else run on a single process
'''
import argparse
import mdsine2 as md2
import logging
import os.path
import time
from mdsine2.names import STRNAMES

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', '-i', type=str, dest='input',
        help='This is the dataset to do inference with.')
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

    # 1) load dataset
    logging.info('Loading dataset {}'.format(args.input))
    study = md2.Study.load(args.input)

    # 2) Load the negative binomial parameters
    negbin = md2.BaseMCMC.load(args.negbin)
    a0 = md2.summary(negbin.graph[STRNAMES.NEGBIN_A0])['mean']
    a1 = md2.summary(negbin.graph[STRNAMES.NEGBIN_A1])['mean']
    print('Setting a0 = {:.4E}, a1 = {:.4E}'.format(a0,a1))

    # 3) Begin inference
    params = md2.config.MDSINE2ModelConfig(
        basepath=args.basepath,
        data_seed=args.seed, init_seed=args.seed, burnin=args.burnin, n_samples=args.n_samples, 
        negbin_a1=a1, negbin_a0=a0, checkpoint=args.checkpoint)
    # Run with multiprocessing if necessary
    if args.mp == 1:
        params.MP_FILTERING = 'full'
        params.MP_CLUSTERING = 'full-4'

    mdata_fname = os.path.join(params.MODEL_PATH, 'metadata.txt')
    mcmc = md2.initialize_graph(params=params, graph_name=study.name, 
        subjset=study)
    params.make_metadata_file(fname=mdata_fname)

    start_time = time.time()
    mcmc = md2.run_graph(mcmc, crash_if_error=True)

    # Record how much time inference took
    t = time.time() - start_time
    t = t / 3600 # Convert to hours

    f = open(mdata_fname, 'a')
    f.write('\n\nTime for inference: {} hours'.format(t))
    f.close()
