'''Learn the Negative Binomial dispersion parameters that parameterize the noise
of the reads. We learn these parameters with replicate measurements of the reads.

Author: David Kaplan
Date: 11/30/20
MDSINE2 version: 4.0.4

Parameters
----------
--input, -i : str
    This is the dataset to do inference with
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
'''
import mdsine2 as md2
import argparse
import logging
import os.path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, dest='input',
        help='This is the dataset to do inference with.')
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
    args = parser.parse_args()
    md2.config.LoggingConfig(level=logging.INFO)

    # 1) Load the dataset
    logging.info('Loading dataset {}'.format(args.input))
    study = md2.Study.load(args.input)
    params = md2.config.NegBinConfig(seed=args.seed, burnin=args.burnin, n_samples=args.n_samples,
        ckpt=args.checkpoint, basepath=args.basepath)

    # 2) Perform inference
    mcmc = md2.negbin.build_graph(params=params, graph_name=study.name, subjset=study)
    mcmc = md2.negbin.run_graph(mcmc, crash_if_error=True)
    mcmc.save()
    study.save(os.path.join(params.MODEL_PATH, md2.config.SUBJSET_FILENAME))

