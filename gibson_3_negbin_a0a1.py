'''Learn the Negative Binomial dispersion parameters that parameterize the noise
of the reads. We learn these parameters with replicate measurements of the reads.
In the Gibson dataset, this can be accessed by `dset = mdsine2.dataset.gibson('replicates)`.


Author: David Kaplan
Date: 11/18/20
MDSINE2 version: 4.0.1

Methodology
-----------
1) Load in dataset
   This is a pickle of an `mdsine2.Study` object. This can be created with the 
   `MDSINE2.gibson_2_filtering.py` script.
2) Filter the ASVs
   We only run the inference on the ASVs/OTUs that pass filtering (either
   `mdsine2.consistency_filtering` or `mdsine2.conditional_consistency_filtering`).
   We set the ASVs that we learn with to be the union of the ASVs/OTUs that pass
   filtering for the 'healthy' and 'uc' datasets.
3) Perform inference
   Learn the negative binomial dispersion parameters. Fine tuning of the parameters
   can be done by accessing the `mdsine2.config.NegBinConfig` class:
   ```python
    import mdsine2 as md2
    from mdsine2.names import STRNAMES

   params = md2.config.NegBinConfig(...)
   # Modify the inference order
   params.INFERENCE_ORDER = ...
   # Modify the target acceptance rate for the a0 parameter
   params.INITIALIZATION_KWARGS[STRNAMES.NEGBIN_A0]['target_acceptance_rate'] = ...
   ```
4) Plot the posterior
   Plots the output of the inference for all of the parameters learned.

Parameters
----------
--input, -i : str
    This is the dataset to do inference with
--other-datasets, -d : str, +
    These are the other datasets that we filter the union of. If nothing is
    provided then we do not do any filtering
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

Reproducability
---------------
To reproduce the paper, first run the script `gibson_2_filtering.py`, then:
python gibson_3_negbin_a0a1.py \
    --input gibson_output/datasets/gibson_replicate_agg.pkl \
    --other-datasets gibson_output/datasets/gibson_healthy_agg_filtered.pkl gibson_output/datasets/gibson_uc_agg_filtered.pkl \
    --seed 0 \
    --burnin 2000 \
    --n-samples 10000 \
    --checkpoint 200 \
    --basepath gibson_output/output_negbin/
'''
import argparse
import mdsine2 as md2
import logging

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', '-i', type=str, dest='input',
        help='This is the dataset to do inference with.')
    parser.add_argument('--other-datasets', '-d', type=str, dest='other_datasets',
        help='These are the other datasets that we filter the union of. If nothing is ' \
             'provided then we do not do any filtering.', nargs='+', default=None)
    parser.add_argument('--seed', '-s', type=int, dest='seed',
        help='This is the seed to initialize the inference with')
    parser.add_argument('--burnin', '-b', type=int, dest='burnin',
        help='How many burn-in Gibb steps for Markov Chain Monte Carlo (MCMC)')
    parser.add_argument('--n-samples', '-n', type=int, dest='n_samples',
        help='Total number Gibb steps to perform during MCMC inference')
    parser.add_argument('--checkpoint', '-c', type=int, dest='checkpoint',
        help='How often to write the posterior to disk. Note that `--burnin` and ' \
             '`--n-samples` must be a multiple of `--checkpoint` (e.g. checkpoint = 100, ' \
             'n_samples = 600, burnin = 300)')
    args = parser.parse_args()
    md2.config.LoggingConfig(level=logging.INFO)

    # 1) Load the dataset
    logging.info('Loading dataset {}'.format(args.input))
    study = md2.Study.load(args.input)

    # 2) Filter if necessary
    if args.other_datasets is not None:
        logging.info('Filtering with datasets {}'.format(args.other_datasets))
        asvs = set([])

        for fname in args.other_datasets:
            temp = md2.Study.load(fname)
            for asv in temp.asvs:
                asvs.add(asv.name)
        
        logging.info('A total of {} unique items were found to do inference with'.format(len(asvs)))

        to_delete = []
        for asv in study.asvs:
            if asv.name not in asvs:
                to_delete.append(asv.name)
        study.pop_asvs(to_delete)

    # 3) Perform inference
    params = md2.config.NegBinConfig(seed=args.seed, burnin=args.burnin, n_samples=args.n_samples,
        ckpt=args.checkpoint, basepath=args.basepath)
    



    
    
    
    
    
    