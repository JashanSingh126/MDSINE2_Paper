'''Learn the Negative Binomial dispersion parameters that parameterize the noise
of the reads. We learn these parameters with replicate measurements of the reads.
In the Gibson dataset, this can be accessed by `dset = mdsine2.dataset.gibson('replicates)`.


Author: David Kaplan
Date: 11/18/20
MDSINE2 version: 4.0.2

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
4) Plot output (if you decide to)

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
--only-plot : int
    If 1, it only plots the output and looks for the runs. Otherwise it runs the model
--plot-output : int
    If 1, plot the output. Otherwise do not plot the output

Reproducability
---------------
To reproduce the paper, first run the script `gibson_2_filtering.py`, then:
Linux/MacOS:
python gibson_3_negbin_a0a1.py \
    --input gibson_output/datasets/gibson_replicate_agg.pkl \
    --other-datasets gibson_output/datasets/gibson_healthy_agg_filtered.pkl \
    --seed 0 \
    --burnin 2000 \
    --n-samples 6000 \
    --checkpoint 200 \
    --basepath gibson_output/output/negbin/ \
    --plot-output 1
PC:
python gibson_3_negbin_a0a1.py `
    --input gibson_output/datasets/gibson_replicate_agg.pkl `
    --other-datasets gibson_output/datasets/gibson_healthy_agg_filtered.pkl `
    --seed 0 `
    --burnin 2000 `
    --n-samples 6000 `
    --checkpoint 200 `
    --basepath gibson_output/output/negbin/ `
    --plot-output 1
'''
import argparse
import mdsine2 as md2
import logging
import os.path
import matplotlib.pyplot as plt
from mdsine2.names import STRNAMES

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', '-i', type=str, dest='input',
        help='This is the dataset to do inference with. If `--only-plot` = 1,' \
            ' then this is the basepath of the run that contains the subjset')
    parser.add_argument('--other-datasets', '-d', type=str, dest='other_datasets',
        help='These are the other datasets that we filter the union of. If nothing is ' \
             'provided then we do not do any filtering.', nargs='+', default=None)
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
    parser.add_argument('--plot-output', '-p', type=str, dest='plot_output',
        help='If 1, plot the output. Otherwise do not plot the output', default=0)
    parser.add_argument('--plot-output', '-p', type=str, dest='only_plot',
        help='If 1, it only plots the output and looks for the runs. Otherwise it ' \
            'runs the model', default=0)
    args = parser.parse_args()
    md2.config.LoggingConfig(level=logging.INFO)

    if args.only_plot == 1:
        mcmc = md2.BaseMCMC.load(os.path.join(args.input, md2.config.MCMC_FILENAME))
        params = md2.config.MDSINE2ModelConfig.load(os.path.join(args.input, md2.config.PARAMS_FILENAME))
        study = md2.Study.load(os.path.join(args.input, md2.config.SUBJSET_FILENAME))
    else:
        # 1) Load the dataset
        logging.info('Loading dataset {}'.format(args.input))
        study = md2.Study.load(args.input)
        params = md2.config.NegBinConfig(seed=args.seed, burnin=args.burnin, n_samples=args.n_samples,
            ckpt=args.checkpoint, basepath=args.basepath)

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
        mcmc = md2.negbin.build_graph(params=params, graph_name=study.name, subjset=study)
        mcmc = md2.negbin.run_graph(mcmc, crash_if_error=True)
        mcmc.save()
        study.save(os.path.join(params.MODEL_PATH, md2.config.SUBJSET_FILENAME))

    if args.plot_output == 1 or args.only_plot == 1:
        print('Plotting learned model')
        fig = md2.negbin.visualize_learned_negative_binomial_model(mcmc)
        fig.tight_layout()
        path = os.path.join(params.MODEL_PATH, 'learned_model.pdf')
        plt.savefig(path)
        plt.close()

        f = open(os.path.join(params.MODEL_PATH, 'a0a1.txt'), 'w')
        mcmc.graph[STRNAMES.NEGBIN_A0].visualize(
            path=os.path.join(params.MODEL_PATH, 'a0.pdf'), 
            f=f, section='posterior')
        mcmc.graph[STRNAMES.NEGBIN_A1].visualize(
            path=os.path.join(params.MODEL_PATH, 'a1.pdf'), 
            f=f, section='posterior')
        f.close()
        print('Plotting filtering')
        mcmc.graph[STRNAMES.FILTERING].visualize(
            basepath=params.MODEL_PATH, section='posterior')





    



    
    
    
    
    
    