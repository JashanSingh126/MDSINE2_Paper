'''Visualize the posterior of the inference of the Negative Binomial dispersion
parameters that was produced with the script `step_3_infer_negbin`

Author: David Kaplan
Date: 11/30/20
MDSINE2 version: 4.0.4

Parameters
----------
--chain, -c : str
    This is the MCMC object that inference was performed on. This is most likely
    the `mcmc.pkl` file that is in the output folder of `step_3_infer_negbin`
--output-basepath, -o : str
    This is the folder to save the output
'''
import argparse
import mdsine2 as md2
import logging
import os.path
import matplotlib.pyplot as plt
from mdsine2.names import STRNAMES

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chain', '-c', type=str, dest='chain',
        help='This is the MCMC object that inference was performed on. This is most likely' \
        'the `mcmc.pkl` file that is in the output folder of `step_3_infer_negbin`')
    parser.add_argument('--output-basepath', '-o', type=str, dest='basepath',
        help='This is the folder to save the output.')
    args = parser.parse_args()
    md2.config.LoggingConfig(level=logging.INFO)

    mcmc = md2.BaseMCMC.load(args.chain)
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
    
