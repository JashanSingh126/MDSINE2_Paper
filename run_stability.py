'''Compute the jacobian of the stability

    let x = -inv(A)@r
    then stability_jacobian = diag(x) @ A

'''
import mdsine2 as md2
import numpy as np
import argparse
import logging
from mdsine2.names import STRNAMES

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chain', '-c', type=str, dest='chain',
        help='This is the path of the chain for inference.')
    parser.add_argument('--outfile', '-o', type=str, dest='outfile',
        help='This is where you are saving the posterior renderings')
    parser.add_argument('--section', '-s', type=str, dest='section',
        help='Section to plot the variables of. Options: (`posterior`, ' \
            '`burnin`, `entire`)', default='posterior')
    args = parser.parse_args()
    md2.config.LoggingConfig()

    mcmc = md2.BaseMCMC.load(args.chain)
    section = args.section

    growth = mcmc.graph[STRNAMES.GROWTH_VALUE].get_trace_from_disk(section=section)
    si = mcmc.graph[STRNAMES.SELF_INTERACTION_VALUE].get_trace_from_disk(section=section)
    interactions = mcmc.graph[STRNAMES.INTERACTIONS_OBJ].get_trace_from_disk(section=section)

    interactions[np.isnan(interactions)] = 0
    for i in range(len(mcmc.graph.data.taxas)):
        interactions[:, i, i] = - si[:, i]

    jacobian = np.zeros(shape=interactions.shape)
    for gibb in range(interactions.shape[0]):
        if gibb % 1000 == 0:
            logging.info('{}/{}'.format(gibb, interactions.shape[0]))
        
        r = growth[i].reshape(-1,1)
        A = interactions[i]
        stability = np.diag(-(np.linalg.pinv(A) @ r).ravel())
        jacobian[i] = stability @ A

    np.save(args.outfile, jacobian)


