import mdsine2 as md2
from mdsine2.names import STRNAMES
import pandas as pd
import logging
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
import os

md2.LoggingConfig(level=logging.INFO)

# h_toy = md2.dataset.parse(name='healthy-toy-dataset', 
#                          taxonomy='bindertutorials/output/data/healthy-toy/taxonomy.tsv', 
#                          reads='bindertutorials/output/data/healthy-toy/reads.tsv', 
#                          qpcr='bindertutorials/output/data/healthy-toy/qpcr.tsv', 
#                          perturbations='bindertutorials/output/data/healthy-toy/perturbations.tsv', 
#                          metadata='bindertutorials/output/data/healthy-toy/metadata.tsv')
# uc_toy = md2.dataset.parse(name='uc-toy-dataset', 
#                          taxonomy='bindertutorials/output/data/uc-toy/taxonomy.tsv', 
#                          reads='bindertutorials/output/data/uc-toy/reads.tsv', 
#                          qpcr='bindertutorials/output/data/uc-toy/qpcr.tsv', 
#                          perturbations='bindertutorials/output/data/uc-toy/perturbations.tsv', 
#                          metadata='bindertutorials/output/data/uc-toy/metadata.tsv')
# rep_toy = md2.dataset.parse(name='replicates-toy-dataset', 
#                          taxonomy='bindertutorials/output/data/replicates-toy/taxonomy.tsv', 
#                          reads='bindertutorials/output/data/replicates-toy/reads.tsv', 
#                          qpcr='bindertutorials/output/data/replicates-toy/qpcr.tsv', 
#                          metadata='bindertutorials/output/data/replicates-toy/metadata.tsv')

# params = md2.config.NegBinConfig(
#     seed=0, burnin=100, n_samples=200,
#     checkpoint=100, basepath='output/negbin/run2')
# mcmc_negbin = md2.negbin.build_graph(params=params, graph_name=rep_toy.name, 
#                               subjset=rep_toy)
# mcmc_negbin = md2.negbin.run_graph(mcmc_negbin, crash_if_error=True)


# # Learn a synthetic system
# # ------------------------
# # Learn the clustering
# a0 = md2.summary(mcmc_negbin.graph[STRNAMES.NEGBIN_A0])['mean']
# a1 = md2.summary(mcmc_negbin.graph[STRNAMES.NEGBIN_A1])['mean']

# basepath = 'output/semisynth/unfixed'
# os.makedirs(basepath, exist_ok=True)
# params = md2.config.MDSINE2ModelConfig(
#     basepath=basepath, seed=0, burnin=50, 
#     n_samples=100, negbin_a0=a0, negbin_a1=a1, checkpoint=50)
# params.INITIALIZATION_KWARGS[STRNAMES.CLUSTERING]['value_option'] = 'no-clusters'
# mcmc_uc0 = md2.initialize_graph(params=params, graph_name=uc_toy.name, subjset=uc_toy)
# mcmc_uc0 = md2.run_graph(mcmc_uc0, crash_if_error=True)

# # Fix the cluster assignments
# basepath='output/semisynth/fixed'
# params = md2.config.MDSINE2ModelConfig(
#     basepath=basepath, seed=0, burnin=50, n_samples=100, 
#     negbin_a0=a0, negbin_a1=a1, checkpoint=50)
# params.LEARN[STRNAMES.CLUSTERING] = False
# params.LEARN[STRNAMES.CONCENTRATION] = False
# params.INITIALIZATION_KWARGS[STRNAMES.CLUSTERING]['value_option'] = 'fixed-clustering'
# params.INITIALIZATION_KWARGS[STRNAMES.CLUSTERING]['value'] = 'output/semisynth/unfixed/mcmc.pkl'
# mcmc = md2.initialize_graph(params=params, graph_name=uc_toy.name, subjset=uc_toy)
# mcmc = md2.run_graph(mcmc, crash_if_error=True)




# Make the synthetic system
mcmc = md2.BaseMCMC.load('output/semisynth/fixed/mcmc.pkl')
syn = md2.synthetic.make_semisynthetic(
    mcmc, min_bayes_factor=10, name='semisynth', set_times=True)

# inter = syn.model.interactions
# for i in range(inter.shape[0]):
#     inter[i,i] = 0
# print(inter)
# md2.visualization.render_interaction_strength(inter, True, syn.taxa)
# plt.show()

# print(syn.model.perturbations)
# print(len(syn.perturbations))
# print(mcmc.graph.perturbations)
# sys.exit()


# make subject names
syn.set_subjects(['subj-{}'.format(i+1) for i in range(4)])

init_dist = md2.variables.Uniform(low=1e5, high=1e7)
print(init_dist.sample(size=12))

# Generate the trajectories
# YOURE HERE, WHY IS THIS RETURNING NAN
syn.generate_trajectories(dt=0.01, init_dist=md2.variables.Uniform(
    low=1e5, high=1e7), processvar=md2.model.MultiplicativeGlobal(value=0.05**2))

d = syn._data['subj-1']
print(d)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# for i in range(d.shape[0]):
#     ax.plot(syn.times, d[i, :], label=syn.taxa[i].name)
# plt.show()


# # Simulate noise
# study = syn.simulateMeasurementNoise(a0=1e-10, a1=0.05, qpcr_noise_scale=0.25, 
#                                      approx_read_depth=50000, name='semi-synth')