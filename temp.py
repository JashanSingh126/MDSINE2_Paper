import mdsine2 as md2
from mdsine2.names import STRNAMES
import pandas as pd
import logging
import re
import sys
import numpy as np
import matplotlib.pyplot as plt

md2.LoggingConfig(level=logging.INFO)

##################################################
# CV
rep_toy = md2.dataset.parse(
        name='replicates-toy-dataset', 
        taxonomy='bindertutorials/data/replicates-toy/taxonomy.tsv', 
        reads='bindertutorials/data/replicates-toy/reads.tsv', 
        qpcr='bindertutorials/data/replicates-toy/qpcr.tsv', 
        metadata='bindertutorials/data/replicates-toy/metadata.tsv')
# params = md2.config.NegBinConfig(
#     seed=0, burnin=100, n_samples=200,
#     ckpt=100, basepath='output/negbin/run2')
# mcmc_negbin = md2.negbin.build_graph(params=params, graph_name=rep_toy.name, 
#         subjset=rep_toy)
# mcmc_negbin = md2.negbin.run_graph(mcmc_negbin, crash_if_error=True)

mcmc_negbin = md2.BaseMCMC.load('bindertutorials/output/negbin/run2/mcmc.pkl')
a0 = md2.summary(mcmc_negbin.graph[STRNAMES.NEGBIN_A0])['mean']
a1 = md2.summary(mcmc_negbin.graph[STRNAMES.NEGBIN_A1])['mean']


# Learn the dataset
study = md2.dataset.parse(
    name='healthy-toy-dataset', 
    taxonomy='bindertutorials/data/healthy-toy/taxonomy.tsv', 
    reads='bindertutorials/data/healthy-toy/reads.tsv', 
    qpcr='bindertutorials/data/healthy-toy/qpcr.tsv', 
    perturbations='bindertutorials/data/healthy-toy/perturbations.tsv', 
    metadata='bindertutorials/data/healthy-toy/metadata.tsv')

val = study.pop_subject('2')
val.name += '-validate'

print(study.name)
for subj in study:
    print(subj.name)
for pert in study.perturbations:
    print(pert)
print()
print(val.name)
for subj in val:
    print(subj.name)
for pert in val.perturbations:
    print(pert)

# params = md2.config.MDSINE2ModelConfig(
#     basepath='bindertutorials/output/mdsine2/cv/'+study.name, seed=0, 
#     burnin=50, n_samples=100, 
#     negbin_a0=a0, negbin_a1=a1, checkpoint=50)
# params.INITIALIZATION_KWARGS[STRNAMES.CLUSTERING]['value_option'] = 'no-clusters'
# params.INITIALIZATION_KWARGS[STRNAMES.PERT_INDICATOR_PROB]['hyperparam_option'] = 'weak-agnostic'
# params.INITIALIZATION_KWARGS[STRNAMES.CLUSTER_INTERACTION_INDICATOR_PROB]['hyperparam_option'] = 'weak-agnostic'
# mcmc = md2.initialize_graph(params=params, graph_name=study.name, subjset=study)
# mcmc = md2.run_graph(mcmc, crash_if_error=True)
mcmc = md2.BaseMCMC.load('bindertutorials/output/mdsine2/cv/healthy-toy-dataset/mcmc.pkl')

subj = val['2']
M_truth = subj.matrix()['abs']
initial_conditions = M_truth[:,0]
initial_conditions[initial_conditions==0] = 1e5
times = subj.times


M = md2.model.gLVDynamicsSingleClustering.forward_sim_from_chain(
    mcmc, val=subj, initial_conditions=initial_conditions, times=times, 
    simulation_dt=0.01)

print(M.shape)

low = np.percentile(M, q=25, axis=0)
high = np.percentile(M, q=75, axis=0)
med = np.percentile(M, q=50, axis=0)

oidx = 12

taxas = subj.taxas
fig = plt.figure()
ax = fig.add_subplot(111)
ax.fill_between(times, y1=low[oidx, :], y2=high[oidx, :], alpha=0.2)
ax.plot(times, med[oidx,:], label='Forward Sim')
ax.plot(times, M_truth[oidx, :], label='Data', marker='x', color='black',
       linestyle=':')
ax.set_yscale('log')

md2.visualization.shade_in_perturbations(ax, perturbations=subj.perturbations, subj=subj)
ax.set_ylim(bottom=1e5, top=1e12)

ax.legend()

fig.suptitle(md2.taxaname_for_paper(taxas[oidx], taxas))
plt.show()


# # Seeeeeeeed
# healthy = md2.Study.load('processed_data/gibson_healthy_agg_taxa_filtered.pkl')
# to_delete = []
# for taxa in healthy.taxas:
#     if taxa.idx > 10:
#         to_delete.append(taxa.name)
# healthy.pop_taxas(to_delete)

# params1 = md2.config.MDSINE2ModelConfig(
#         basepath='tmp/params1', seed=0, 
#         burnin=5, n_samples=10, negbin_a1=0.0025, 
#         negbin_a0=0.025, checkpoint=5)
# params1.INITIALIZATION_KWARGS[STRNAMES.CLUSTERING]['value_option'] = 'no-clusters'
# mcmc1 = md2.initialize_graph(params=params1, graph_name=healthy.name, subjset=healthy)

# params2 = md2.config.MDSINE2ModelConfig(
#         basepath='tmp/params1', seed=1, 
#         burnin=5, n_samples=10, negbin_a1=0.0025, 
#         negbin_a0=0.025, checkpoint=5)
# params2.INITIALIZATION_KWARGS[STRNAMES.CLUSTERING]['value_option'] = 'no-clusters'
# mcmc2 = md2.initialize_graph(params=params2, graph_name=healthy.name, subjset=healthy)

# mcmc1 = md2.run_graph(mcmc1, crash_if_error=True)
# mcmc2 = md2.run_graph(mcmc2, crash_if_error=True)


# ele1 = mcmc1.graph[STRNAMES.FILTERING].x.value
# ele2 = mcmc2.graph[STRNAMES.FILTERING].x.value

# for idx in range(len(ele1)):
#     e1 = ele1[idx].value
#     e2 = ele2[idx].value
#     print(e1-e2)