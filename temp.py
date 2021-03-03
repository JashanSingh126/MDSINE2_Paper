import mdsine2 as md2
from mdsine2.names import STRNAMES
import numpy as np
import matplotlib.pyplot as plt

# Make the dynamics
syn = md2.Synthetic(name='icml', seed=0)
syn.icml_dynamics(n_taxa=13)

md2.visualization.render_interaction_strength(
    syn.model.interactions, log_scale=False, taxa=syn.taxa,
    center_colors=True)

# Set the number of subjects
syn.set_subjects(['subj{}'.format(i+1) for i in range(5)])
syn.subjs

# Set the timepoints
syn.set_timepoints(times=np.arange(14))
print(syn.times)

# Forward simulate
pv = md2.model.MultiplicativeGlobal(0.01**2) # 5% process variation
syn.generate_trajectories(dt=0.01, init_dist=md2.variables.Uniform(low=2, high=20),
                          processvar=pv)

# Plot the data without measurement noise
fig = plt.figure()
ax = fig.add_subplot(111)
for taxon in range(13):
    ax.plot(syn.times, syn._data['subj1'][taxon, :], label=syn.taxa[taxon].name)
ax.legend(bbox_to_anchor=(1.05, 1))
ax.set_ylabel('CFUs/g')
ax.set_xlabel('Days')

# Make a study object and simulate measurement noise
study = syn.simulateMeasurementNoise(
    a0=1e-10, a1=0.001, qpcr_noise_scale=0.01, approx_read_depth=60000, 
    name='sim-study')

print(len(study.taxa))
print(len(study))
      

# Visualize with noise
md2.visualization.abundance_over_time(study['subj1'], dtype='abs', yscale_log=False)

plt.show()

# Inference with the ICML dataset
params = md2.config.MDSINE2ModelConfig(
    basepath='output/mdsine2', seed=1, burnin=200, n_samples=400, 
    checkpoint=100, negbin_a0=1e-10, negbin_a1=0.001)

params.INITIALIZATION_KWARGS[STRNAMES.CLUSTERING]['value_option'] = 'no-clusters'
mcmc = md2.initialize_graph(params=params, graph_name='icml-graph', subjset=study)

clustering = mcmc.graph[STRNAMES.CLUSTERING_OBJ]
print(clustering.coclusters.value.shape)
print(len(clustering))

mcmc = md2.run_graph(mcmc, crash_if_error=True)


clustering = mcmc.graph[STRNAMES.CLUSTERING_OBJ]
coclusters = md2.summary(clustering.coclusters)['mean']
md2.visualization.render_cocluster_probabilities(coclusters, taxa=study.taxa)

interactions = mcmc.graph[STRNAMES.INTERACTIONS_OBJ]
A = md2.summary(interactions, set_nan_to_0=True)['mean']
md2.visualization.render_interaction_strength(A, log_scale=True, taxa=study.taxa, 
                                              center_colors=True)

bf = md2.generate_interation_bayes_factors_posthoc(mcmc)
md2.visualization.render_bayes_factors(bf, taxa=study.taxa)

plt.show()