import numpy as np
import os
import sys
import matplotlib.pyplot as plt

import pylab as pl

sys.path.append('..')
import synthetic
import model

chain_fname = '../output_real/pylab24/real_runs/strong_priors/fixed_top/healthy1_5_0.0001_rel_2_5/ds0_is0_b5000_ns15000_mo-1_logTrue_pertsmult/graph_leave_out-1/mcmc.pkl'

synth = synthetic.make_semisynthetic(chain=chain_fname, 
    min_bayes_factor=10, set_times=False,
    init_dist_start=1e3, init_dist_end=1e7,
    hdf5_filename='../output_real/pylab24/real_runs/strong_priors/fixed_top/healthy1_5_0.0001_rel_2_5/ds0_is0_b5000_ns15000_mo-1_logTrue_pertsmult/graph_leave_out-1/traces.hdf5')
synth.save('base_data/preprocessed_semisynthetic_healthy_uniform_dist.pkl')
sys.exit()


processvar = model.MultiplicativeGlobal(asvs=synth.asvs)
processvar.value = 0.05
for ridx in range(5):
    print(ridx)
    synth.generate_trajectories(dt=0.01)

subjset = synth.simulateExactSubjset()

for ridx in range(5):
    subj = subjset.iloc(ridx)
    M = subj.matrix()['abs']
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    for aidx in range(len(synth.asvs)):
        ax.plot(subj.times, M[aidx, :])
    ax.set_ylim(1e4,1e12)
    ax.set_yscale('log')
    pl.visualization.shade_in_perturbations(ax, synth.dynamics.perturbations)

    plt.savefig('ridx{}.pdf'.format(ridx))
    plt.close()
