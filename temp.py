import sys
import logging
import os
import pandas as pd
import os.path
import numpy as np

import mdsine2 as md2
from mdsine2.names import STRNAMES
from Bio import SeqIO, Seq, SeqRecord

md2.config.LoggingConfig()

import matplotlib.pyplot as plt


original_seqs = md2.dataset.gibson()
otu_seqs = md2.Study.load('gibson')



sys.exit()

# consec = 9
# dset = 'uc'
# subjset = md2.Study.load('gibson_output/datasets/gibson_{}_agg.pkl'.format(dset))
# subjset = md2.consistency_filtering(subjset, dtype='rel', threshold=0.0001,
#     min_num_consecutive=consec, colonization_time=5, min_num_subjects=2)
# f = open('{}_{}.txt'.format(dset, consec), 'w')
# for i, taxa in enumerate(subjset.taxas):
#     f.write(taxa.name)
#     if i != len(subjset.taxas) - 1:
#         f.write('\n')
# f.close()
# sys.exit()

# dset = 'uc'
# subjset = md2.Study.load('gibson_output/datasets/gibson_{}_agg.pkl'.format(dset))

# df = subjset.df(dtype='abs', agg='mean', times='union')

# df.to_csv(dset + '_abund.tsv', index=True, header=True, sep='\t')



# sys.exit()



# subjset = md2.Study.load('gibson_output/datasets/gibson_healthy_agg.pkl')
# basedir = 'figs/'
# os.makedirs(basedir, exist_ok=True)

# for taxa in subjset.taxas:
#     if md2.isaggregatedtaxa(taxa):

#         fig = plt.figure(figsize=(15,8))
#         for i, subj in enumerate(subjset):
#             legend = i == 1
#             md2.visualization.aggregate_taxa_abundances(subj, agg=taxa, 
#                 ax=fig.add_subplot(2,2,i+1), vmin=1e-5, vmax=0.99, legend=legend)
        
#         fig.suptitle(md2.taxaname_for_paper(taxa, taxas=subjset.taxas))
#         fig.tight_layout()
#         plt.savefig(basedir + taxa.name + '.pdf')
#         plt.close()
#         # sys.exit()
            


# sys.exit()



# healthy = md2.dataset.gibson(dset='healthy')
# uc = md2.dataset.gibson(dset='uc')

# seqs = SeqIO.parse('gibson_files/preprocessing/prefiltered_out_taxas_aligned.fa', format='fasta')
# dset = md2.dataset.gibson()

# ret = []
# fmt = '%(kingdom)s, %(phylum)s, %(class)s, %(order)s, %(family)s, %(genus)s, %(species)s'
# for record in seqs:



#     seq = SeqRecord.SeqRecord(
#         name=record.name,
#         id=record.name,
#         description=md2.taxaname_formatter(format=fmt, taxa=record.name, taxas=dset.taxas),
#         seq=record.seq)    
#     ret.append(seq)

# SeqIO.write(ret, 'gibson_files/preprocessing/prefiltered_out_taxas_aligned.fa', format='fasta')


# dset = md2.dataset.gibson()
# to_delete = []
# for taxa in dset.taxas:
#     if taxa.idx >= 10:
#         to_delete.append(taxa.name)
# dset.pop_taxas(to_delete)
# for i, taxa in enumerate(dset.taxas):
#     print('{}:{}'.format(i+1, taxa.name))

# clusters = [0,2,0,1,2,1,3,2,0,4]
# clustering = md2.Clustering(clusters=clusters, items=dset.taxas)
# print(clustering)
# print()
# clustering.move_item(9, 0)
# print(clustering)

import numpy as np

def _single_calc_mean_var(means, variances, a0, a1, rels, read_depths):
    i = 0
    for col in range(rels.shape[1]):
        for oidx in range(rels.shape[0]):
            mean = rels[oidx, col] * read_depths[col]
            disp = a0 / mean + a1
            variances[i] = mean + disp * (mean**2)
            means[i] = mean

            i += 1
    return means, variances

import seaborn as sns
def visualize_learned_negative_binomial_model(a0,a1, subjset):
    '''Visualize the negative binomial dispersion model.

    Plot variance on y-axis, mean on x-axis. both in logscale.

    Parameters
    ----------
    mcmc : mdsine2.BaseMCMC
        This is the inference object with the negative binomial posteriors
        and the data it was learned on
    section : str
        Section of the trace to compute on. Options:
            'posterior' : posterior samples
            'burnin' : burn-in samples
            'entire' : both burn-in and posterior samples
    
    Returns
    -------
    matplotlib.pyplot.Figure
    '''
    # Get the data
    # ------------
    reads = []
    for subj in subjset:
        reads.append(subj.matrix()['raw'])
    reads = np.hstack(reads)
    read_depths = np.sum(reads, axis=0)
    rels = reads / read_depths + 1e-20

    # Get the traces of a0 and a1
    # ---------------------------
    means = np.zeros(rels.size, dtype=float)
    variances = np.zeros(rels.size, dtype=float)

    _single_calc_mean_var(
        means=means,
        variances=variances,
        a0=a0, a1=a1, rels=rels, 
        read_depths=read_depths)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # plot the data
    colors = sns.color_palette()
    for sidx, subj in enumerate(subjset):
        reads_subj = subj.matrix()['raw']

        x = np.mean(reads_subj, axis=1)
        y = np.var(reads_subj, axis=1)

        idxs = x > 0
        x = x[idxs]
        y = y[idxs]

        ax.scatter(
            x=x, y=y, alpha=0.5,
            c=colors[sidx], rasterized=False, 
            label='Subject {}'.format(subj.name))
    
    idxs = np.argsort(means)
    print(means.shape)
    print(variances.shape)

    ax.plot(means[idxs], variances[idxs], color='black', label='Fitted NegBin Model', rasterized=False)

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Mean (counts)')
    ax.set_ylabel('Variance (counts)')
    ax.set_title('Empirical mean vs variance of counts')
    ax.set_xlim(left=0.5)
    ax.set_ylim(bottom=0.5)
    ax.legend()

    plt.show()


from mdsine2.names import STRNAMES

# healthy = md2.Study.load('gibson_output/datasets/gibson_healthy_agg.pkl')
# healthy = md2.consistency_filtering(healthy, dtype='rel', threshold=0.0001,
#     min_num_consecutive=7, colonization_time=5, min_num_subjects=2)
subjset = md2.Study.load('gibson_output/datasets/gibson_replicate_agg.pkl')
# subjset.pop_taxas_like(healthy)

# visualize_learned_negative_binomial_model(a0=0.1, a1=.036, subjset=subjset)

params = md2.config.NegBinConfig(0, 1000, 2000, 500, 'output_negbin1/')
params.LEARN[STRNAMES.FILTERING] = True
params.LEARN[STRNAMES.NEGBIN_A0] = True
mcmc = md2.negbin.build_graph(params=params, graph_name='replicates', subjset=subjset)
mcmc = md2.negbin.run_graph(mcmc, True)

fig = md2.negbin.visualize_learned_negative_binomial_model(mcmc)
fig.tight_layout()
plt.savefig('output_negbin/learned_model.pdf')
plt.close()

f = open('output_negbin/a0a1.txt', 'w')

mcmc.graph[STRNAMES.NEGBIN_A0].visualize(path='output_negbin1/a0.pdf', f=f, section='posterior')
mcmc.graph[STRNAMES.NEGBIN_A1].visualize(path='output_negbin1/a1.pdf', f=f, section='posterior')
mcmc.graph[STRNAMES.FILTERING].visualize(basepath='output_negbin1/', section='posterior')


sys.exit()



# Get the Gibson dataset and filter
# ---------------------------------

subjset = md2.dataset.gibson(dset='healthy')
subjset = md2.consistency_filtering(subjset, dtype='rel', threshold=0.0001,
    min_num_consecutive=5, colonization_time=5, min_num_subjects=2)

# Run the model
# -------------
params = md2.config.MDSINE2ModelConfig(basepath='output_real/', data_seed=0, init_seed=0,
    burnin=50, n_samples=150, negbin_a0=0.25, negbin_a1=0.0025, 
    qpcr_scale=0.3, checkpoint=50)

params.LEARN[STRNAMES.CLUSTERING] = False
params.LEARN[STRNAMES.CONCENTRATION] = False

mcmc = md2.build_graph(params=params, graph_name='healthy_cohort', 
    subjset=subjset)

# healthy = md2.aggregate_items(healthy, hamming_dist=2)
# uc = md2.aggregate_items(uc, hamming_dist=2)
# os.makedirs('pickles/', exist_ok=True)
# healthy.save('pickles/healthy_agg2.pkl')
# uc.save('pickles/uc_agg2.pkl')


study = md2.dataset.gibson()
poor_seqs = [
    'ASV_192', 'ASV_544', 'ASV_600', 'ASV_637', 'ASV_711', 'ASV_768', 'ASV_811',
    'ASV_938', 'ASV_946', 'ASV_997', 'ASV_998', 'ASV_1009', 'ASV_1160', 'ASV_1175',
    'ASV_1180', 'ASV_1202', 'ASV_1272', 'ASV_1362', 'ASV_1418', 'ASV_1423', 'ASV_1430',
    'ASV_1458', 'ASV_1465', 'ASV_1468']

Ms = [subj.matrix()['raw'] for subj in study]

for seq in poor_seqs:
    print('\n{}'.format(study.taxas[seq]))
    print(study.taxas[seq].sequence)
    aidx = study.taxas[seq].idx
    for M in Ms:
        print(M[aidx,:])
        