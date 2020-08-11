import logging
import random
import time
import sys
import os
import shutil
import h5py
import warnings
import pickle
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import data
import pylab as pl
import config
import synthetic
import model as model_module
import preprocess_filtering as filtering
import metrics
from names import STRNAMES, LATEXNAMES, REPRNAMES

ASV_FMT = '%(name)s %(genus)s %(species)s'
TAXONOMIES = ['phylum', 'class', 'order', 'family', 'genus', 'species', 'asv']

def is_gram_negative(asv):
    '''Return true if the asv is gram - or gram positive
    '''
    if not asv.tax_is_defined('phylum'):
        return None
    if asv.taxonomy['phylum'].lower() == 'bacteroidetes':
        return True
    if asv.taxonomy['phylum'].lower() == 'firmicutes':
        return False
    if asv.taxonomy['phylum'].lower() == 'verrucomicrobia':
        return True
    if asv.taxonomy['phylum'].lower() != 'proteobacteria':
        print(asv)
        print('Not included')
        return None

    # Deltaproteobacteria are all gram -
    return True

def is_gram_negative_taxa(taxa, taxalevel, asvs):
    '''Checks if the taxa `taxa` at the taxonomic level `taxalevel`
    is a gram negative or gram positive
    '''
    for asv in asvs:
        if asv.taxonomy[taxalevel] == taxa:
            return is_gram_negative(asv)

    else:
        raise ValueError('`taxa` ({}) not found at taxonomic level ({})'.format(
            taxa. taxalevel))

def analyze_clusters_df(chain, taxlevel, include_nan=False, prop_total=True):
    '''Do analysis on the clusters and return the results as a dataframe
    '''
    asvs = chain.graph.data.asvs
    clustering = chain.graph[STRNAMES.CLUSTERING_OBJ]
    # clustering.generate_cluster_assignments_posthoc(n_clusters='mean', set_as_value=True)

    # # Order the clusters from largest to smallest
    # cids = []
    # cids_sizes = []
    # for cluster in clustering:
    #     cids.append(cluster.id)
    #     cids_sizes.append(len(cluster))

    # idxs = np.argsort(cids_sizes)
    # idxs = idxs[::-1]
    # cids = np.asarray(cids)
    # cids = cids[idxs]

    s = {}
    for asv in chain.graph.data.subjects.asvs:
        if asv.tax_is_defined(taxlevel):
            tax = asv.taxonomy[taxlevel]
        else:
            if include_nan:
                tax = 'NA'
            else:
                continue
        if tax in s:
            s[tax] += 1
        else:
            s[tax] = 1

    columns = list(s.keys())
    tax2taxidx = {}
    for i,v in enumerate(columns):
        tax2taxidx[v] = i

    M = np.zeros(shape=(len(clustering), len(columns)))
    index = clustering.order


    for cidx, cid in enumerate(clustering.order):
        cluster = clustering[cid]

        # print('\nCluster {}'.format(cidx))
        # print(cluster.members)
        # print(len(cluster))


        taxas_each_asv = {}
        # gram_each_asv = {}
        for aidx in cluster.members:
            asv = asvs[aidx]
            asv_taxa = asv.taxonomy[taxlevel]
            if not asv.tax_is_defined(taxlevel):
                if include_nan:
                    asv_taxa = 'NA'
                else:
                    continue
            if asv_taxa not in taxas_each_asv:
                taxas_each_asv[asv_taxa] = 0
            taxas_each_asv[asv_taxa] += 1

            # gram_status = is_gram_negative(asv)
            # if gram_status not in gram_each_asv:
            #     gram_each_asv[gram_status] = 0
            # gram_each_asv[gram_status] += 1

        for taxa in taxas_each_asv:
            if prop_total is None:
                M[cidx, tax2taxidx[taxa]] = taxas_each_asv[taxa]

            else:
                if prop_total:
                    M[cidx, tax2taxidx[taxa]] = taxas_each_asv[taxa] / s[taxa]
                else:
                    M[cidx, tax2taxidx[taxa]] = taxas_each_asv[taxa] / len(cluster)

    df = pd.DataFrame(M, columns=columns, index=index)
    return df

def analyze_clusters(chain, basepath,
    yticklabels='%(genus)s %(species)s %(index)s',
    xticklabels='%(index)s'):
    '''Do analysis on the clusters
    '''
    if basepath[-1] != '/':
        basepath += '/'
    os.makedirs(basepath, exist_ok=True)

    asvs = chain.graph.data.asvs
    clustering = chain.graph[STRNAMES.CLUSTERING_OBJ]
    clustering.generate_cluster_assignments_posthoc(n_clusters='mean', set_as_value=True)

    f = open(basepath + 'analyze_clusters.txt', 'w')
    f.write('Cluster analysis\n')
    f.write('================\n\n')
    f.write('{}\n'.format(chain.graph.name))

    # # plot the cocluster trace and the number of clusters
    # print('Plot coclusters')
    # cluster_trace = clustering.coclusters.get_trace_from_disk()
    # coclusters = pl.variables.summary(cluster_trace)['mean']
    # for i in range(coclusters.shape[0]):
    #     coclusters[i,i] = np.nan
    # pl.visualization.render_cocluster_proportions(
    #     coclusters=coclusters, asvs=asvs, clustering=clustering,
    #     yticklabels=yticklabels, include_tick_marks=False, xticklabels=xticklabels,
    #     title='Cluster Assignments, {}'.format(LATEXNAMES.CLUSTERING))
    # plt.savefig(basepath + 'coclusters.pdf')
    # plt.close()

    # print('Plot n clusters')
    # pl.visualization.render_trace(var=clustering.n_clusters, plt_type='both', 
    #     section='posterior', include_burnin=False, rasterized=True)
    # fig = plt.gcf()
    # fig.suptitle('Number of Clusters')
    # plt.savefig(basepath + 'n_clusters.pdf')
    # plt.close()

    n_clusters = clustering.n_clusters
    f.write('Number of clusters\n')
    summ = pl.variables.summary(n_clusters)
    for key,val in summ.items():
        f.write('\t{}: {}\n'.format(key,val))

    f.write('\n\n-----------------\n')
    f.write('Individual cluster output\n')
    f.write('-----------------\n')

    # Order the clusters from largest to smallest
    cids = []
    cids_sizes = []
    for cluster in clustering:
        cids.append(cluster.id)
        cids_sizes.append(len(cluster))

    idxs = np.argsort(cids_sizes)
    idxs = idxs[::-1]
    cids = np.asarray(cids)
    cids = cids[idxs]

    cluster_coherence_max = {}
    gram_coherence_max = []

    for cidx, cid in enumerate(cids):
        cluster = clustering[cid]
        f.write('Cluster {} - {} ASVs\n'.format(cidx, len(cluster)))
        f.write('======================\n')
        f.write('ASVs\n')
        f.write('----\n')
        for aidx in cluster:
            asv = asvs[aidx]
            f.write('\t{}\n'.format(pl.asvname_formatter(format=ASV_FMT, asv=asv, asvs=asvs)))

        f.write('Taxonomic coherence\n')
        # Dont want to include the species
        for taxa in TAXONOMIES[:-2]:
            f.write('\t{}\n'.format(taxa.capitalize()))

            taxas_each_asv = {}
            gram_each_asv = {}
            for aidx in cluster.members:
                asv = asvs[aidx]
                asv_taxa = asv.taxonomy[taxa]
                if not asv.tax_is_defined(taxa):
                    asv_taxa = 'NA'
                if asv_taxa not in taxas_each_asv:
                    taxas_each_asv[asv_taxa] = 0
                taxas_each_asv[asv_taxa] += 1

                gram_status = is_gram_negative(asv)
                if gram_status not in gram_each_asv:
                    gram_each_asv[gram_status] = 0
                gram_each_asv[gram_status] += 1

            for key in taxas_each_asv:
                taxas_each_asv[key] /= len(cluster)

            keys = np.asarray(list(taxas_each_asv.keys()))
            idxs = np.argsort(np.asarray(list(taxas_each_asv.values())))[::-1]

            for iiii, idx in enumerate(idxs):
                key = keys[idx]
                val = taxas_each_asv[key]

                if iiii == 0:
                    if len(cluster) > 1:
                        if taxa not in cluster_coherence_max:
                            cluster_coherence_max[taxa] = []
                        cluster_coherence_max[taxa].append([val, len(cluster)])
                
                f.write('\t\t{:.3f}: {}\n'.format(val, key))

        if len(cluster) > 1:
            f.write('Gram positive/negative coherence\n')
            gcm = -1
            for k,v in gram_each_asv.items():
                v = v/len(cluster)
                f.write('\t{}: {}\n'.format(k,v))
                if v > gcm:
                    gcm = v
            gram_coherence_max.append(gcm)
                

        f.write('\n\n')
    f.write('Max coherence stats (clusters > 1):\n')
    for taxa in cluster_coherence_max:
        f.write('\t{}\n'.format(taxa.capitalize()))
        m = np.asarray(cluster_coherence_max[taxa])
        print(m)
        a = pl.variables.summary(m[:,0])
        for k,v in a.items():
            f.write('\t\t{}: {:.3f}\n'.format(k,v))

    f.write('Gram coherence stats (clusters > 1)\n')
    a = pl.variables.summary(gram_coherence_max)
    for k,v in a.items():
        f.write('\t\t{}: {:.3f}\n'.format(k,v))

    f.close()

def analyze_coclusters(chain1, chain2, basepath,
    yticklabels='%(name)s %(index)s',
    xticklabels='%(index)s'):
    '''See if the the common asvs between chain1 and chain2 cocluster the same way
    '''
    # f = open(basepath + 'common_asvs_coclustering.txt', 'w')

    # Get the coclusterings of each set
    asv1_set = set()
    asvs1 = chain1.graph.data.asvs
    clustering1 = chain1.graph[STRNAMES.CLUSTERING_OBJ]
    coclusters1 = pl.variables.summary(clustering1.coclusters.get_trace_from_disk())['mean']
    for asv in chain1.graph.data.asvs:
        asv1_set.add(asvs1[asv].name)

    asv2_set = set()
    asvs2 = chain2.graph.data.asvs
    clustering2 = chain2.graph[STRNAMES.CLUSTERING_OBJ]
    coclusters2 = pl.variables.summary(clustering2.coclusters.get_trace_from_disk())['mean']
    for asv in chain2.graph.data.asvs:
        asv2_set.add(asvs2[asv].name)

    common_asvs = asv1_set.intersection(asv2_set)
    common_asvs = list(common_asvs)
    # Index to only the common asvs for each set
    idxs1 = []
    idxs2 = []
    for asv in common_asvs:
        idxs1.append(asvs1[asv].idx)
        idxs2.append(asvs2[asv].idx)

    coclusters1 = coclusters1[idxs1,:]
    coclusters1 = coclusters1[:, idxs1]
    coclusters2 = coclusters2[:, idxs2]
    coclusters2 = coclusters2[idxs2, :]

    for i in range(coclusters2.shape[0]):
        coclusters1[i,i] = np.nan
        coclusters2[i,i] = np.nan


    # pl.visualization.render_cocluster_proportions(
    #     coclusters=coclusters1, asvs=asvs1, clustering=clustering1,
    #     yticklabels=yticklabels, include_tick_marks=False, xticklabels=xticklabels,
    #     title='Cluster Assignments1')
    plt.figure()
    sns.heatmap(coclusters1, cmap='Blues')
    plt.savefig(basepath + 'coclusters1.pdf')
    plt.close()

    # pl.visualization.render_cocluster_proportions(
    #     coclusters=coclusters2, asvs=asvs2, clustering=clustering2,
    #     yticklabels=yticklabels, include_tick_marks=False, xticklabels=xticklabels,
    #     title='Cluster Assignments2')
    plt.figure()
    sns.heatmap(coclusters2, cmap='Blues')
    plt.savefig(basepath + 'coclusters2.pdf')
    plt.close()

    # plt.show()

def abundance(chain):

    d = []
    subjset = chain.graph.data.subjects

    subjset.denormalize_qpcr()

    print(len(subjset))

    for subj in subjset:
        # print(subj.matrix()['abs'])
        data = subj.matrix()['abs']
        d = np.append(d, data)

    print('percent 0', len(d[d == 0])/len(d))
    d = d[d > 0]

    a = pl.variables.summary(d)
    for k,v in a.items():
        print('{}: {:.2E}'.format(k,v))

    print('gmean {:.3E}'.format(np.exp(np.sum(np.log(d))/len(d) )))



    d = d.reshape(-1,1)
    d = pd.DataFrame(d, columns=['A'])

    plt.figure()
    ax = sns.boxplot(data=d, y='A')
    ax.set_yscale('log')
    plt.savefig('abundance_boxplot{}.pdf'.format(len(chain.graph.data.data)))
    plt.close()

    print('\n\n')

# chainpaths = [
#     'output_real/pylab24/real_runs/perts_mult/healthy1_5_0.0001_rel_2_5/ds0_is0_b5000_ns20000_mo-1_logTrue_pertsmult/graph_leave_out-1/'
# ]

# chains = []
# for chainpath in chainpaths:
#     chains.append(pl.inference.BaseMCMC.load(chainpath + 'mcmc.pkl'))

# # analyze_coclusters(chains[0], chains[1], './')

# for chainpath in chainpaths:
#     chain = pl.inference.BaseMCMC.load(chainpath + 'mcmc.pkl')
#     analyze_clusters(chain=chain, basepath='./cluster_analysis/{}_'.format(len(chain.graph.data.data)))
#     # abundance(chain)








