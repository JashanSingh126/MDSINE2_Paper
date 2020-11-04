import numpy as np
import pandas as pd
import logging
import sys
import scipy.stats
import scipy.sparse
import scipy.spatial
import numba
import time
import collections
import h5py

import names
import pylab as pl
import model

def parse_mdsine1_cdiff(basepath):
    '''Biomass is in `biomass.txt`
    counts are in `counts.txt`
    metadata are in `metadata.txt`

    Parameters
    ----------
    basepath : str
        Path to the folder holding all of the information
    
    Returns
    -------
    pl.base.SubjectSet
    '''
    biomass = basepath + 'biomass.txt'
    dfbiomass = pd.read_csv(biomass, sep='\t')
    rows = ['{}'.format(i) for i in range(1, dfbiomass.shape[0]+1)]
    dfbiomass = pd.DataFrame(dfbiomass.values, index=rows, columns=dfbiomass.columns)

    counts = basepath + 'counts.txt'
    dfcounts = pd.read_csv(counts, sep='\t')
    dfcounts = dfcounts.set_index('#OTU ID')
    # print(dfcounts.head())

    metadata = basepath + 'metadata.txt'
    dfmeta = pd.read_csv(metadata, sep='\t')
    dfmeta = pd.DataFrame(dfmeta.values, index=rows, columns=dfmeta.columns)
    # print(dfmeta)

    asvs = pl.base.ASVSet(use_sequences=False)
    for row in dfcounts.index:
        asvs.add_asv(row)
    subjset = pl.base.SubjectSet(asvs=asvs)


    for ridx, row in enumerate(dfmeta.index):
        meta = dfmeta.loc[row]
        if not meta['isIncluded']:
            continue

        sid = str(int(meta['subjectID']))
        t = float(meta['measurementid'])

        if sid not in subjset:
            subjset.add(name=sid)
        
        subj = subjset[sid]
        subj.add_reads(timepoints=t, reads=np.asarray(list(dfcounts[row])))
        subj.add_qpcr(timepoints=t, qpcr=np.asarray(list(dfbiomass.loc[row])))
    return subjset

def parse_mdsine1_diet(basepath):
    '''There are no qPCR measurements here
    '''
    counts = basepath + 'counts.txt'
    dfcounts = pd.read_csv(counts, sep='\t')
    dfcounts = dfcounts.set_index('#OTU ID')

    metadata = basepath + 'metadata.txt'
    dfmeta = pd.read_csv(metadata, sep='\t')
    dfmeta = dfmeta.set_index('sampleID')


    asvs = pl.base.ASVSet(use_sequences=False)
    for row in dfcounts.index:
        asvs.add_asv(row)
    subjset = pl.base.SubjectSet(asvs=asvs)

    pert_start = None
    pert_end = None
    cont_pert = False
    for idx, row in enumerate(dfmeta.index):
        meta = dfmeta.loc[row]
        if meta['isIncluded']:

            sid = str(int(meta['subjectID']))
            perturbID = int(meta['perturbID'])
            t = float(meta['measurementID'])

            if sid not in subjset:
                subjset.add(name=sid)
            
            subj = subjset[sid]
            subj.add_reads(timepoints=t, reads=np.asarray(list(dfcounts[str(row)])))
            subj.add_qpcr(timepoints=t, qpcr=np.asarray([1]))

            # Add the perturbation if it is included
            if perturbID != 0:
                if not cont_pert:
                    cont_pert = True
                    pert_start = t
                    pert_end = t
                else:
                    pert_end = t
            else:
                if cont_pert:
                    # Finished looking at the perturbation
                    cont_pert = False
                    if subjset.perturbations is not None:
                        for pert in subjset.perturbations:
                            if pert_start == pert.start and pert_end == pert.end:
                                # dont need to add it because it is the same perturbations
                                pass
                            else:
                                subjset.add_perturbation(pert_start, end=pert_end)
                    else:
                        subjset.add_perturbation(pert_start, end=pert_end)
            

    return subjset

def make_data_like_mdsine1(subjset, basepath):
    '''Parse subjset into data format like mdsine1
    '''
    os.makedirs(basepath, exist_ok=True)

    n_times = 0 # for `#OTU ID`
    for subj in subjset:
        n_times += len(subj.times)

    metadata_columns = ['sampleID', 'isIncluded', 'subjectID', 'measurementid', 'perturbid']
    

    # Make the counts
    counts_columns = ['{}'.format(i+1) for i in range(n_times)]
    count_Ms = []
    for subj in subjset:
        d = subj.matrix()
        M = d['raw']
        count_Ms.append(M)
    
    counts = np.hstack(count_Ms)
    df_counts = pd.DataFrame(counts, columns=counts_columns, index=[asv.name for asv in subjset.asvs])
    df_counts.index = df_counts.index.rename('#OTU ID')
    df_counts.to_csv(basepath + 'counts.txt', sep='\t', header=True, index=True)

    # Make the biomass
    biomass_columns = ['mass1', 'mass2', 'mass3']
    qs = []
    for subj in subjset:
        for t in subj.times:
            qs.append(subj.qpcr[t].data)
    biomass = np.vstack(qs)
    df_biomass = pd.DataFrame(data=biomass, columns=biomass_columns, index=counts_columns)
    df_biomass.to_csv(basepath + 'biomass.txt', sep='\t', header=True, index=False, float_format='%.2E')
    

    # Make metadata
    sampleID = counts_columns
    is_included = [1 for i in range(len(sampleID))]
    subjectID = []
    measurementid = []
    perturbid = []
    for subjidx, subj in enumerate(subjset):
        for t in subj.times:
            subjectID.append(int(subj.name))
            measurementid.append(t)

            p = 0
            for pidx, pert in enumerate(subjset.perturbations):
                if t >= pert.start and t <= pert.end:
                    p = pidx+1
                    break
            perturbid.append(p)

    print(perturbid)

    df_metadata = pd.DataFrame({
        'sampleID': sampleID, 
        'isIncluded': is_included, 
        'subjectID': subjectID,
        'measurementid': measurementid,
        'perturbid': perturbid})
    df_metadata.to_csv(basepath + 'metadata.txt', sep='\t', header=True, index=False)

def asvname_for_paper(asv, asvs):
    '''Makes the name in the format needed for the paper

    Parameters
    ----------
    asv : pylab.base.ASV
        This is the ASV we are making the name for
    asvs : pylab.base.ASVSet
        This is the ASVSet object that contains the ASV

    Returns
    -------
    '''
    if asv.tax_is_defined('species'):
        species = asv.taxonomy['species']
        species = species.split('/')
        if len(species) >= 3:
            species = species[:2]
        species = '/'.join(species)
        label = pl.asvname_formatter(
            format='%(genus)s {spec} %(name)s'.format(
                spec=species), 
            asv=asv, asvs=asvs)
    elif asv.tax_is_defined('genus'):
        label = pl.asvname_formatter(
            format='* %(genus)s %(name)s',
            asv=asv, asvs=asvs)
    elif asv.tax_is_defined('family'):
        label = pl.asvname_formatter(
            format='** %(family)s %(name)s',
            asv=asv, asvs=asvs)
    elif asv.tax_is_defined('order'):
        label = pl.asvname_formatter(
            format='*** %(order)s %(name)s',
            asv=asv, asvs=asvs)
    elif asv.tax_is_defined('class'):
        label = pl.asvname_formatter(
            format='**** %(class)s %(name)s',
            asv=asv, asvs=asvs)
    elif asv.tax_is_defined('phylum'):
        label = pl.asvname_formatter(
            format='***** %(phylum)s %(name)s',
            asv=asv, asvs=asvs)
    elif asv.tax_is_defined('kingdom'):
        label = pl.asvname_formatter(
            format='****** %(kingdom)s %(name)s',
            asv=asv, asvs=asvs)
    else:
        raise ValueError('Something went wrong - no taxnonomy: {}'.format(str(asv)))

    return label

def is_gram_negative(asv):
    '''Return true if the asv is gram - or gram positive
    '''
    if not asv.tax_is_defined('phylum'):
        return None
    elif asv.taxonomy['phylum'].lower() == 'bacteroidetes':
        return True
    elif asv.taxonomy['phylum'].lower() == 'firmicutes':
        return False
    elif asv.taxonomy['phylum'].lower() == 'verrucomicrobia':
        return True
    elif asv.taxonomy['phylum'].lower() == 'proteobacteria':
        return True
    else:
        raise ValueError('{} phylum not specified. If not bacteroidetes, firmicutes, verrucomicrobia, or ' \
            'proteobacteria, you must add another phylum'.format(str(asv)))


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