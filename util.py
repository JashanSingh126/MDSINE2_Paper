import numpy as np
import pandas as pd

import pylab as pl

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
    
