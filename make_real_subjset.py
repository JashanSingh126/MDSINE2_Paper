'''This is a script to read in the raw data in a subjectset data format

Species assignment is done by merging the assignments run on RDP and Silva:
    - We take a union of the species output from RDP and Silva

We save 2 `pylab.base.SubjectSet` objects:
    'real_subjectset.pkl'
        This has all of the data of the mice
    'inoculum_subjectset.pkl'
        This is the data of the innoculum samples. The two subjects are:
            'healthy'
            'ulcerative colitis'
'''
import sys
import copy
import pickle
import collections
import logging
import numpy as np
import csv
import pandas as pd
import os

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
import pylab as pl

logging.basicConfig(level=logging.DEBUG)

#################################################################################
# Constants
#################################################################################
BASEPATH = 'pickles/'
os.makedirs(BASEPATH, exist_ok=True)

FILENAME = BASEPATH + 'real_subjectset.pkl'
INOCULUM_FILENAME = BASEPATH + 'inoculum_subjectset.pkl'
RUN_INNOCULUM = False
STANDARD_CURVE_WELLS = 'hand_loaded_b_frag'
QPCR_DADA_DATA_FOLDER = 'raw_data/'
PHYLOGENETIC_TREE_FILENAME = 'raw_data/phylogenetic_tree.nhx'
PERTURBATIONS = [(21.5, 28.5, 'High fat diet'), (35.5, 42.5, 'Gram + ABX'), (50.5, 57.5, 'Gram - ABX')]
NEVER_ACCEPT = [ 
    '10d29am' # We have a repeat sample of this called  '10d29am repeat'. First sample is bad
]
IGNORE_SUBJECTS = ['1']

if STANDARD_CURVE_WELLS == 'hand_loaded_b_frag':
    STANDARD_CURVE_WELLS = [
        'B22','B23','B24','D22','D23','D24','F22','F23','F24',
        'H22','H23','H24','J22','J23','J24','L22','L23','L24']
elif STANDARD_CURVE_WELLS == 'ep_motion_b_frag':
    STANDARD_CURVE_WELLS = [
        'B19','B20','B21','D19','D20','D21','F19','F20','F21',
        'H19','H20','H21','J19','J20','J21','L19','L20','L21']
elif STANDARD_CURVE_WELLS == 'hand_loaded_s_scidens':
    STANDARD_CURVE_WELLS =[
        'A22','A23','A24','C22','C23','C24','E22','E23','E24',
        'G22','G23','G24','I22','I23','I24','K22','K23','K24']
elif STANDARD_CURVE_WELLS == 'ep_motion_s_scidens':
    STANDARD_CURVE_WELLS = [
        'A19','A20','A21','C19','C20','C21','E19','E20','E21',
        'G19','G20','G21','I19','I20','I21','K19','K20','K21']
else:
    raise ValueError('STANDARD_CURVE_WELLS Not Recognized')

# Paths for the files to read in for qPCR
CTSCORE_PATHS = [
    QPCR_DADA_DATA_FOLDER + 'qPCR/16s qPCR/1d0am-6d4pm/wellctscores.xls',
    QPCR_DADA_DATA_FOLDER + 'qPCR/16s qPCR/1d8-10d21pm/wellctscores.xls',
    QPCR_DADA_DATA_FOLDER + 'qPCR/16s qPCR/1d22am-7d28pm 7d4pm-3d6/wellctscores.xls',
    QPCR_DADA_DATA_FOLDER + 'qPCR/16s qPCR/2d42pm-10d50am/wellctscores.xls',
    QPCR_DADA_DATA_FOLDER + 'qPCR/16s qPCR/2D50PM-10D54 2D63-10D64PM/wellctscores.xls',
    QPCR_DADA_DATA_FOLDER + 'qPCR/16s qPCR/2d57am-10d62 inoculum/wellctscores.xls',
    QPCR_DADA_DATA_FOLDER + 'qPCR/16s qPCR/3d35pm-10d42am 4d6-10d7/wellctscores.xls',
    QPCR_DADA_DATA_FOLDER + 'qPCR/16s qPCR/8d28pm-2d35pm_2/wellctscores.xls']
WELLPLATETOSAMPLEID_PATHS = [
    QPCR_DADA_DATA_FOLDER + 'qPCR/16s qPCR/1d0am-6d4pm/well2sampleid.xlsx',
    QPCR_DADA_DATA_FOLDER + 'qPCR/16s qPCR/1d8-10d21pm/well2sampleid.xlsx',
    QPCR_DADA_DATA_FOLDER + 'qPCR/16s qPCR/1d22am-7d28pm 7d4pm-3d6/well2sampleid.xlsx',
    QPCR_DADA_DATA_FOLDER + 'qPCR/16s qPCR/2d42pm-10d50am/well2sampleid.xlsx',
    QPCR_DADA_DATA_FOLDER + 'qPCR/16s qPCR/2D50PM-10D54 2D63-10D64PM/well2sampleid.xlsx',
    QPCR_DADA_DATA_FOLDER + 'qPCR/16s qPCR/2d57am-10d62 inoculum/well2sampleid.xlsx',
    QPCR_DADA_DATA_FOLDER + 'qPCR/16s qPCR/3d35pm-10d42am 4d6-10d7/well2sampleid.xlsx',
    QPCR_DADA_DATA_FOLDER + 'qPCR/16s qPCR/8d28pm-2d35pm_2/well2sampleid.xlsx']
DILUTION_FACTORS = [100, 100, 100, 100, 100, 100, 100, 100]
EXCEPTION_DILUTION_FACTORS = {}

# Paths for the rerun files
CTSCORE_PATHS_RERUN = [
    QPCR_DADA_DATA_FOLDER + 'qPCR/16s qPCR/RERUN SAMPLES/M5D0AM- M6D57PM repeat plate 1/wellctscores.xls',
    QPCR_DADA_DATA_FOLDER + 'qPCR/16s qPCR/RERUN SAMPLES/M9D30AM-M2D35PM repeat plate 3/wellctscores.xls',
    QPCR_DADA_DATA_FOLDER + 'qPCR/16s qPCR/RERUN SAMPLES/M10D57PM- M8D30AM repeat plate 2/wellctscores.xls']
WELLPLATETOSAMPLEID_PATHS_RERUN = [
    QPCR_DADA_DATA_FOLDER + 'qPCR/16s qPCR/RERUN SAMPLES/M5D0AM- M6D57PM repeat plate 1/well2sampleid.xlsx',
    QPCR_DADA_DATA_FOLDER + 'qPCR/16s qPCR/RERUN SAMPLES/M9D30AM-M2D35PM repeat plate 3/well2sampleid.xlsx',
    QPCR_DADA_DATA_FOLDER + 'qPCR/16s qPCR/RERUN SAMPLES/M10D57PM- M8D30AM repeat plate 2/well2sampleid.xlsx']
DILUTION_FACTORS_RERUN = [1000, 1000, 1000]
EXCEPTION_DILUTION_FACTORS_RERUN = {('10', (0.0)): 10, ('5', 0.0): 10}

# Paths for the sample masses
SAMPLE_MASS_PATHS = [
    QPCR_DADA_DATA_FOLDER + 'qPCR/Sample Submission Sheets/Sample Submission Sheet D0-D7.xlsx',
    QPCR_DADA_DATA_FOLDER + 'qPCR/Sample Submission Sheets/Sample Submission Sheet D8-D21PM.xlsx',
    QPCR_DADA_DATA_FOLDER + 'qPCR/Sample Submission Sheets/Sample Submission Sheet D22AM-D29PM.xlsx',
    QPCR_DADA_DATA_FOLDER + 'qPCR/Sample Submission Sheets/Sample Submission Sheet D30AM-D36PM.xlsx',
    QPCR_DADA_DATA_FOLDER + 'qPCR/Sample Submission Sheets/Sample Submission Sheet D37AM-D44PM.xlsx',
    QPCR_DADA_DATA_FOLDER + 'qPCR/Sample Submission Sheets/Sample Submission Sheet D45AM-D52PM.xlsx',
    QPCR_DADA_DATA_FOLDER + 'qPCR/Sample Submission Sheets/Sample Submission Sheet D53-D60PM.xlsx',
    QPCR_DADA_DATA_FOLDER + 'qPCR/Sample Submission Sheets/Sample Submission Sheet D61-D64PM.xlsx']

# DADA stuff
DADA_PATH = QPCR_DADA_DATA_FOLDER + 'counts_16S/counts_pseudo_pooling.tsv'
TAXONOMY_RDP_PATH = QPCR_DADA_DATA_FOLDER + 'counts_16S/taxonomy_rdp.tsv'
TAXONOMY_SILVA_PATH = QPCR_DADA_DATA_FOLDER + 'counts_16S/taxonomy_silva.tsv'
DADA_TAXONOMY_COL_NAMES = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
HEALTHY_INOCULUM_COL = 'inoculum1-5'
ULCERATIVE_COLITIS_INOCULUM_COL = 'inoculum-6-10'
INOCULUM_COLS = [ULCERATIVE_COLITIS_INOCULUM_COL, HEALTHY_INOCULUM_COL]
SEQUENCE_COLS = ['sequence']
REPLICATE_DATA_COLS = [
    'M2-D10-1A','M2-D10-1B','M2-D10-2A','M2-D10-2B','M2-D10-3A','M2-D10-3B',	
    'M2-D8-1A',	'M2-D8-1B',	'M2-D8-2A',	'M2-D8-2B',	'M2-D8-3A',	'M2-D8-3B',	
    'M2-D9-1A',	'M2-D9-1B',	'M2-D9-2A',	'M2-D9-2B',	'M2-D9-3A',	'M2-D9-3B']

#################################################################################
# Functions
#################################################################################
def floatify_timepoint(a):
    return float(a.upper().replace('PM','.5').replace('AM','.0'))

def get_qpcr(qpcr, masses, ctscore_path, wellplatetosampleid_path,
    dilution_factor, exception_dilution_factor, plot_standard_curve=False):
    '''Get the qpcr data

    Parameters
    ----------
    qpcr : dict
        This is the dictionary of pylab.base.qPCRdata objects we are adding to
    masses : dict
        These are the samples that we are using to make the qPCR objects
    ctscore_path, wellplatetosampleid_path : str
        These are the paths for the CT score and the well plate to sample id
        Excel files
    dilution_factor : numeric
        This is the default dilution factor
    exception_dilution_factor : dict
        These map the (subject name, timepoint) to a different dilution factor 
        if it has one  that is different than the defualt
    plt_standard_curve : bool
        If True, plot the standard curve
    '''
    # Read in the qpcr data
    # subject name -> timepoint -> list of measurements
    standardCTscores = {}
    sampleCTscores = {}

    ctscore = pd.read_excel(
        ctscore_path, 
        sheet_name='Results', 
        header=32, 
        index_col='Well Position')
    ctscore = ctscore['CT']
    wellplatetosample = pd.read_excel(wellplatetosampleid_path, index_col=0)

    # from the well position in `ctscore`, get the subject and sample id 
    # from `wellplatetosample`
    for row in ctscore.index:
        if pl.isstr(ctscore[row]):
            logging.debug('well `{}` has an undetermined CT score ({})'.format(
                row, ctscore[row]))
            continue

        # Get the sample id
        _row = row[0]
        _col = int(row[1:])
        ss = wellplatetosample.loc[_row, _col]
        if ss == 'NTC' or pl.isnumeric(ss):
            continue
        if ss in NEVER_ACCEPT:
            continue
        ss = ss.upper().split('D')

        if len(ss) == 1:
            # This means this is one of our wells for the standard curve
            if row in STANDARD_CURVE_WELLS:
                # `ss` is something like ['BF9'], and we want to get the 9
                abnd = int(ss[0][2])
                if abnd not in standardCTscores:
                    standardCTscores[abnd] = []
                standardCTscores[abnd].append(ctscore[row])
        else:
            # Else this is a regular sample
            # Check if this is a repeat sample
            if ss[0] in IGNORE_SUBJECTS:
                continue

            if 'REPEAT' in ss[1]:
                ss[1] = ss[1].replace('REPEAT', '').replace(' ', '')
            
            timepoint = floatify_timepoint(ss[1])
            if ss[0] not in sampleCTscores:
                sampleCTscores[ss[0]] = {}
            if timepoint not in sampleCTscores[ss[0]]:
                sampleCTscores[ss[0]][timepoint] = []
            sampleCTscores[ss[0]][timepoint].append(ctscore[row])

    # Now we have the data and the standard data to calculate CFUs from
    # the CT score
    X = []
    y = []
    for key, val in standardCTscores.items():
        y = np.append(y, np.full(len(val), float(key)))
        X = np.append(X, val)
    X = X.reshape(-1,1)
    y = y.reshape(-1,1)
    model = LinearRegression().fit(X,y)
    m = np.squeeze(model.coef_)
    b = np.squeeze(model.intercept_)
    r2 = model.score(X,y)

    logging.info('R^2 Fit: {}'.format(r2))
    if plot_standard_curve:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.scatter(X,y,alpha = 0.6, rasterized=True)
        ax.set_xlabel('CT score')
        ax.set_ylabel('log10 CFUs')
        ax.set_title('Standard Curve')
        x_ = np.arange(np.min(X),np.max(X),step=0.01)
        y_ = m * x_ + b
        ax.plot(x_,y_,color='red')
        logging.info('intercept: {}, slope: {}, r2: {}'.format(b,m,r2))
        plt.show()

    # Make the qpcr data
    for mid in sampleCTscores:
        if mid not in qpcr:
            qpcr[mid] = {}
        for sid in sampleCTscores[mid]:
            if sid not in masses[mid]:
                logging.warning('skipping over {}, {} because there is no mass '.format(
                    mid,sid))
                continue
            mass = masses[mid][sid]
            if (mid, sid) in exception_dilution_factor:
                dfactor = exception_dilution_factor[(mid,sid)]
            else:
                dfactor = dilution_factor
            cfus = [10 **(m*x + b) for x in sampleCTscores[mid][sid]]
            
            if sid in qpcr[mid]:
                logging.debug('OVERRIDING {}, {}'.format(mid, sid))
            qpcr[mid][sid] = pl.base.qPCRdata(
                cfus=cfus, mass=mass, dilution_factor=dfactor)

    return qpcr

#################################################################################
# Set the taxonomies of the ASVs
#################################################################################
taxonomy_rdp = pd.read_csv(TAXONOMY_RDP_PATH,sep='\t')
taxonomy_rdp = taxonomy_rdp.set_index('otuName')

taxonomy_silva = pd.read_csv(TAXONOMY_SILVA_PATH,sep='\t')
taxonomy_silva = taxonomy_silva.set_index('otuName')

asvs_rdp = pl.ASVSet(df=taxonomy_rdp)
asvs_silva = pl.ASVSet(df=taxonomy_silva)

for name in range(len(asvs_rdp)):
    asv_sil = asvs_silva[name]
    asv_rdp = asvs_rdp[name]

    if type(asv_rdp.taxonomy['species']) == float:
        if type(asv_sil.taxonomy['species']) != float:
            asv_rdp.taxonomy['species'] = asv_sil.taxonomy['species']
        # Else both nan
    else:
        if type(asv_sil.taxonomy['species']) != float:
            # Union
            spec_rdp = asv_rdp.taxonomy['species'].split('/')
            spec_sil = asv_sil.taxonomy['species'].split('/')

            spec_rdp = set(spec_rdp + spec_sil)
            aaa = None
            for i, spec in enumerate(spec_rdp):
                if i == 0:
                    aaa = spec
                else:
                    aaa += '/{}'.format(spec)
            spec_rdp = aaa
            asv_rdp.taxonomy['species'] = spec_rdp
        # Else we do just rdp, which is already set

asvs = asvs_rdp

# Only keep ASV species identification if there are 2 or less options
for asv in asvs:
    if asv.tax_is_defined('species'):
        species = asv.taxonomy['species'].split('/')
        if len(species) >= 3:
            # Too long
            asv.taxonomy['species'] = pl.base.DEFAULT_TAXA_NAME


#################################################################################
# Read the relvative abudnance (reads) and the ASVs, set the perturbations
#################################################################################
reads_base = pd.read_csv(DADA_PATH,sep='\t')
reads_base = reads_base.set_index('otuName')

subjset = pl.SubjectSet(asvs=asvs)
subjset_inoculum = pl.SubjectSet(asvs=asvs)
reads = reads_base.drop(columns=INOCULUM_COLS+SEQUENCE_COLS)
reads_inoculum = reads_base[INOCULUM_COLS]

for col in reads.columns:
    ss = col.split('-')
    if ss[0] in IGNORE_SUBJECTS:
        continue
    if col in REPLICATE_DATA_COLS:
        continue
    subjset.add(ss[0])

subjset_inoculum.add('healthy')
subjset_inoculum.add('ulcerative colitis')

# Get the reads for each inoculum sample
subj_healthy = subjset_inoculum['healthy']
subj_healthy.times = np.asarray([0], dtype=float)
subj_healthy.reads[0] = np.asarray(reads_inoculum[HEALTHY_INOCULUM_COL].to_numpy().ravel(), dtype=int)

subj_uc = subjset_inoculum['ulcerative colitis']
subj_uc.times = np.asarray([0], dtype=float)
subj_uc.reads[0] = np.asarray(reads_inoculum[ULCERATIVE_COLITIS_INOCULUM_COL].to_numpy().ravel(), dtype=int)


# Get the reads for each subject
for subj in subjset:
    name = subj.name

    # First get the columns for the subject and then add them in order
    timepoints = []
    cols = []
    for col in reads.columns:
        ss = col.split('-')
        if ss[0] != name:
            continue
        timepoint = floatify_timepoint(ss[1][1:])
        timepoints.append(timepoint)
        cols.append(col)

    idxs = np.argsort(timepoints)
    timepoints = np.asarray(timepoints)[idxs]
    cols = np.asarray(cols)[idxs]

    subj.times = timepoints
    for idx, timepoint in enumerate(timepoints):
        subj.reads[timepoint] = np.asarray(list(reads[cols[idx]]), dtype=int)

for start,end,name in PERTURBATIONS:
    subjset.add_perturbation(start, end, name=name)

#################################################################################
# Read in the qPCR data
#################################################################################
# First read in the masses
# subject name -> timepoint -> mass (g)

qpcr_masses = {}
for filename in SAMPLE_MASS_PATHS:
    df = pd.read_excel(filename, 0, header=32, index_col='SubjectID')
    df = df['Mass of Stool (g)']
    for row in df.index:
        if not pl.isstr(row):
            # This means it is a nan and we can skip
            continue
        ss = row.upper()
        ss = ss.split('D')
        if ss[0] not in qpcr_masses:
            if ss[0] in IGNORE_SUBJECTS:
                continue
            qpcr_masses[ss[0]] = {}
        timepoint = floatify_timepoint(ss[1])
        qpcr_masses[ss[0]][timepoint] = df[row]

# Throw out any samples that have a nan mass
to_delete = []
for mid in qpcr_masses:
    for tid in qpcr_masses[mid]:
        if np.isnan(qpcr_masses[mid][tid]):
            logging.warning('{}, {} has a NaN mass. Throwing out'.format(mid,tid))
            to_delete.append((mid,tid))
for mid,tid in to_delete:
    qpcr_masses[mid].pop(tid, None)

# Make the qPCR objects
qpcr = {}
for i in range(len(CTSCORE_PATHS)):
    qpcr = get_qpcr(
        qpcr=qpcr,
        masses=qpcr_masses,
        ctscore_path=CTSCORE_PATHS[i],
        wellplatetosampleid_path=WELLPLATETOSAMPLEID_PATHS[i],
        dilution_factor=DILUTION_FACTORS[i],
        exception_dilution_factor=EXCEPTION_DILUTION_FACTORS)
for i in range(len(CTSCORE_PATHS_RERUN)):
    qpcr = get_qpcr(
        qpcr=qpcr,
        masses=qpcr_masses,
        ctscore_path=CTSCORE_PATHS_RERUN[i],
        wellplatetosampleid_path=WELLPLATETOSAMPLEID_PATHS_RERUN[i],
        dilution_factor=DILUTION_FACTORS_RERUN[i],
        exception_dilution_factor=EXCEPTION_DILUTION_FACTORS_RERUN)

# Set the qpcr data into the subjectset
for subj in subjset:
    for t in subj.times:
        subj.qpcr[t] = qpcr[subj.name][t]

logging.info('Did not record qPCR measurements for the inoculum subjectset')

# Rename the perturbations
# ------------------------
for perturbation in subjset.perturbations:
    if perturbation.name == 'Gram + ABX':
        perturbation.name = 'Vancomycin'
    if perturbation.name == 'Gram - ABX':
        perturbation.name = 'Gentamicin'

# Rename the OTUs to ASVs (start at 1, replace OTU with ASV)
# ----------------------------------------------------------
for asv in subjset_inoculum.asvs:
    asvname = asv.name
    oldname = asvname
    # print(asvname)
    n = int(asvname.replace('OTU_', ''))
    asv.name = 'ASV_{}'.format(n+1)
    newname = asv.name

    subjset_inoculum.asvs.names.pop(oldname)
    subjset_inoculum.asvs.names[newname] = asv
    subjset_inoculum.asvs.names.update_order()


# # Set the phylogenetic tree
# subjset_inoculum.asvs.set_phylogenetic_tree(PHYLOGENETIC_TREE_FILENAME)
# subjset.asvs.set_phylogenetic_tree(PHYLOGENETIC_TREE_FILENAME)

subjset_inoculum.save(INOCULUM_FILENAME)
subjset.save(FILENAME)