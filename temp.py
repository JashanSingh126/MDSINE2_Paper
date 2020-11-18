import sys
import logging
import os
import pandas as pd

import mdsine2 as md2
from mdsine2.names import STRNAMES



healthy = md2.dataset.gibson(dset='healthy')
uc = md2.dataset.gibson(dset='uc')
inoc = md2.dataset.gibson(dset='inoculum')



# Before
subjset = pl.base.SubjectSet(...)
subjset = pl.base.SubjectSet.load(fname)

# Now
subjset = md2.Study(...)
subjset = md2.Study.load(fname)




# Get the Gibson dataset and filter
# ---------------------------------
md2.config.LoggingConfig()

# subjset = md2.dataset.gibson(dset='healthy')
# subjset = md2.consistency_filtering(subjset, dtype='rel', threshold=0.0001,
#     min_num_consecutive=5, colonization_time=5, min_num_subjects=2)

# # Run the model
# # -------------
# params = md2.config.MDSINE2ModelConfig(basepath='output_real/', data_seed=0, init_seed=0,
#     burnin=50, n_samples=150, negbin_a0=0.25, negbin_a1=0.0025, 
#     qpcr_scale=0.3, checkpoint=50)

# params.LEARN[STRNAMES.CLUSTERING] = False
# params.LEARN[STRNAMES.CONCENTRATION] = False

# mcmc = md2.build_graph(params=params, graph_name='healthy_cohort', 
#     subjset=subjset)

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
    print('\n{}'.format(study.asvs[seq]))
    print(study.asvs[seq].sequence)
    aidx = study.asvs[seq].idx
    for M in Ms:
        print(M[aidx,:])
        