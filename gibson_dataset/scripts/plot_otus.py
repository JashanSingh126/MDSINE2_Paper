'''Plot the OTU abundances for each subject with the inner ASVs
'''

import mdsine2 as md2
import matplotlib.pyplot as plt
import os
import argparse
import logging

if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage=__doc__)
    parser.add_argument('--output-basepath', '-o', type=str, dest='basepath',
        help='This is where you want to save the parsed dataset.')
    parser.add_argument('--study', '-s', type=str, dest='study',
        help='Dataset that contains all of the information')
    parser.add_argument('--top', '-t', type=int, dest='top',
        help='Plot up to this number', default=None)
    args = parser.parse_args()

    basepath = args.basepath
    os.makedirs(basepath, exist_ok=True)
    study = md2.Study.load(args.study)
    basepath = os.path.join(basepath, study.name)
    os.makedirs(basepath, exist_ok=True)

    md2.LoggingConfig(level=logging.INFO)

    if args.top is None:
        top = len(study.taxas)
    else:
        top = args.top

    for subj in study:
        subjpath = os.path.join(basepath, 'Subject {}'.format(subj.name))
        os.makedirs(subjpath, exist_ok=True)
        logging.info('Subject {}'.format(subj.name))
        for iii, taxa in enumerate(study.taxas):
            if not md2.isotu(taxa):
                continue
            if iii >= top:
                break
            logging.info('taxa {}/{}'.format(taxa.idx, len(study.taxas)))
            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot(111)
            ax = md2.visualization.aggregate_taxa_abundances(subj=subj, agg=taxa, dtype='rel', ax=ax)
            fig = plt.gcf()
            fig.tight_layout()
            plt.savefig(os.path.join(subjpath, '{}.pdf'.format(taxa.name)))
            plt.close()