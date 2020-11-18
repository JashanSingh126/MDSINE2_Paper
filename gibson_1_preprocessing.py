'''Preprocess (aggregate and filter) the Gibson dataset for Healthy cohort, 
Ulcerative Colitis cohort, inoculum, and replicate read datasets.

Author: David Kaplan
Date: 11/17/20
MDSINE2 version: 4.0.1

Methodology
-----------
1) Load the dataset
2) Set the sequences of the ASVs to the aligned sequences used in building the 
   phylogenetic tree, instead of them straight from DADA2 (those are unaligned).
   We set the sequences of the ASVs to be gapless, i.e. we remove all alignment
   positions where any sequence has a gap.
3) Aggregate ASVs with a specified hamming distance
4) Rename aggregated ASVs into OTUs

Parameters
----------
--dataset, -i, -d : str (multiple)
    This is the Gibson dataset that you want to parse. You can load in multiple datasets.
--hamming-distance, -hd : int
    This is the hamming radius to aggregate ASV sequences. If nothing is provided, 
    then there will be no aggregation.
--rename-prefix, -rp : str
    This is the prefix you are renaming the aggregate asvs to. If nothing is provided,
    then they will not be renamed.
--sequences, -s : str
    This is the fasta file location of the aligned sequences for each ASV that was 
    used for placement in the phylogenetic tree. If nothing is provided, then do 
    not replace them.
--outfile, -o : str (multiple)
    This is where you want to save the parsed dataset. Each dataset in `--dataset` must
    have an output.

Reproducability
---------------
To reproduce the paper, run:
Linux/MacOS:
python gibson_preprocessing.py \
    --dataset healthy uc replicates \
    --hamming-distance 2 \
    --rename-prefix OTU \
    --sequences gibson_files/preprocessing/gibson_16S_rRNA_v4_seqs_aligned_filtered.fa \
    --outfile gibson_output/datasets/gibson_healthy_agg.pkl gibson_output/datasets/gibson_uc_agg.pkl \
              gibson_output/datasets/gibson_replicate_agg.pkl 
PC:
python gibson_1_preprocessing.py `
    --dataset healthy uc replicates `
    --hamming-distance 2 `
    --rename-prefix OTU `
    --sequences gibson_files/preprocessing/gibson_16S_rRNA_v4_seqs_aligned_filtered.fa `
    --outfile gibson_output/datasets/gibson_healthy_agg.pkl gibson_output/datasets/gibson_uc_agg.pkl `
              gibson_output/datasets/gibson_replicate_agg.pkl 

The file `paper_files/preprocessing/gibson_16S_rRNA_v4_seqs_aligned_filtered.fa` 
was prepared by first aligning the ASV sequences to the reference sequeunces in the 
phylogenetic tree. Once aligned, ASVs were manually filtered out if they had poor alignment 
within the 16S rRNA v4 region. A fasta file of the ASVs removed as well as their alignments 
can be found in `paper_files/preprocessing/prefiltered_asvs.fa`. 


See Also
--------
mdsine2.visualization.aggregate_asv_abundances
'''
import argparse
import pandas as pd
from Bio import SeqIO
import numpy as np
import logging
import mdsine2 as md2

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', '-i', type=str, dest='datasets',
        help='This is the Gibson dataset that you want to parse. You can load in ' \
            'multiple datasets.', nargs='+')
    parser.add_argument('--outfile', '-o', type=str, dest='outfiles',
        help='This is where you want to save the parsed dataset. Each dataset in ' \
            '`--dataset` must have an output.', nargs='+')
    parser.add_argument('--hamming-distance', '-hd', type=int, dest='hamming_distance',
        help='This is the hamming radius to aggregate ASV sequences. If nothing ' \
            'is provided, then there will be no aggregation.', default=None)
    parser.add_argument('--rename-prefix', '-rp', type=str, dest='rename_prefix',
        help='This is the prefix you are renaming the aggregate asvs to. ' \
            'If nothing is provided, then they will not be renamed', default=None)
    parser.add_argument('--sequences', '-s', type=str, dest='sequences',
        help='This is the fasta file location of the aligned sequences for each ASV' \
            ' that was used for placement in the phylogenetic tree. If nothing is ' \
            'provided, then do not replace them.', default=None)
    
    args = parser.parse_args()
    md2.config.LoggingConfig(level=logging.INFO)
    if len(args.datasets) != len(args.outfiles):
        raise ValueError('Each dataset ({}) must have an outfile ({})'.format(
            len(args.datasets), len(args.outfiles)))

    for iii, dset in enumerate(args.datasets):
        print('\n\n----------------------------')
        print('On Dataset {}'.format(dset))

        # 1) Load the dataset
        study = md2.dataset.gibson(dset=dset, as_df=False, species_assignment='both')

        # 2) Set the sequences for each ASV
        #    Remove all asvs that are not contained in that file
        #    Remove the gaps
        if args.sequences is not None:
            print('Replacing sequences with the file {}'.format(args.sequences))
            seqs = SeqIO.to_dict(SeqIO.parse(args.sequences, format='fasta'))
            to_delete = []
            for asv in study.asvs:
                if asv.name not in seqs:
                    to_delete.append(asv.name)
            for name in to_delete:
                print('Deleting {} because it was not in {}'.format(
                    name, args.sequences))
            study.pop_asvs(to_delete)

            M = []
            for asv in study.asvs:
                seq = list(str(seqs[asv.name].seq))
                M.append(seq)
            M = np.asarray(M)
            gaps = M == '-'
            n_gaps = np.sum(gaps, axis=0)
            idxs = np.where(n_gaps == 0)[0]
            print('There are {} positions where there are no gaps out of {}. Setting those ' \
                'to the sequences'.format(len(idxs), M.shape[1]))
            M = M[:, idxs]
            for i,asv in enumerate(study.asvs):
                asv.sequence = ''.join(M[i])

        # 3) Aggregate with specified hamming distance
        if args.hamming_distance is not None:
            print('Aggregating ASVs with a hamming distance of {}'.format(args.hamming_distance))
            study = md2.aggregate_items(subjset=study, hamming_dist=args.hamming_distance)
        
        # 4) Rename asvs
        if args.rename_prefix is not None:
            print('Renaming ASVs with prefix {}'.format(args.rename_prefix))
            study.asvs.rename(prefix=args.rename_prefix, zero_based_index=False)

        # Save
        study.save(args.outfiles[iii])




        




        



    
    