'''Filter the Gibson dataset using `mdsine2.consistency_filtering` or 
`mdsine2.conditional_consistency_filtering`.

Author: David Kaplan
Date: 11/18/20
MDSINE2 version: 4.0.2

Methodology
-----------
1) Load in dataset
   This is a pickle of an `mdsine2.Study` object. This can be created with the 
   `MDSINE2.gibson_1_preprocessing.py` script. To quickly generate this pickle:
    ```python
    import mdsine2
    dset = mdsine2.dataset.gibson()
    dset.save('file/location.pkl')
    ```
2) Perform filtering
   Filter out the ASVs/OTUs that do not have enough dynamical information for 
   effective inference.
3) Save the filtered dataset

Parameters
----------
--dataset, -i : str, +
    This is the file location for the dataset you're going to filter.
    This can be multiple datasets
--other : str, +
    This is the file locations for the other dataset used in `conditional_consistency_filtering`.
    Only necessary if `--filtering = conditional_consistency_filtering`.
--outfile, -o : str, +
    This is the location to save the filtered dataset. There has to be an
    output location for each input datasets
--filtering, -f : str
    What type of filtering you are doing. Options are 'consistency_filtering' or 
    'conditional_consistency_filtering'
--dtype, -d : str
    This is what data we are thresholding. Options are 'raw' (counts), 'rel' (relative
    abundance), or 'abs' (absolute abundance).
--threshold, -t : float, int
    This is the threshold the asv must pass at each timepoint
--min-num-consecutive, -m : int
    Number of consecutive timepoints to look for in a row
--min-num-consecutive-lower, -ml : int
    Number of consecutive timepoints to look for in a row for conditional. Only necessary if
    `--filtering = conditional_consistency_filtering`.
--min-num-subjects, -s : int
    This is the minimum number of subjects this needs to be valid for.
--colonization-time, -c : int
    This is the time we are looking after for colonization. If None we assume 
    there is no colonization time.

Reproducability
---------------
To reproduce the paper, first run the script `gibson_1_preprocessing.py`, then:
Linux/MacOS:
python gibson_2_filtering.py \
    --dataset gibson_output/datasets/gibson_healthy_agg.pkl gibson_output/datasets/gibson_uc_agg.pkl \
    --outfile gibson_output/datasets/gibson_healthy_agg_filtered.pkl gibson_output/datasets/gibson_uc_agg_filtered.pkl \
    --filtering consistency_filtering \
    --dtype rel \
    --threshold 0.0001 \
    --min-num-consecutive 7 \
    --min-num-subjects 2 \
    --colonization-time 5
PC:
python gibson_2_filtering.py `
    --dataset gibson_output/datasets/gibson_healthy_agg.pkl gibson_output/datasets/gibson_uc_agg.pkl `
    --outfile gibson_output/datasets/gibson_healthy_agg_filtered.pkl gibson_output/datasets/gibson_uc_agg_filtered.pkl `
    --filtering consistency_filtering `
    --dtype rel `
    --threshold 0.0001 `
    --min-num-consecutive 7 `
    --min-num-subjects 2 `
    --colonization-time 5
'''
import argparse
import mdsine2 as md2
import logging

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-i', type=str, dest='datasets',
        help='This is the Gibson dataset that you want to parse. You can load in ' \
            'multiple datasets.', nargs='+')
    parser.add_argument('--other', type=str, dest='others',
        help='This is the file locations for the other dataset used in `conditional_consistency_filtering`. ' \
             'Only necessary if `--filtering = conditional_consistency_filtering`.', nargs='+')
    parser.add_argument('--outfile', '-o', type=str, dest='outfiles',
        help='This is where you want to save the parsed dataset. Each dataset in ' \
            '`--dataset` must have an output.', nargs='+')
    parser.add_argument('--filtering', '-f', type=str, dest='filtering',
        help='What type of filtering you are doing. Options are "consistency_filtering" or ' \
             '"conditional_consistency_filtering"')
    parser.add_argument('--dtype', '-d', type=str, dest='dtype',
        help='This is what data we are thresholding. Options are "raw" (counts), "rel" (relative ' \
             'abundance), or "abs" (absolute abundance).')
    parser.add_argument('--threshold', '-t', type=float, dest='threshold',
        help='This is the threshold the asv must pass at each timepoint')
    parser.add_argument('--min-num-consecutive', '-m', type=int, dest='min_num_consecutive',
        help='Number of consecutive timepoints to look for in a row')
    parser.add_argument('--min-num-consecutive-lower', '-ml', type=int, dest='min_num_consecutive_lower',
        help='Number of consecutive timepoints to look for in a row for conditional. Only necessary if '\
             '`--filtering = conditional_consistency_filtering`.', default=None)
    parser.add_argument('--min-num-subjects', '-s', type=int, dest='min_num_subjects',
        help='This is the minimum number of subjects this needs to be valid for.')
    parser.add_argument('--colonization-time', '-c', type=int, dest='colonization_time',
        help='This is the time we are looking after for colonization. Default to nothing', default=None)

    args = parser.parse_args()
    md2.config.LoggingConfig(level=logging.INFO)
    if len(args.datasets) != len(args.outfiles):
        raise ValueError('Each dataset ({}) must have an outfile ({})'.format(
            len(args.datasets), len(args.outfiles)))

    if args.filtering == 'conditional_consistency_filtering':
        if args.min_num_consecutive_lower is None:
            raise ValueError('If filtering is conditional, must also provide `--min-num-consecutive-lower`')
        if len(args.datasets) != len(args.others):
            raise ValueError('Each dataset ({}) must have an other ({})'.format(
                len(args.dataset), len(args.others)))

    for iii, dset in enumerate(args.datasets):
        print('\n\n----------------------------')
        print('On Dataset {}'.format(dset))

        # 1) Load dataset
        study = md2.Study.load(dset)

        # 2) Filtering
        initial_num = len(study.asvs)
        if args.filtering == 'consistency_filtering':
            print('Consistency filtering')
            study = md2.consistency_filtering(subjset=study,
                dtype=args.dtype,
                threshold=args.threshold,
                min_num_consecutive=args.min_num_consecutive,
                min_num_subjects=args.min_num_subjects,
                colonization_time=args.colonization_time)

        elif args.filtering == 'conditional_consistency_filtering':
            print('Conditional consistency filtering')
            study = md2.conditional_consistency_filtering(subjset=study,
                other=md2.Study.load(args.others[iii]),
                dtype=args.dtype,
                threshold=args.threshold,
                min_num_consecutive_upper=args.min_num_consecutive,
                min_num_consecutive_lower=args.min_num_consecutive_lower,
                min_num_subjects=args.min_num_subjects,
                colonization_time=args.colonization_time)

        else:
            raise ValueError('`filtering` ({}) not recognized'.format(args.filtering))

        print('{} ASVs remaining from {}'.format(len(study.asvs), initial_num))

        # 3) Save file
        study.save(args.outfiles[iii])
    
