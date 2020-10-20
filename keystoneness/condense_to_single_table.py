'''Condense the output of keystoneness into single tables.

The only thing we need is the filename.
We assume that the folder where the runs are in a folder with the same name
as the leave out table. We will save the output table as a tsv file

'''
import numpy as np
import pandas as pd
import logging
import sys
import os
import time
import argparse
import re

import pylab as pl

logging.basicConfig(level=logging.DEBUG)


if __name__ == '__main__':

    # Parse the input
    # ---------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--leave-out-table', type=str, dest='leave_out_table',
        help='Filename containing which ASVs to remove at each iteration')
    parser.add_argument('--subjset', type=str, dest='subjset',
        help='Filename for the SubjectSet')
    args = parser.parse_args()

    subjset = pl.SubjectSet.load(args.subjset)
    suffix = None
    if '.txt' in args.leave_out_table:
        suffix = '.txt'
    elif '.csv' in args.leave_out_table:
        suffix = '.csv'
    else:
        raise ValueError('Suffix not recognized (only parsing .csv and .txt).')
    run_basepath = args.leave_out_table.replace(suffix, '/')
    
    # Get the list of ASVs
    rev_idx = {}
    for asv in subjset.asvs:
        rev_idx[asv.name] = asv.idx

    # Get the list of leave out
    f = open(args.leave_out_table, 'r')
    txt = f.read()
    f.close()
    tbl_ = txt.split('\n')
    tbl = []
    for a in tbl_:
        if a == '':
            continue
        tbl.append(a)


    data = []
    onlyfiles = [f for f in os.listdir(run_basepath) if os.path.isfile(run_basepath+f)]

    get_idx = re.compile(r'(\d+)')
    index = []

    for fname in onlyfiles:
        # get the asvs

        ret = np.zeros(len(subjset.asvs), dtype=float)*np.nan
        mask = np.ones(len(subjset.asvs), dtype=bool)

        try:
            idx = int(get_idx.findall(fname)[0])
            asvnames = tbl[idx].split(',')
            asvidxs = [rev_idx[asvname] for asvname in asvnames]
            mask[asvidxs] = False
            index.append(tuple(asvnames))
        except:
            # This is the base line (there is a None there)
            idx = None
            index.append('base')

        values = np.load(run_basepath + fname)
        ret[mask]=values
        # print(ret)
        data.append(ret)

    df = pd.DataFrame(data, columns=[asv.name for asv in subjset.asvs], index=index)
    fname = args.leave_out_table.replace(suffix, '.tsv')
    df.to_csv(fname, sep='\t', index=True, header=True)

    


