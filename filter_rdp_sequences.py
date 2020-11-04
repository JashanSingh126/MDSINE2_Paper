'''Filter RDP sequences
'''
import numpy as np
import pandas as pd
from Bio import SeqIO
import os
import sys

import matplotlib.pyplot as plt

import config

config.LoggingConfig()

def truncate_sequences(seqs, truncation_length):
    '''Get rid of all sequences longer than `truncation_length`

    Parameters
    ----------
    seqs : dict (str->Bio.record)
    truncation_length : int

    Returns
    -------
    list
    '''
    ret = []
    for seq, record in seqs.items():
        # print(len(record.seq))
        if len(record.seq) <= truncation_length:
            ret.append(record)
    return ret

# seqs = SeqIO.to_dict(SeqIO.parse('tmp/sequences/RDP_typed_cultured.fa', 'fasta'))
# ret = truncate_sequences(seqs, 1600)
# SeqIO.write(ret, 'tmp/sequences/RDP_typed_cultured_trunc1600.fa', 'fasta')
# sys.exit()

def filter_seqs(seqs_aligned, seqs_raw, gaplen, n_occur, fname=None, M=None):
    '''Find areas in the total alignment that have less than or equal to
    `n_occur` bases along an alignment position for at least `gaplen` consecutive
    positions.

    Parameters
    ----------
    seqs_aligned : dict(str -> Bio.record)
        Sequences
    gaplen : int
    n_occur : int

    Returns
    -------
    list(str)
    '''
    seqnames = list(seqs_aligned.keys())
    if M is None:
        for i, seqname in enumerate(seqs_aligned):
            if i % 1000 == 0:
                print('{}/{}'.format(i,len(seqs_aligned)))
            seq = str(seqs_aligned[seqname].seq)
            arr = np.asarray([a in ['.', '-'] for a in seq], dtype=bool)

            if M is None:
                M = np.zeros(shape=(len(seqs_aligned), len(arr)), dtype=bool)
            M[i] = arr

    countbases = len(seqs_aligned) - M.sum(axis=0)

    if fname is not None:
        f = open(fname, 'w')

    start_pos = None
    end_pos = None
    to_delete = []
    already_deleted = set([])
    for pos, counts in enumerate(countbases):
        if counts <= n_occur:
            if start_pos is None:
                start_pos = pos
        else:
            if start_pos is not None:
                end_pos = pos

                if end_pos - start_pos >= gaplen:

                    # Get all of the sequences that should be deleting for this insertion
                    idxs = np.unique(np.where(~M[:, start_pos:end_pos])[0])
                    # print(idxs)

                    for idx in idxs:
                        if idx in already_deleted:
                            continue
                        # print('Deleting {} because it contains basepairs ' \
                        #     'in the location {} to {}'.format(seqnames[idx], start_pos, end_pos))
                        to_delete.append((idx,start_pos,end_pos))
                        already_deleted.add(idx)
            start_pos = None
            end_pos = None

            # Else this just means that there are many in a row that fit the filtering
    if start_pos is not None:
        end_pos = M.shape[1]
        if end_pos - start_pos >= gaplen:
            # Get all of the sequences that should be deleting for this insertion
            idxs = np.unique(np.where(~M[:, start_pos:end_pos])[0])
            print('here mf')
            print(idxs)
            for idx in idxs:
                if idx in already_deleted:
                    continue
                # print('Deleting {} because it contains basepairs ' \
                #     'in the location {} to {}'.format(seqnames[idx], start_pos, end_pos))
                print('here')
                print(seqnames[idx])
                to_delete.append((idx,start_pos,end_pos))
                already_deleted.add(idx)

    
    if fname is not None:
        for seq,start,end in to_delete:
            aaa = seqs_aligned[seqnames[seq]]
            f.write('-------------\n')
            f.write('start: {}, end: {}\n'.format(start,end))
            f.write(str(aaa))
            f.write('\n')
        f.close()

    names_to_delete = [seqnames[seq] for seq,_,_ in to_delete]
    print('N to delete', len(names_to_delete))
    ret = []
    for seq in seqs_aligned:
        if seq == '#=GC_RF':
            continue
        if seq in names_to_delete:
            continue
        ret.append(seqs_raw[seq])

    return ret

# # ---------
# # Make plot
# # ---------
# for i, seqname in enumerate(seqs):
#     if i % 1000 == 0:
#         print('{}/{}'.format(i,len(seqs)))
#     seq = str(seqs[seqname].seq)
#     arr = np.asarray([a in ['.', '-'] for a in seq], dtype=bool)

#     if M is None:
#         M = np.zeros(shape=(len(seqs), len(arr)), dtype=bool)
#     M[i] = arr

# fig = plt.figure(figsize=(10,5))
# ax = fig.add_subplot(111)
# for gaplen in np.arange(1,11):
#     print('{}/{}'.format(gaplen-1, 10))
#     ls = []
#     for n_occur in np.arange(2,11):
#         ls.append(filter_seqs(seqs, gaplen=gaplen, n_occur=n_occur, M=M))
    
#     ax.plot(np.arange(2,11), ls, label='Gap Length {}'.format(gaplen), marker='x')
# ax.legend(bbox_to_anchor=(1.05, 1))
# ax.set_ylabel('Reference sequences deleted')
# ax.set_xlabel('Max N')
# # ax.set_yscale('log')
# fig.subplots_adjust(right=0.795)
# plt.savefig('tmp/reference_seqs_filter_thresh.pdf')
# plt.show()
# sys.exit()

# -----------------
# Perform filtering
# -----------------
rdp_fname = 'tmp/sequences/RDP_typed_cultured_trunc1600_aligned.fa'
seqs = SeqIO.to_dict(SeqIO.parse(rdp_fname, 'fasta'))
rdp_fname_raw = 'tmp/sequences/RDP_typed_cultured_trunc1600.fa'
seqs_raw = SeqIO.to_dict(SeqIO.parse(rdp_fname_raw, 'fasta'))

gaplen = 1
n_occur = 3



# print(len(seqs_raw))
# print(len(seqs))
# sys.exit()

ret = filter_seqs(seqs, seqs_raw=seqs_raw, gaplen=gaplen, n_occur=n_occur)

print('orig len')
print(len(seqs))
print('new len')
print(len(ret))
print('todelte')
print(len(seqs_raw))

SeqIO.write(ret, 'tmp/sequences/RDP_typed_cultured_trunc1600_gaplen{}_ninsert{}.fa'.format(gaplen, n_occur), 'fasta')