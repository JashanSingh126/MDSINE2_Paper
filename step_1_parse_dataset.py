'''Parse your dataset into MDSINE2 format
'''
import argparse
import mdsine2 as md2
import logging

if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--taxonomy', '-t', type=str, dest='taxonomy',
    #     help='This is the table showing the sequences and the taxonomy for each' \
    #         ' ASV or OTU')
    # parser.add_argument('--metadata', '-m', type=str, dest='metadata',
    #     help='This is the metadata table')
    # parser.add_argument('--reads', '-r', type=str, dest='reads',
    #     help='This is the reads table', default=None)
    # parser.add_argument('--qpcr', '-q', type=str, dest='qpcr',
    #     help='This is the qPCR table', default=None)
    # parser.add_argument('--reads', '-r', type=str, dest='reads',
    #     help='This is the reads table', default=None)
    # parser.add_argument('--perturbations', '-p', type=str, dest='perturbations',
    #     help='This is the perturbation table', default=None)
    # parser.add_argument('--outfile', '-o', type=str, dest='outfile',
    #     help='This is where you want to save the parsed dataset')
    # args = parser.parse_args()

    # md2.config.LoggingConfig(level=logging.INFO)

    # taxonomy = pd.read_csv(args.taxonomy)
    # taxas = md2.ASVSet()


    study = md2.dataset.gibson(dset='healthy')
    



    
    
    