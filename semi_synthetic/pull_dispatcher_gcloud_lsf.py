import os
import sys

mystr = '''
#!/bin/bash
#BSUB -J test_seqs
#BSUB -o lsf_place_seqs_output.out
#BSUB -e lsf_place_seqs_error.err
# Please make a copy of this script for your own modifications

#BSUB -q big-multi
#BSUB -n 4
#BSUB -M 10000
#BSUB -R rusage[mem=10000]

# Add your job command here
# Load module

source activate dispatcher
module load gcloud/default

cd /data/cctm/darpa_perturbation_mouse_study/data/cctm/darpa_perturbation_mouse_study/semi_synthetic_output/
python pull_dispatcher_gcloud.py
'''

i = 13
num = 20
most = 130

while i < most:

    bottom = i
    top = i + num
    if top > most:
        top = most
    
    fname = 'pull_dispatcher_lsf{}.py'.format(i)
    f = open(fname, 'w')
    f.write(mystr.format(bottom, top))
    f.close()

    command = 'bsub < ' + fname
    # command = 'python pull_dispatcher_gcloud.py --bottom {} --top {}'.format(bottom,top)
    os.system(command)
    sys.exit()



