'''Dispatch a set of jobs using docker
'''
import numpy as np
import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n-samples', '-ns', type=int,
        help='Total number of Gibbs steps to do',
        dest='n_samples', default=None)
parser.add_argument('--burnin', '-nb', type=int,
    help='Total number of burnin steps',
    dest='burnin', default=None)
parser.add_argument('--basepath', '-b', type=str,
    help='Basepath to save the output', default=None,
    dest='basepath')
parser.add_argument('--data-path', '-db', type=str,
    help='Folder to lead the data from', dest='data_path')
parser.add_argument('--mesh', type=str, 
    help='What kind of mesh to do ("time", "noise", "replicates")', 
    dest='mesh_type', default=None)
parser.add_argument('--mnt-path', '-mnt', type=str,
    help='Mount path', dest='mount_path', default='/mnt/disks/data')
args = parser.parse_args()
if args.mesh_type not in ['time', 'noise', 'replicates']:
    raise ValueError('Must specify the mesh type')

meshes = { 
    'time': ([4], [35, 45, 55, 65], 1, 1, [0.3], [0.1], [1], 1, 2),
    'noise': ([5], [55], 10, 1, [0.1, 0.2, 0.3, 0.4], [0.1], [1], 0, 0),
    'replicates': ([3,4,5], [55], 10, 1, [0.3], [0.1], [1], 0, 1)}
mesh = meshes[args.mesh_type]


basepath = args.basepath

my_str = '''
FROM python:3.7.3

WORKDIR /usr/src/app

COPY ./PyLab ./PyLab
RUN pip install PyLab/.

COPY ./MDSINE2 ./MDSINE2/
WORKDIR MDSINE2

RUN pip install --no-cache-dir -r requirements.txt
RUN python make_real_subjset.py

WORKDIR semi_synthetic
CMD python main_mcmc.py -m {} -p {} -d {} -i {} -b {} -nb {} -ns {} -nr {} -c {} -nt {} -db {} -us {}
'''

os.makedirs(basepath, exist_ok=True)
basepath = basepath + mesh + '/'
os.makedirs(basepath, exist_ok=True)
dockerdir = '../../dockers/'
os.makedirs(dockerdir, exist_ok=True)

n_replicates = mesh[0]
n_timepoints = mesh[1]
n_data_seeds = mesh[2]
n_init_seeds = mesh[3]
measurement_noises = mesh[4]
process_variances = mesh[5]
clustering_ons = mesh[6]
uniform_sampling = mesh[7]
boxplot_type = mesh[8]

for d in range(n_data_seeds):
    for i in range(n_init_seeds):
        for nr in n_replicates:
            for nt in n_timepoints:
                for mn in measurement_noises:
                    for pv in process_variances:
                
                        for co in clustering_ons:
                            if boxplot_type == 0:
                                # Do measurement 
                                jobname = 'mn{}_{}'.format(d,mn)
                            elif boxplot_type == 1:
                                # Do replicates
                                jobname = 'rep{}_{}'.format(d,nr)
                            else:
                                # Do number of timepoints
                                jobname = 'times{}_{}'.format(d,nt)

                            print(jobname)

                            name = 'd{}_i{}_ns{}_nb{}_nr{}_m{}_p{}_co{}_nt{}_us{}'.format(
                                d,i, args.n_samples, args.burnin, nr, mn, pv, 
                                co, nt, uniform_sampling)
                            
                            fname = dockerdir + name
                            f = open('../../Dockerfile', 'w')
                            f.write(my_str.format(
                                mn, pv, d, i, basepath, args.burnin, args.n_samples, 
                                nr, co, nt, args.data_path, uniform_sampling))
                            f.close()

                            command = 'more ../../Dockerfile'
                            print('\n\n\n\n\n\n')
                            print(command)
                            print('\n\n\n\n')
                            os.system(command)
                            command = 'docker build -t {} ../../'.format(jobname)
                            print('\n\n\n\n\n\n')
                            print(command)
                            print('\n\n\n\n')
                            os.system(command)
                            os.rename('../../Dockerfile', fname)

                            command = 'docker run --name {} --cpus 1 -v {}:' \
                                '/usr/src/app/MDSINE2/semi_synthetic/output {}'.format(
                                    jobname, args.mount_path, jobname)
                            print('\n\n\n\n\n\n')
                            print(command)
                            print('\n\n\n\n')
                            os.system(command)
