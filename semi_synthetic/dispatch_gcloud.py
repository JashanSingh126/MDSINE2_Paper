'''
Mount the external drive and run with the desired option
'''

import os
import argparse
import sys

from pprint import pprint
import requests
from googleapiclient import discovery
from oauth2client.client import GoogleCredentials

def StopVm():
    credentials = GoogleCredentials.get_application_default()

    service = discovery.build('compute', 'v1', credentials=credentials)
    metadata_server = "http://metadata/computeMetadata/v1/instance/"
    metadata_flavor = {'Metadata-Flavor' : 'Google'}
    res =(requests.get(metadata_server + 'hostname', headers = metadata_flavor).text).split('.')
    # Project ID for this request.
    project = res[3]

    # The name of the zone for this request.
    zone = res[1]

    # Name of the instance resource to stop.
    instance = res[0]  

    request = service.instances().stop(project=project, zone=zone, instance=instance)
    response = request.execute()

    pprint(response)

f = open('output/args.txt', 'r')
argument_option = int(f.read())
f.close()

meshes = [
    ([5], [55], 10, 1, [0.1, 0.2, 0.25, 0.3, 0.4], [0.1], [1], 0, 0),    
    ([2,3,4], [55], 10, 1, [0.3], [0.1], [1], 0, 1), # 5 replicates is included in the measurement noise one
    ([4], [35, 45, 50, 55, 65], 10, 1, [0.3], [0.1], [1], 1, 2)]

arguments_global = []

agg_repliates = set([])
agg_times = set([])
agg_measurement_noise = set([])
max_dataseeds = -1
agg_process_variances = set([])

for mesh in meshes:
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
        if d < max_dataseeds:
            max_dataseeds = d
        for i in range(n_init_seeds):
            for nr in n_replicates:
                agg_repliates.add(str(nr))
                for nt in n_timepoints:
                    agg_times.add(str(nt))
                    for mn in measurement_noises:
                        agg_measurement_noise.add(str(mn))
                        for pv in process_variances:
                            agg_process_variances.add(str(pv))
                            for co in clustering_ons:
                                arr = [nr, nt, d, i, mn, pv, uniform_sampling, boxplot_type]
                                arguments_global.append(arr)

lst_replicates = ' '.join(agg_repliates)
lst_measurement_noises = ' '.join(agg_measurement_noise)
lst_times = ' '.join(agg_times)
lst_process_variances = ' '.join(agg_process_variances)

if argument_option >= len(arguments_global):
    raise ValueError('`argument_option` ({}) too large. {} max'.format(
        argument_option, len(arguments_global)-1))

mesh = arguments_global[argument_option]
n_replicates = mesh[0]
n_timepoints = mesh[1]
data_seed = mesh[2]
init_seed = mesh[3]
measurement_noise = mesh[4]
process_variance = mesh[5]
uniform_sampling = mesh[6]
boxplot_type = mesh[7]


# Make the base data
base_data_path = 'output/base_data/'
command = 'python make_subjsets.py -b {basepath} -nr {nrs} -m {mns} -p {pvs} -d {nd} -dset semi-synthetic -nt {nts}'.format(
    basepath=base_data_path, nrs=lst_replicates, mns=lst_measurement_noises,
    pvs=lst_process_variances, nd=max_dataseeds, nts=lst_times)
print('EXECUTING:', command)
# os.system(command)

print('Arguments: {}'.format(arguments_global[argument_option]))

# Run the docker
output_path = 'output/'
command = 'python main_mcmc.py -d {} -i {} -m {} -p {} -b {} -db {} -ns {} -nb {} -nt {} -nr {} -us {}'.format(
    data_seed, init_seed, measurement_noise, process_variance, output_path, base_data_path,
    100, 50, n_timepoints, n_replicates, uniform_sampling)
print('EXECUTING:', command)
# os.system(command)

# Kill the vm
StopVm()