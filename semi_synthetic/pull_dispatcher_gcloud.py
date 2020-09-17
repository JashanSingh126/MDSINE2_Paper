'''Pull to ErisOne

Resume
Pull
Delete
'''
import os
import argparse




local_path = '/data/cctm/darpa_perturbation_mouse_study/semi_synthetic_output/runs/'
resume_format = 'gcloud beta compute instances resume {name}'
scp_format = "gcloud comute scp --recurse {local_path}:/mnt/disks/data/ {name} --zone {zone}"
delete_format = 'gcloud compute instances delete {name} --zone {zone}'

# os.makedirs(local_path, exist_ok=True)

meshes = [
    ([5], [55], 10, 1, [0.1, 0.2, 0.25, 0.3, 0.4], [0.1], [1], 0, 0),    
    ([2,3,4,5], [55], 10, 1, [0.3], [0.1], [1], 0, 1), # 5 replicates is included in the measurement noise one
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
        if d > max_dataseeds:
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


boxplot_names = ['noise{n}-{seed}', 'replicates{n}-{seed}', 'times{n}-{seed}']
n_cpus = 2
ram_mem = 6656
zones = ['us-central1-a', 'us-east1-b', 'us-west1-a', 'us-west2-a', 'us-west3-a', 'us-east4-a']
n_zone = 0
i_zone = 0

lstnames = []
lstzones = []

for mesh_n in range(len(arguments_global)):
    mesh = arguments_global[mesh_n]
    n_replicates = mesh[0]
    n_timepoints = mesh[1]
    data_seed = mesh[2]
    init_seed = mesh[3]
    measurement_noise = mesh[4]
    process_variance = mesh[5]
    uniform_sampling = mesh[6]
    boxplot_type = mesh[7]

    if i_zone+n_cpus > 69:
        i_zone += 1
        n_zone = 0
    
    zone = zones[i_zone]
    n_zone += n_cpus

    if boxplot_type == 0:
        ns = measurement_noise
        namefmt = boxplot_names[0]

        ns = str(ns).replace('.', '-').replace('[', '').replace(']', '')

    elif boxplot_type == 1:
        ns = n_replicates
        namefmt = boxplot_names[1]
    else:
        ns = n_timepoints
        namefmt = boxplot_names[2]

    name = namefmt.format(n=ns, seed=data_seed)

    lstnames.append(name)
    lstzones.append(zone)

# Resume
for i in range(len(lstnames)):
    name = lstnames[i]
    command = resume_format.format(name=name)
    os.system(command)
    break

# Move the data
for i in range(len(lstnames)):
    name = lstnames[i]
    zone = lstzones[i]
    command = scp_format.format(name=name, local_path=local_path, zone=zone)
    print(command)
    os.system(command)
    break

# delete the instance
for i in range(len(lstnames)):
    name = lstnames[i]
    zone = lstzones[i]
    command = delete_format.format(name=name, zone=zone)
    print(command)
    os.system(command)
    break


    
