'''You can only have 69 VMs in a single zone
'''
import os
import sys

# import googleapiclient.discovery
# from six.moves import input

my_str = '''gcloud compute instances create-with-container {name} \
--zone {zone} \
--project=sinuous-mind-277319 \
--machine-type=custom-{n_cpus}-{ram_mem} \
--container-image=us.gcr.io/sinuous-mind-277319/test-auto-kill \
--container-restart-policy=never \
--container-command=python \
--container-arg=dispatch_gcloud.py \
--container-mount-host-path=mount-path=/usr/src/app/MDSINE2/semi_synthetic/output,host-path=/mnt/disks/data,mode=rw \
--no-restart-on-failure \
--labels=container-vm=cos-stable-81-12871-1196-0 \
--metadata-from-file=startup-script=startup_script.sh \
--service-account=56612871331-compute@developer.gserviceaccount.com \
--scopes=https://www.googleapis.com/auth/cloud-platform \
--tags=http-server,https-server \
--boot-disk-size=10GB --boot-disk-type=pd-standard --boot-disk-device-name={name}-boot \
--create-disk=mode=rw,size=100,type=projects/sinuous-mind-277319/zones/{zone}/diskTypes/pd-ssd,device-name={name}-data'''

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

    if n_zone+n_cpus > 69:
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
    command = my_str.format(name=name, zone=zone, mesh_n=mesh_n, n_cpus=n_cpus, 
        ram_mem=ram_mem)
    print(name)
    print(zone)
    print(mesh)
    print(command)
    os.system(command)
