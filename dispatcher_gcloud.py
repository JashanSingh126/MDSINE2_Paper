'''You can only have 69 VMs in a single zone
'''
import os
import sys

my_str = '''gcloud beta compute --project=sinuous-mind-277319 instances create-with-container {name} --zone={zone} --machine-type=n1-standard-1 --subnet=default --network-tier=PREMIUM --metadata=^,@^google-logging-enabled=true,@startup-script=\#\!/usr/bin/env\ bash$'\n'$'\n'\#\ Mount\ disk$'\n'sudo\ mkfs.ext4\ -m\ 0\ -F\ -E\ lazy_itable_init=0,lazy_journal_init=0,discard\ /dev/sdb$'\n'sudo\ mkdir\ -p\ /mnt/disks/data$'\n'sudo\ mount\ -o\ discard,defaults\ /dev/sdb\ /mnt/disks/data$'\n'sudo\ chmod\ a\+w\ /mnt/disks/data/$'\n'echo\ \"{mesh_n}\"\ \>\>\ /mnt/disks/data/args.txt$'\n'$'\n'gcloud\ beta\ auth\ application-default\ login --no-restart-on-failure --maintenance-policy=MIGRATE --service-account=56612871331-compute@developer.gserviceaccount.com --scopes=https://www.googleapis.com/auth/cloud-platform --tags=http-server,https-server --image=cos-stable-81-12871-1196-0 --image-project=cos-cloud --boot-disk-size=10GB --boot-disk-type=pd-standard --boot-disk-device-name={name}-boot --create-disk=mode=rw,size=200,type=projects/sinuous-mind-277319/zones/{zone}/diskTypes/pd-ssd,device-name={name}-data --container-image=us.gcr.io/sinuous-mind-277319/test-auto-kill-full --container-restart-policy=never --container-command=python --container-arg=dispatch_gcloud.py --container-mount-host-path=mount-path=/usr/src/app/MDSINE2/semi_synthetic/output,host-path=/mnt/disks/data,mode=rw --labels=container-vm=cos-stable-81-12871-1196-0'''

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
zones = ['us-central1-a', 'us-east1-b', 'us-west1-a', 'northamerica-northeast1-a']


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

    zone = zones[mesh_n // 70]

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
    command = my_str.format(name=name, zone=zone, mesh_n=mesh_n)
    print()
    print(name)
    print(mesh)
    os.system(command)
    # sys.exit()

