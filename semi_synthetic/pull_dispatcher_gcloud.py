'''Pull to ErisOne
'''
import os


local_path = '/data/cctm/darpa_perturbation_mouse_study/perturbation_study_old/test_scp/'
scp_format = "gcloud comute scp --recurse {}:/mnt/disks/data/ {} --zone {}"
delete_format = 'gcloud compute instances delete {} --zone {}'
instance_format = 'semi-synth-dispatch-gcloud-test-final-{}'
for i in range(130):
    if i < 70:
        zone = 'us-central1-a'
    else:
        zone = 'us-east1-b'

    # Move the data
    instance_name = instance_format.format(i)
    command = scp_format.format(instance_name, local_path, zone)
    print(command)
    os.system(command)

    # delete the instance
    command = delete_format.format(instance_name, zone)
    print(command)
    os.system(command)


    
