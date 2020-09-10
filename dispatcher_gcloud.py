'''You can only have 69 VMs in a single zone
'''
import os

my_str = '''gcloud beta compute --project=sinuous-mind-277319 instances create-with-container semi-synth-dispatch-gcloud-test-final-{0} --zone={1} --machine-type=n1-standard-1 --subnet=default --network-tier=PREMIUM --metadata=^,@^google-logging-enabled=true,@startup-script=sudo\ mkfs.ext4\ -m\ 0\ -F\ -E\ lazy_itable_init=0,lazy_journal_init=0,discard\ /dev/sdb$'\n'sudo\ mkdir\ -p\ /mnt/disks/data$'\n'sudo\ mount\ -o\ discard,defaults\ /dev/sdb\ /mnt/disks/data$'\n'sudo\ chmod\ a\+w\ /mnt/disks/data/$'\n'echo\ \"{0}\"\ \>\>\ /mnt/disks/data/args.txt --maintenance-policy=MIGRATE --service-account=56612871331-compute@developer.gserviceaccount.com --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append --tags=http-server,https-server --image=cos-stable-81-12871-1196-0 --image-project=cos-cloud --boot-disk-size=10GB --boot-disk-type=pd-standard --boot-disk-device-name=semi-synth-dispatch-gcloud-test-final-{0} --create-disk=mode=rw,size=100,type=projects/sinuous-mind-277319/zones/us-central1-a/diskTypes/pd-ssd,device-name=persistent-disk-{0} --container-image=us.gcr.io/sinuous-mind-277319/semi-synth-dispatch-gcloud-test-final --container-restart-policy=never --container-privileged --container-command=python --container-arg=dispatch_gcloud.py --container-mount-host-path=mount-path=/usr/src/app/MDSINE2/semi_synthetic/output,host-path=/mnt/disks/data,mode=rw --labels=container-vm=cos-stable-81-12871-1196-0'''
for i in range(130):
    if i < 70:
        zone = 'us-central1-a'
    else:
        zone = 'us-east1-b'
    os.system(my_str.format(i,zone))
