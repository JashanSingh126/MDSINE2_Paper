#!/bin/bash

export HOSTNAME=$(curl -X GET http://metadata.google.internal/computeMetadata/v1/instance/name -H 'Metadata-Flavor: Google')
echo $HOSTNAME > output/hostname.txt

export ZONE=$(curl -X GET http://metadata.google.internal/computeMetadata/v1/instance/zone -H 'Metadata-Flavor: Google')
echo $ZONE > output/zone.txt

export PROJECTID=$(curl -X GET http://metadata.google.internal/computeMetadata/v1/project/project-id -H 'Metadata-Flavor: Google')
echo $PROJECTID > output/project_id.txt
