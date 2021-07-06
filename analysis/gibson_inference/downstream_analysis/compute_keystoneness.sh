#!/bin/bash

set -e
source gibson_inference/settings.sh

echo "Youn TODO: update this with the new keystoneness fwsim script."

#TABLE_BASEPATH="../output/keystoneness/tables"
#CLUSTER_BASEPATH="../output/keystoneness/clusters"
#TAXA_BASEPATH="../output/keystoneness/taxa"
#SEP=","
#SIMULATION_DT="0.01"
#N_DAYS="60"
#
#echo "Computing Keystoneness"
#echo "Tables are saved in ${TABLE_BASEPATH}"
#echo "Output for leaving out taxa are saved in ${TAXA_BASEPATH}"
#echo "Output for leaving out clusters are saved in ${CLUSTER_BASEPATH}"
#
## Make the tables
## ---------------
#python scripts/make_leave_out_tables.py \
#    --chain ../output/mdsine2/healthy-seed0/mcmc.pkl \
#    --output-basepath $TABLE_BASEPATH \
#    --sep $SEP
#python scripts/make_leave_out_tables.py \
#    --chain ../output/mdsine2/healthy-seed1/mcmc.pkl \
#    --output-basepath $TABLE_BASEPATH \
#    --sep $SEP
#python scripts/make_leave_out_tables.py \
#    --chain ../output/mdsine2/uc-seed0/mcmc.pkl \
#    --output-basepath $TABLE_BASEPATH \
#    --sep $SEP
#python scripts/make_leave_out_tables.py \
#    --chain ../output/mdsine2/uc-seed1/mcmc.pkl \
#    --output-basepath $TABLE_BASEPATH \
#    --sep $SEP
#
## Compute keystoneness
## --------------------
## Healthy
#python ../keystoneness.py \
#    --input ../output/mdsine2/healthy-seed0/mcmc.pkl \
#    --study ../output/mdsine2/healthy-seed0/subjset.pkl \
#    --leave-out-table "${TABLE_BASEPATH}/healthy-seed0-taxa.csv" \
#    --sep $SEP \
#    --simulation-dt $SIMULATION_DT \
#    --n-days ${N_DAYS} \
#    --output-basepath ${TAXA_BASEPATH}
#python ../keystoneness.py \
#    --input ../output/mdsine2/healthy-seed0/mcmc.pkl \
#    --study ../output/mdsine2/healthy-seed0/subjset.pkl \
#    --leave-out-table "${TABLE_BASEPATH}/healthy-seed0-clusters.csv" \
#    --sep $SEP \
#    --simulation-dt $SIMULATION_DT \
#    --n-days ${N_DAYS} \
#    --output-basepath ${CLUSTER_BASEPATH}
#
#python ../keystoneness.py \
#    --input ../output/mdsine2/healthy-seed1/mcmc.pkl \
#    --study ../output/mdsine2/healthy-seed1/subjset.pkl \
#    --leave-out-table "${TABLE_BASEPATH}/healthy-seed1-taxa.csv" \
#    --sep $SEP \
#    --simulation-dt $SIMULATION_DT \
#    --n-days ${N_DAYS} \
#    --output-basepath ${TAXA_BASEPATH}
#python ../keystoneness.py \
#    --input ../output/mdsine2/healthy-seed1/mcmc.pkl \
#    --study ../output/mdsine2/healthy-seed1/subjset.pkl \
#    --leave-out-table "${TABLE_BASEPATH}/healthy-seed1-clusters.csv" \
#    --sep $SEP \
#    --simulation-dt $SIMULATION_DT \
#    --n-days ${N_DAYS} \
#    --output-basepath ${CLUSTER_BASEPATH}
#
## UC
#python ../keystoneness.py \
#    --input ../output/mdsine2/uc-seed0/mcmc.pkl \
#    --study ../output/mdsine2/uc-seed0/subjset.pkl \
#    --leave-out-table "${TABLE_BASEPATH}/uc-seed0-taxa.csv" \
#    --sep $SEP \
#    --simulation-dt $SIMULATION_DT \
#    --n-days ${N_DAYS} \
#    --output-basepath ${TAXA_BASEPATH}
#python ../keystoneness.py \
#    --input ../output/mdsine2/uc-seed0/mcmc.pkl \
#    --study ../output/mdsine2/uc-seed0/subjset.pkl \
#    --leave-out-table "${TABLE_BASEPATH}/uc-seed0-clusters.csv" \
#    --sep $SEP \
#    --simulation-dt $SIMULATION_DT \
#    --n-days ${N_DAYS} \
#    --output-basepath ${CLUSTER_BASEPATH}
#
#python ../keystoneness.py \
#    --input ../output/mdsine2/uc-seed1/mcmc.pkl \
#    --study ../output/mdsine2/uc-seed1/subjset.pkl \
#    --leave-out-table "${TABLE_BASEPATH}/uc-seed1-taxa.csv" \
#    --sep $SEP \
#    --simulation-dt $SIMULATION_DT \
#    --n-days ${N_DAYS} \
#    --output-basepath ${TAXA_BASEPATH}
#python ../keystoneness.py \
#    --input ../output/mdsine2/uc-seed1/mcmc.pkl \
#    --study ../output/mdsine2/uc-seed1/subjset.pkl \
#    --leave-out-table "${TABLE_BASEPATH}/uc-seed1-clusters.csv" \
#    --sep $SEP \
#    --simulation-dt $SIMULATION_DT \
#    --n-days ${N_DAYS} \
#    --output-basepath ${CLUSTER_BASEPATH}