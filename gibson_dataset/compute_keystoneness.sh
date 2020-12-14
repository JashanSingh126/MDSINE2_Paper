#!/bin/bash

# Make the tables
# ---------------
python scripts/make_leave_out_tables.py \
    --chain ../output/mdsine2/healthy-seed0/mcmc.pkl \
    --output-basepath ../output/keystoneness/tables \
    --sep ,
python scripts/make_leave_out_tables.py \
    --chain ../output/mdsine2/healthy-seed1/mcmc.pkl \
    --output-basepath ../output/keystoneness/tables \
    --sep ,
python scripts/make_leave_out_tables.py \
    --chain ../output/mdsine2/uc-seed0/mcmc.pkl \
    --output-basepath ../output/keystoneness/tables \
    --sep ,
python scripts/make_leave_out_tables.py \
    --chain ../output/mdsine2/uc-seed1/mcmc.pkl \
    --output-basepath ../output/keystoneness/tables \
    --sep ,

# Compute keystoneness
# --------------------
# Healthy
python ../keystoneness.py \
    --input ../output/mdsine2/healthy-seed0/mcmc.pkl \
    --study ../output/mdsine2/healthy-seed0/subjset.pkl \
    --leave-out-table ../output/keystoneness/tables/healthy-seed0-taxa.csv \
    --sep , \
    --simulation-dt 0.01 \
    --n-days 60 \
    --output-basepath ../output/keystoneness/taxa
python ../keystoneness.py \
    --input ../output/mdsine2/healthy-seed0/mcmc.pkl \
    --study ../output/mdsine2/healthy-seed0/subjset.pkl \
    --leave-out-table ../output/keystoneness/tables/healthy-seed0-clusters.csv \
    --sep , \
    --simulation-dt 0.01 \
    --n-days 60 \
    --output-basepath ../output/keystoneness/clusters

python ../keystoneness.py \
    --input ../output/mdsine2/healthy-seed1/mcmc.pkl \
    --study ../output/mdsine2/healthy-seed1/subjset.pkl \
    --leave-out-table ../output/keystoneness/tables/healthy-seed1-taxa.csv \
    --sep , \
    --simulation-dt 0.01 \
    --n-days 60 \
    --output-basepath ../output/keystoneness/taxa
python ../keystoneness.py \
    --input ../output/mdsine2/healthy-seed1/mcmc.pkl \
    --study ../output/mdsine2/healthy-seed1/subjset.pkl \
    --leave-out-table ../output/keystoneness/tables/healthy-seed1-clusters.csv \
    --sep , \
    --simulation-dt 0.01 \
    --n-days 60 \
    --output-basepath ../output/keystoneness/clusters

# UC
python ../keystoneness.py \
    --input ../output/mdsine2/uc-seed0/mcmc.pkl \
    --study ../output/mdsine2/uc-seed0/subjset.pkl \
    --leave-out-table ../output/keystoneness/tables/uc-seed0-taxa.csv \
    --sep , \
    --simulation-dt 0.01 \
    --n-days 60 \
    --output-basepath ../output/keystoneness/taxa
python ../keystoneness.py \
    --input ../output/mdsine2/uc-seed0/mcmc.pkl \
    --study ../output/mdsine2/uc-seed0/subjset.pkl \
    --leave-out-table ../output/keystoneness/tables/uc-seed0-clusters.csv \
    --sep , \
    --simulation-dt 0.01 \
    --n-days 60 \
    --output-basepath ../output/keystoneness/clusters

python ../keystoneness.py \
    --input ../output/mdsine2/uc-seed1/mcmc.pkl \
    --study ../output/mdsine2/uc-seed1/subjset.pkl \
    --leave-out-table ../output/keystoneness/tables/uc-seed1-taxa.csv \
    --sep , \
    --simulation-dt 0.01 \
    --n-days 60 \
    --output-basepath ../output/keystoneness/taxa
python ../keystoneness.py \
    --input ../output/mdsine2/uc-seed1/mcmc.pkl \
    --study ../output/mdsine2/uc-seed1/subjset.pkl \
    --leave-out-table ../output/keystoneness/tables/uc-seed1-clusters.csv \
    --sep , \
    --simulation-dt 0.01 \
    --n-days 60 \
    --output-basepath ../output/keystoneness/clusters