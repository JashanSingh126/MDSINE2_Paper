#!/bin/bash

set -e
source gibson_inference/settings.sh

#HEALTHY_UNFIXED_MCMC="${MDSINE_OUT_DIR}/healthy-seed0/mcmc.pkl"
#HEALTHY_UNFIXED_OUTDIR="${DOWNSTREAM_ANALYSIS_OUT_DIR}/cycles/unfixed_clustering/healthy"
#UC_UNFIXED_MCMC="${MDSINE_OUT_DIR}/uc-seed0/mcmc.pkl"
#UC_UNFIXED_OUTDIR="${DOWNSTREAM_ANALYSIS_OUT_DIR}/cycles/unfixed_clustering/uc"

HEALTHY_FIXED_MCMC="${MDSINE_FIXED_CLUSTER_OUT_DIR}/healthy/mcmc.pkl"
HEALTHY_FIXED_OUTDIR="${DOWNSTREAM_ANALYSIS_OUT_DIR}/cycles/fixed_clustering/healthy"

UC_FIXED_MCMC="${MDSINE_FIXED_CLUSTER_OUT_DIR}/uc/mcmc.pkl"
UC_FIXED_OUTDIR="${DOWNSTREAM_ANALYSIS_OUT_DIR}/cycles/fixed_clustering/uc"

#python helpers/cycle_count_otu.py \
#    --mcmc_path $HEALTHY_UNFIXED_MCMC \
#    --out_dir $HEALTHY_UNFIXED_OUTDIR \
#    --max_path_len 3
#python helpers/cycle_count_otu.py \
#    --mcmc_path $UC_UNFIXED_MCMC \
#    --out_dir $UC_UNFIXED_OUTDIR \
#    --max_path_len 3

python helpers/cycle_count_cluster.py \
  --mcmc_path $HEALTHY_FIXED_MCMC \
  --out_dir $HEALTHY_FIXED_OUTDIR \
  --max_path_len 3

python helpers/cycle_count_cluster.py \
  --mcmc_path $UC_FIXED_MCMC \
  --out_dir $UC_FIXED_OUTDIR \
  --max_path_len 3
