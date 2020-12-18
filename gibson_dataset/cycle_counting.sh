#!/bin/bash

HEALTHY_UNFIXED_MCMC="../output/mdsine2/healthy-seed0/mcmc.pkl"
HEALTHY_UNFIXED_OUTDIR="../output/postprocessing/cycles/unfixed_clustering/healthy"
UC_UNFIXED_MCMC="../output/mdsine2/uc-seed0/mcmc.pkl"
UC_UNFIXED_OUTDIR="../output/postprocessing/cycles/unfixed_clustering/uc"
HEALTHY_FIXED_MCMC="../output/mdsine2/fixed_clustering/healthy/mcmc.pkl"
HEALTHY_FIXED_OUTDIR="../output/postprocessing/cycles/fixed_clustering/healthy"
UC_FIXED_MCMC="../output/mdsine2/fixed_clustering/uc/mcmc.pkl"
UC_FIXED_OUTDIR="../output/postprocessing/cycles/fixed_clustering/uc"

python ../cycle_count_otu.py \
    --mcmc_path $HEALTHY_UNFIXED_MCMC \
    --out_dir $HEALTHY_UNFIXED_OUTDIR \
    --max_path_len 4

python ../cycle_count_otu.py \
    --mcmc_path $UC_UNFIXED_MCMC \
    --out_dir $UC_UNFIXED_OUTDIR \
    --max_path_len 4

python ../cycle_count_cluster.py \
  --mcmc_path $HEALTHY_FIXED_MCMC \
  --out_dir $HEALTHY_FIXED_OUTDIR \
  --max_path_len 4

python ../cycle_count_cluster.py \
  --mcmc_path $UC_FIXED_MCMC \
  --out_dir $UC_FIXED_OUTDIR \
  --max_path_len 4
