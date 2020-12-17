#!/bin/bash

HEALTHY_UNFIXED_MCMC=TODO
HEALTHY_UNFIXED_OUTDIR=TODO
UC_UNFIXED_MCMC=TODO
UC_UNFIXED_OUTDIR=TODO
HEALTHY_FIXED_MCMC=TODO
HEALTHY_FIXED_OUTDIR=TODO
UC_FIXED_MCMC=TODO
UC_FIXED_OUTDIR=TODO

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
