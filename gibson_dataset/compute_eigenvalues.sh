#!/bin/bash

HEALTHY_MCMC="../output/mdsine2/healthy-seed0/mcmc.pkl"
UC_MCMC="../output/mdsine2/uc-seed0/mcmc.pkl"
OUTPATH="../output/postprocessing"

python ../compute_eigenvalues.py \
    --healthy $HEALTHY_MCMC \
    --uc $UC_MCMC \
    --out_dir $OUTPATH
