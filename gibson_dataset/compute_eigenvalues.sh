#!/bin/bash

HEALTHY_MCMC=TODO
UC_MCMC=TODO
OUTPATH=TODO

python ../compute_eigenvalues.py \
    --healthy $HEALTHY_MCMC \
    --uc $UC_MCMC \
    --out_path $OUTPATH
