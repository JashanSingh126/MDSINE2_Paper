#!/bin/bash

python scripts/make_icml.py \
    --output-basepath ../processed_data/synthetic/icml.pkl

python ../step_5_infer_mdsine2.py \
    --input ../processed_data/synthetic/icml.pkl \
    --negbin 0.00025 0.0025 \
    --seed 0 \
    --burnin 100 \
    --n-samples 900 \
    --checkpoint 100 \
    --multiprocessing 0 \
    --basepath ../output/mdsine2/synthetic \
    --interaction-ind-prior strong-sparse \
    --perturbation-ind-prior strong-sparse
python ../step_6_visualize_mdsine2.py \
    --chain  ../output/mdsine2/synthetic/icml/mcmc.pkl \
    --output-basepath ../output/mdsine2/synthetic/icml/posterior