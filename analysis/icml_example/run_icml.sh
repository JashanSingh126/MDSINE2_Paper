#!/bin/bash

export OUT_DIR="output/icml"


python icml_example/make_icml.py \
    --output-basepath ${OUT_DIR}/icml.pkl

python helpers/step_5_infer_mdsine2.py \
    --input ${OUT_DIR}/icml.pkl \
    --basepath ${OUT_DIR}/mdsine2 \
    --negbin 0.00025 0.0025 \
    --seed 0 \
    --burnin 100 \
    --n-samples 900 \
    --checkpoint 100 \
    --multiprocessing 0 \
    --interaction-ind-prior strong-sparse \
    --perturbation-ind-prior strong-sparse

python helpers/step_6_visualize_mdsine2.py \
    --chain  ${OUT_DIR}/mdsine2/mcmc.pkl \
    --output-basepath ${OUT_DIR}/mdsine2/posterior
