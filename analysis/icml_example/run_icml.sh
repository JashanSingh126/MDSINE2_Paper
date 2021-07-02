#!/bin/bash

export OUT_DIR="files/synthetic"


python icml_example/make_icml.py \
    --output-basepath ${OUT_DIR}/icml_input.pkl

python helpers/step_5_infer_mdsine2.py \
    --input ${OUT_DIR}/icml_input.pkl \
    --negbin 0.00025 0.0025 \
    --seed 0 \
    --burnin 100 \
    --n-samples 900 \
    --checkpoint 100 \
    --multiprocessing 0 \
    --basepath ${OUT_DIR}/mdsine2 \
    --interaction-ind-prior strong-sparse \
    --perturbation-ind-prior strong-sparse

python helpers/step_6_visualize_mdsine2.py \
    --chain  ${OUT_DIR}/mdsine2/icml/mcmc.pkl \
    --output-basepath ${OUT_DIR}/mdsine2/icml/posterior