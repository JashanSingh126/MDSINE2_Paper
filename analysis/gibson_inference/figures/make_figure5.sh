#!/bin/bash

#Plot figure3
python gibson_inference/figures/figure5.py \
    -file1 "/data/cctm/darpa_perturbation_mouse_study/sawal_test/coarsening/healthy_seed0_mixed_uc_seed0_mixed/distance.csv" \
    -file2 "/data/cctm/darpa_perturbation_mouse_study/sawal_test/coarsening/healthy_seed0_mixed_uc_seed0_mixed/arithmetic_mean_data.csv" \
    -file3 "/data/cctm/darpa_perturbation_mouse_study/sawal_test/coarsening/healthy_seed0_mixed_uc_seed0_mixed/arithmetic_mean_null_all.csv"
