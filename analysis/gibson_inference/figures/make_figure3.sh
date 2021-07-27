#!/bin/bash

#Plot figure3
python gibson_inference/figures/figure3.py \
    --mdsine_path "/data/cctm/darpa_perturbation_mouse_study/sawal_test/forward_sim_incomplete/cv_mixed/forward_sims/"\
    --clv_elas_path "/data/cctm/darpa_perturbation_mouse_study/sawal_test/clv_final/results_rel_elastic/"\
    --clv_ridge_path "/data/cctm/darpa_perturbation_mouse_study/sawal_test/clv_final/results_rel_ridge/"\
    --glv_elas_path "/data/cctm/darpa_perturbation_mouse_study/sawal_test/clv_final/results_abs_elastic/"\
    --glv_ridge_path "/data/cctm/darpa_perturbation_mouse_study/sawal_test/clv_final/results_abs_ridge/"\
    --output_path "gibson_inference/figures/output_figures/"
