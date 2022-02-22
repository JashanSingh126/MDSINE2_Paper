#!/bin/bash

set -e
source gibson_inference/settings.sh

#Plot figure3; asssumes that other files are downloaded from zenodo. If the files are not present,
#following, uncomment the code from the line 10 to 14 and run it. It should download the necessary
#files from zenodo. Once downloaded, comment the code again to avoid re-downloading the files.

#cd output/gibson
#mkdir zenodo
#cd zenodo
#wget https://zenodo.org/record/5781848/files/other_files.tgz
#tar -xzvf "other_files.tgz"

python gibson_inference/figures/figure3.py \
    --mdsine_path "output/gibson/forward_sim/"\
    --clv_elas_path "output/gibson/zenodo/clv_results/results_rel_elastic/"\
    --clv_ridge_path "output/gibson/zenodo/clv_results/results_rel_ridge/"\
    --glv_elas_path "output/gibson/zenodo/clv_results/results_abs_elastic/"\
    --glv_ridge_path "output/gibson/zenodo/clv_results/results_abs_ridge/forward_sims_abs_ridge/"\
    --output_path "${PLOTS_OUT_DIR}/"

