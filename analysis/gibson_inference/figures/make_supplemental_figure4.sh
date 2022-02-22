#!/bin/bash
#equivalent to supplemental figure 4

set -e
source gibson_inference/settings.sh

#Plot supplemental figure4; asssumes that other files are downloaded from zenodo. If the files are not present,
# uncomment the code from the line 11 to 15 and run it. It should download the necessary
#files from zenodo. Once downloaded, comment the code again to avoid re-downloading the files.

#cd output/gibson
#mkdir zenodo
#cd zenodo
#wget https://zenodo.org/record/5781848/files/other_files.tgz
#tar -xzvf "other_files.tgz"

#Plot figure3

python gibson_inference/figures/supplemental_figure4.py \
--mdsine_path "output/gibson/forward_sim/"\
--clv_elas_path "output/gibson/zenodo/clv_results/results_rel_elastic/"\
--clv_ridge_path "output/gibson/zenodo/clv_results/results_rel_ridge/"\
--glv_elas_path "output/gibson/zenodo/clv_results/results_abs_elastic/"\
--glv_ridge_path "output/gibson/zenodo/clv_results/results_abs_ridge/forward_sims_abs_ridge/"\
--output_path "${PLOTS_OUT_DIR}/"
