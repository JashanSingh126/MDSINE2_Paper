export OUT_DIR="files/gibson"
export TAX_OUT_DIR="${OUT_DIR}/taxonomy"
export NEGBIN_OUT_DIR="${OUT_DIR}/negbin"
export MDSINE_OUT_DIR="${OUT_DIR}/mdsine2"
export MDSINE_FIXED_CLUSTER_OUT_DIR="${OUT_DIR}/mdsine2_fixed_clustering"
export PLOTS_OUT_DIR="${OUT_DIR}/plots"

export DOWNSTREAM_ANALYSIS_OUT_DIR="${OUT_DIR}/downstream_analysis"

export DATASET_DIR="../datasets/gibson"
export MDSINE2_LOG_INI="gibson_inference/log_config.ini"


# ============== Cross-Validation Settings
export CV_BASEDIR="${OUT_DIR}/cv"
export CV_DATASET_PATH="${CV_BASEDIR}/dataset"  # training/test datasets
export CV_OUT_DIR="${CV_BASEDIR}/mdsine2"
export CV_FWSIM_OUT_DIR="${CV_BASEDIR}/forward_sims"
