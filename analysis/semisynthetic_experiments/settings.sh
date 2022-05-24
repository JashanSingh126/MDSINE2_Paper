export ANALYSIS_DIR="/data/cctm/darpa_perturbation_mouse_study/sawal_test/MDSINE2_Paper/analysis"
export BASE_DIR="${ANALYSIS_DIR}/semisynthetic_experiments"
export OUTPUT_DIR="output/semisynthetic_experiments"
export MCMC_DIR="/data/cctm/darpa_perturbation_mouse_study/MDSINE2_Paper/output/mdsine2/fixed_clustering_mixed_prior/healthy-seed0-mixed"

export CONDA_ENV="mdsine2"
export LSF_DIR="${BASE_DIR}/lsf"
export LSF_QUEUE="gpu"
export LSF_N_CORES=1
export LSF_MEM=16000

export LOG_DIR="${OUTPUT_DIR}/logs"

export PROCESS_VAR=0.01
export SIMULATION_DT=0.01
export NEGBIN_A0_LOW=1e-6
export NEGBIN_A1_LOW=3e-6
export NEGBIN_A0_MEDIUM=1e-5
export NEGBIN_A1_MEDIUM=1.5e-2
export NEGBIN_A0_HIGH=1e-4
export NEGBIN_A1_HIGH=9e-2
export READ_DEPTH=75000
export LOW_NOISE_SCALE=0.01
export MEDIUM_NOISE_SCALE=0.15
export HIGH_NOISE_SCALE=0.3

require_arg()
{
	var_name=$1
	var_value=$2
	if [ -z "$var_value" ]
	then
		echo "var \"${var_name}\" is empty"
		exit 1
	fi
}

export require_arg
