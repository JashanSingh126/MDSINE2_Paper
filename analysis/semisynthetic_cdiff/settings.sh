export ANALYSIS_DIR="."  # Replace with absolute path!
export BASE_DIR="${ANALYSIS_DIR}/semisynthetic_cdiff"
export OUTPUT_DIR="output/semisynthetic_cdiff"

export CONDA_ENV="mdsine2"
export LSF_DIR="${BASE_DIR}/lsf"
export LSF_QUEUE=""
export LSF_N_CORES=1
export LSF_MEM=16000

export LOGDIR="${OUTPUT_DIR}/logs"
export MDSINE_BVS_PATH=${BASE_DIR}/files/BVS.mat

export COHORT_SIZE=5
export PROCESS_VAR=0.01
export SIMULATION_DT=0.01
export NEGBIN_A0=1e-10
export NEGBIN_A1=0.05
export READ_DEPTH=50000
export LOW_NOISE_SCALE=0.01
export MEDIUM_NOISE_SCALE=0.1
export HIGH_NOISE_SCALE=0.2


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
