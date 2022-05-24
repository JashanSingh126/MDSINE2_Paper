#!/bin/bash
#submits an lsf job to run programs that generate semi-synthetic data 
set -e
source semisynthetic_experiments/settings.sh

mkdir -p "${LSF_DIR}"
mkdir -p "${LOG_DIR}"

seed=0
num_seeds=1
for (( i = 0; i < ${num_seeds}; i += 1 )); do
	seed=$((seed+3))  # increment the seed.
	SAMPLE_LSF_PATH=${LSF_DIR}/sample_seed_${seed}.lsf

	# ============ Create LSF ===========
	echo "Creating ${SAMPLE_LSF_PATH}"
	cat <<- EOFDOC > $SAMPLE_LSF_PATH
#!/bin/bash
#BSUB -J semisynthetic_seed_${seed}
#BSUB -o ${LOG_DIR}/sample_seed_${seed}_JOB_%J.out
#BSUB -e ${LOG_DIR}/sample_seed_${seed}_JOB_%J.err
#BSUB -q ${LSF_QUEUE}
#BSUB -n ${LSF_N_CORES}
#BSUB -M ${LSF_MEM}
#BSUB -R rusage[mem=${LSF_MEM}]

source activate ${CONDA_ENV}
cd ${ANALYSIS_DIR}
echo "Current loc"
pwd
bash ${BASE_DIR}/scripts/run_semisynthetic.sh $seed

EOFDOC
done

for f in ${LSF_DIR}/*; do
	echo "Submitting job in file $f"
	bsub < $f
done