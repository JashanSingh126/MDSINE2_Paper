#!/bin/bash
set -e
source semisynthetic_cdiff/settings.sh


seed=0
num_seeds=10
for (( i = 0; i < ${num_seeds}; i += 1 )); do
	seed=$((seed+1))  # increment the seed.
	SAMPLE_LSF_PATH=${LSF_DIR}/sample_seed_${seed}.lsf

	# ============ Create LSF ===========
	echo "Creating ${SAMPLE_LSF_PATH}"
	cat <<- EOFDOC > $SAMPLE_LSF_PATH
#!/bin/bash
#BSUB -J sample_cdiff
#BSUB -o ${LOGDIR}/sample_seed_${seed}_JOB_%J.out
#BSUB -e ${LOGDIR}/sample_seed_${seed}_JOB_%J.err
#BSUB -q ${LSF_QUEUE}
#BSUB -n ${LSF_N_CORES}
#BSUB -M ${LSF_MEM}
#BSUB -R rusage[mem=${LSF_MEM}]

source activate ${CONDA_ENV}
cd ${ANALYSIS_DIR}
bash ${BASE_DIR}/scripts/sample.sh
EOFDOC
done


# Uncomment to run all.
for f in ${LSF_DIR}; do
	bsub < $f
done
