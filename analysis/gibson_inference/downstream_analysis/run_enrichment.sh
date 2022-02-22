set -e
source gibson_inference/settings.sh

python helpers/run_enrichment.py \
    -m_loc "${MDSINE_FIXED_CLUSTER_OUT_DIR}/healthy/mcmc.pkl"\
    -l "family" \
    -o_loc "${OUT_DIR}/enrichment"\
    -o_name "healthy_family"

python helpers/run_enrichment.py \
    -m_loc "${MDSINE_FIXED_CLUSTER_OUT_DIR}/uc/mcmc.pkl"\
    -l "family" \
    -o_loc "${OUT_DIR}/enrichment"\
    -o_name "uc_family"

python helpers/run_enrichment.py \
    -m_loc "${MDSINE_FIXED_CLUSTER_OUT_DIR}/healthy/mcmc.pkl"\
    -l "class" \
    -o_loc "${OUT_DIR}/enrichment"\
    -o_name "healthy_class"

python helpers/run_enrichment.py \
    -m_loc "${MDSINE_FIXED_CLUSTER_OUT_DIR}/uc/mcmc.pkl"\
    -l "class" \
    -o_loc "${OUT_DIR}/enrichment"\
    -o_name "uc_class"

python helpers/run_enrichment.py \
    -m_loc "${MDSINE_FIXED_CLUSTER_OUT_DIR}/healthy/mcmc.pkl"\
    -l "order" \
    -o_loc "${OUT_DIR}/enrichment"\
    -o_name "healthy_order"

python helpers/run_enrichment.py \
    -m_loc "${MDSINE_FIXED_CLUSTER_OUT_DIR}/uc/mcmc.pkl"\
    -l "order" \
    -o_loc "${OUT_DIR}/enrichment"\
    -o_name "uc_order"

python helpers/run_enrichment.py \
    -m_loc "${MDSINE_FIXED_CLUSTER_OUT_DIR}/healthy/mcmc.pkl"\
    -l "phylum" \
    -o_loc "${OUT_DIR}/enrichment"\
    -o_name "healthy_phylum"

python helpers/run_enrichment.py \
    -m_loc "${MDSINE_FIXED_CLUSTER_OUT_DIR}/uc/mcmc.pkl"\
    -l "phylum" \
    -o_loc "${OUT_DIR}/enrichment"\
    -o_name "uc_phylum"
