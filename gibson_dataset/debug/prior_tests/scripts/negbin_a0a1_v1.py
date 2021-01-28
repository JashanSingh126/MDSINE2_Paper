import mdsine2 as md2
from mdsine2.names import STRNAMES
import default
import logging


def load_settings(cfg: md2.config.MDSINE2ModelConfig, study: md2.Study, interaction_prior: str, perturbation_prior: str):
    n_taxa = len(study.taxa)

    # ====== Negative binomial params ======
    negbin_a0 = 0.
    negbin_a1 = 0.
    cfg.set_negbin_params(negbin_a0, negbin_a1)

    # Set the sparsities
    cfg.INITIALIZATION_KWARGS[STRNAMES.CLUSTER_INTERACTION_INDICATOR_PROB]['hyperparam_option'] = interaction_prior
    cfg.INITIALIZATION_KWARGS[STRNAMES.PERT_INDICATOR_PROB]['hyperparam_option'] = perturbation_prior

    # Change the cluster initialization to no clustering if there are less than 30 taxa
    if n_taxa <= 30:
        logging.info('Since there are fewer than 30 taxa, we set the initialization of the clustering to `no-clusters`')
        cfg.INITIALIZATION_KWARGS[STRNAMES.CLUSTERING]['value_option'] = 'no-clusters'

    return cfg


if __name__ == "__main__":
    default.load_settings = load_settings
    default.main()
