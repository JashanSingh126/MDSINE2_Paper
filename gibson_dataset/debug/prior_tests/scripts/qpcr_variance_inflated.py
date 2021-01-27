import mdsine2 as md2
from mdsine2.names import STRNAMES
from gibson_dataset.debug.prior_tests.scripts import default
import logging


def load_settings(cfg: md2.config.MDSINE2ModelConfig, study: md2.Study):
    n_taxa = len(study.taxa)

    # ====== Indicator priors ======
    interaction_prior = 'weak-agnostic'
    perturbation_prior = 'weak-agnostic'

    # Set the sparsities
    cfg.INITIALIZATION_KWARGS[STRNAMES.CLUSTER_INTERACTION_INDICATOR_PROB]['hyperparam_option'] = interaction_prior
    cfg.INITIALIZATION_KWARGS[STRNAMES.PERT_INDICATOR_PROB]['hyperparam_option'] = perturbation_prior

    # Change the cluster initialization to no clustering if there are less than 30 taxa
    if n_taxa <= 30:
        logging.info('Since there are fewer than 30 taxa, we set the initialization of the clustering to `no-clusters`')
        cfg.INITIALIZATION_KWARGS[STRNAMES.CLUSTERING]['value_option'] = 'no-clusters'

    # ====== Initializations of inference variables ======
    # GLV process variance
    cfg.INITIALIZATION_KWARGS[STRNAMES.QPCR_VARIANCES] = {
        'value_option': 'inflated',
        'inflated': 10
    }

    return cfg


if __name__ == "__main__":
    default.load_settings = load_settings
    default.main()
