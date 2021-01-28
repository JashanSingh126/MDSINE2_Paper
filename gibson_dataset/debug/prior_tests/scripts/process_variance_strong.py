import mdsine2 as md2
from mdsine2.names import STRNAMES
import default
import logging


def load_settings(cfg: md2.config.MDSINE2ModelConfig, study: md2.Study, interaction_prior: str, perturbation_prior: str):
    n_taxa = len(study.taxa)

    # ====== Negative binomial params ======
    negbin_a0 = 1e-2
    negbin_a1 = 1e-4
    cfg.set_negbin_params(negbin_a0, negbin_a1)

    # Set the sparsities
    cfg.INITIALIZATION_KWARGS[STRNAMES.CLUSTER_INTERACTION_INDICATOR_PROB]['hyperparam_option'] = interaction_prior
    cfg.INITIALIZATION_KWARGS[STRNAMES.PERT_INDICATOR_PROB]['hyperparam_option'] = perturbation_prior

    # Change the cluster initialization to no clustering if there are less than 30 taxa
    if n_taxa <= 30:
        logging.info('Since there are fewer than 30 taxa, we set the initialization of the clustering to `no-clusters`')
        cfg.INITIALIZATION_KWARGS[STRNAMES.CLUSTERING]['value_option'] = 'no-clusters'

    # ====== Initializations of inference variables ======
    # Process Variance
    cfg.INITIALIZATION_KWARGS[STRNAMES.PROCESSVAR] = {
        'dof_option': 'half',  # Reminder: 'half' indicates the usage of len(G.data.LHS) to determine the DOF.
        'scale_option': 'manual',
        'scale': 0.5,
        'value_option': 'prior-mean',
        'delay': 0
    }

    return cfg


if __name__ == "__main__":
    default.load_settings = load_settings
    default.main()
