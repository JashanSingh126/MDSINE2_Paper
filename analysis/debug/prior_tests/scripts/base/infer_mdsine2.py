'''Run MDSINE2 inference

Author: Youn
A copy of step_5_infer_mdsine2.py for executing custom configurations.
'''
import mdsine2 as md2
import os
import time


def create_config(output_basepath: str,
                  negbin_a1: float,
                  negbin_a0: float,
                  seed: int,
                  burnin: int,
                  n_samples: int,
                  checkpoint: int,
                  multithreaded: bool=False
                  ) -> md2.config.MDSINE2ModelConfig:

    # Construct the param object
    cfg = md2.config.MDSINE2ModelConfig(
        basepath=output_basepath,
        seed=seed,
        burnin=burnin,
        n_samples=n_samples,
        negbin_a1=negbin_a1,
        negbin_a0=negbin_a0,
        checkpoint=checkpoint
    )

    # Run with multiprocessing if necessary
    if multithreaded:
        cfg.MP_FILTERING = 'full'
        cfg.MP_CLUSTERING = 'full-4'

    return cfg


def run_mdsine(params: md2.config.MDSINE2ModelConfig,
               study: md2.Study):
    mcmc = md2.initialize_graph(params=params, graph_name=study.name, subjset=study)
    mdata_fname = os.path.join(params.MODEL_PATH, 'metadata.txt')
    params.make_metadata_file(fname=mdata_fname)

    start_time = time.time()
    _ = md2.run_graph(mcmc, crash_if_error=True)

    # Record how much time inference took
    t = time.time() - start_time
    t = t / 3600  # Convert to hours

    f = open(mdata_fname, 'a')
    f.write('\n\nTime for inference: {} hours'.format(t))
    f.close()
