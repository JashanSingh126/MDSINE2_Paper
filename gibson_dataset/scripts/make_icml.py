'''Make simple ICML system
'''

import mdsine2 as md2
import argparse
import os
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage=__doc__)
    parser.add_argument('--output-basepath', '-o', type=str, dest='basepath',
        help='Where to save the output')
    args = parser.parse_args()

    synth = md2.synthetic.Synthetic(name='synth', seed=5)
    synth.icml_dynamics()
    synth.set_timepoints(times=np.arange(21))
    synth.set_subjects(['subj{}'.format(i+1) for i in range(5)])

    init_dist = md2.variables.Uniform(low=2,high=10)
    processvar = md2.model.MultiplicativeGlobal(value=0.05**2)
    synth.generate_trajectories(dt=0.01, init_dist=init_dist, processvar=processvar)

    study = synth.simulateMeasurementNoise(a0=0.00025, a1=0.0025, qpcr_noise_scale=0.05, 
            approx_read_depth=50000, name='icml')
    try:
        os.makedirs(args.basepath, exist_ok=True)
    except FileExistsError:
        logger.info('File already exists, no nothing')

    try:
        study.save(args.basepath)
    except FileExistsError:
        logger.warning('Overwriting')
        os.remove(args.basepath)
        study.save(args.basepath)