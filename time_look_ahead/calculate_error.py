import logging
import random
import time
import sys
import os
import shutil
import h5py
import warnings
import pickle
import numpy as np
import argparse
import time

import matplotlib.pyplot as plt
import seaborn as sns
import pylab as pl


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--basepath', type=float, dest='n_days',
        help='Maximum number of days to look ahead')
    parser.add_argument('--times-to-start-on', type=float, dest='times_to_start_on', nargs='+',
        help='comma separated times to start the forward simulation on')
    parser.add_argument('--input-basepaths', type=str, dest='input_basepath',
        help='The path to find the growth, interaction, and perturbation traces as wll as the subject')
    parser.add_argument('--validation-subject', type=str, dest='input',
        help='Pylab.base.Subject obejct we want to do the inference over')
    parser.add_argument('--output-basepath', type=str, dest='output_basepath',
        help='Where to save the outputs')
    parser.add_argument('--simulation-dt', type=float, dest='simulation_dt',
        help='Timesteps we go in during forward simulation', default=0.01)
    parser.add_argument('--max-posterior', type=int, dest='max_posterior',
        help='TESTING USE ONLY', default=None)

    args = parser.parse_args()
    return args