#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 19:49:09 2019

@author: vbucci
"""
import numpy as np
import numpy.random as npr
import sys

import linear_stability as ls

linear_stability = ls.make_synthetic_stability_class('premade_50')
linear_stability.perform_combinatorial_cluster_stability(output_folder='~/Downloads', n=2)

linear_stability.calculate_keystoneness(n=3)
