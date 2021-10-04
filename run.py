# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 09:36:26 2016

@author: bruno
"""
import sys
sys.path.append('/home/bruno/PESQUISAS/Python')

import features_geo
import gen_samples_cardiac_cycle
import classify
import config


run_gen_samples = True
run_features    = True

print '-----------------------------------------------------------'

#gera as amostras
if run_gen_samples:
    gen_samples_cardiac_cycle.run(config.DS_TRAINING, config.DS_TEST)

#calcula as características
if run_features:
    features_geo.run(config.DS_TRAINING, config.DS_TEST)

#treina e classifica os dados
classify.run(config.DS_TRAINING, config.DS_TEST)

