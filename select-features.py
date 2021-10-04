#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 06:23:02 2018

@author: bruno
"""

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np
import config
import pylab as plt

features_training = np.loadtxt(config.DIR_FILES + config.FILE_FEATURES_TRAINING + '_' + config.DS_TRAINING + '.txt')
targets_training  = np.loadtxt(config.DIR_FILES + config.FILE_TARGET_TRAINING + '_' + config.DS_TRAINING + '.txt')

new_features = SelectKBest(chi2, k=3)
new_features.fit(features_training, targets_training)

labels   = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11', 'a12']
scores   = np.array(new_features.scores_)
p_values = np.array(new_features.pvalues_)

sort_scores = np.flip(np.sort(scores), axis=0)
for sc in sort_scores:
    idx = np.where(scores == sc)[0][0]
    print '{:>3} - score: {:1.4e}, p-value: {:1.4e}'.format(labels[idx], sc, p_values[idx])