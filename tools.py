#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Funções úteis
"""

#import numpy as np
#import config
from sklearn.preprocessing import scale
from sklearn.impute import SimpleImputer
import numpy as np

def rectify(data):
    '''
    Remove valores perdidos, NaN, e Escalona os dados
    '''
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    return imp.fit_transform(data)
    #return scale(imp.fit_transform(data))

'''
Para calcular outliers
'''
def outliers(features, cut_off_rate = 3, remove = False):
    outliers = []
    for k in range(0, features.T.shape[0]):
        data = features.T[k,:]
        data_mean, data_std = np.mean(data), np.std(data)
        cut_off = data_std * cut_off_rate
        lower, upper = data_mean - cut_off, data_mean + cut_off
        outliers.append(len([x for x in data if x < lower or x > upper]))
    
    return [outliers, round(lower, 4), round(upper, 4)]

'''
def outliers_remove(data, lower, upper):
    return [x for x in data if x >= lower and x <= upper]
'''