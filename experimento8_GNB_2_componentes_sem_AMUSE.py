# -*- coding: utf-8 -*-
"""
Neste experimento são utilizadas a base DS1 para treinamento e DS2 para teste e duas componentes ocultas da abordagem proposta.
"""
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import config
import tools
import measures_performance
import AHP

def print_measures(TP, TN, FP, FN):   
    Acc, Pr_P, Pr_N, Se, Sp, F_P, F_N, Cm = measures_performance.calc(TP, TN, FP, FN)
    print(' Acc: ', round(Acc, 4))
    print(' F+:  ', round(F_P, 4))
    print(' F-:  ', round(F_N, 4))
    print(' Se:  ', round(Se, 4))
    print(' Sp:  ', round(Sp, 4))
    print(' P+:  ', round(Pr_P, 4))
    print(' P-:  ', round(Pr_N, 4))

len_cycle_tra = str(config.LEN_CARDIAC_CYCLE(config.DB_TRAINING))
len_cycle_tes = str(config.LEN_CARDIAC_CYCLE(config.DB_TEST))

'''
Carrega os dados de treinamento
'''
features_train = tools.rectify(np.loadtxt(config.DIR_FILES + config.FILE_SAMPLES_TRAINING + '_' + config.DS_TRAINING + '_' + len_cycle_tra + '.txt'))
targets_train   = np.loadtxt(config.DIR_FILES + config.FILE_TARGET_TRAINING + '_' + config.DS_TRAINING + '_' + len_cycle_tra + '.txt')

features_train = LinearDiscriminantAnalysis().fit_transform(features_train, targets_train)

'''
Carrega os dados de teste
'''
features_test = tools.rectify(np.loadtxt(config.DIR_FILES + config.FILE_SAMPLES_TEST + '_' + config.DS_TEST + '_' + len_cycle_tes + '.txt'))
targets_test   = np.loadtxt(config.DIR_FILES + config.FILE_TARGET_TEST + '_' + config.DS_TEST + '_' + len_cycle_tes + '.txt')

features_test = LinearDiscriminantAnalysis().fit_transform(features_test, targets_test)

'''
Treinamento dos Classificadores
'''
clf = GaussianNB()
    
clf.fit(features_train, targets_train)
    
predicted_values0 = clf.predict(features_test)

'''
Computa as medidas de performance para cada classificador
'''
M0 = measures_performance.every(predicted_values0, targets_test, config.POSITIVE_CLASS, config.NEGATIVE_CLASS)

'''
Experimento 1: Performance individual de cada classificador
'''

M = [M0]

print('######################## EXP1 #####################')
print('Performance individual para cada classificador')

i = 0
for m in M:
    print('Clf', i)
    print(' Acc: ', round(m[0], 4))
    print(' F+:  ', round(m[1], 4))
    print(' F-:  ', round(m[2], 4))
    print(' Se:  ', round(m[3], 4))
    print(' Sp:  ', round(m[4], 4))
    print(' P+:  ', round(m[5], 4))
    print(' P-:  ', round(m[6], 4))
    i += 1
