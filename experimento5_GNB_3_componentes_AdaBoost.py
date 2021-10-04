# -*- coding: utf-8 -*-
"""
Neste experimento são utilizadas a base DS1 para treinamento e DS2 para teste e duas componentes ocultas da abordagem proposta.
"""
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import config
import tools
import measures_performance

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
features0_train = tools.rectify(np.loadtxt(config.DIR_FILES + '3componentes/' + config.FILE_FEATURES_TRAINING + '_' + config.DS_TRAINING + '_' + len_cycle_tra + '_0.txt'))
features1_train = tools.rectify(np.loadtxt(config.DIR_FILES + '3componentes/' + config.FILE_FEATURES_TRAINING + '_' + config.DS_TRAINING + '_' + len_cycle_tra + '_1.txt'))
features2_train = tools.rectify(np.loadtxt(config.DIR_FILES + '3componentes/' + config.FILE_FEATURES_TRAINING + '_' + config.DS_TRAINING + '_' + len_cycle_tra + '_2.txt'))
targets_train   = np.loadtxt(config.DIR_FILES + config.FILE_TARGET_TRAINING + '_' + config.DS_TRAINING + '_' + len_cycle_tra + '.txt')

features0_train = LinearDiscriminantAnalysis().fit_transform(features0_train, targets_train)
features1_train = LinearDiscriminantAnalysis().fit_transform(features1_train, targets_train)
features2_train = LinearDiscriminantAnalysis().fit_transform(features2_train, targets_train)

'''
Carrega os dados de teste
'''
features0_test = tools.rectify(np.loadtxt(config.DIR_FILES + '3componentes/' + config.FILE_FEATURES_TEST + '_' + config.DS_TEST + '_' + len_cycle_tes + '_0.txt'))
features1_test = tools.rectify(np.loadtxt(config.DIR_FILES + '3componentes/' + config.FILE_FEATURES_TEST + '_' + config.DS_TEST + '_' + len_cycle_tes + '_1.txt'))
features2_test = tools.rectify(np.loadtxt(config.DIR_FILES + '3componentes/' + config.FILE_FEATURES_TEST + '_' + config.DS_TEST + '_' + len_cycle_tes + '_2.txt'))
targets_test   = np.loadtxt(config.DIR_FILES + config.FILE_TARGET_TEST + '_' + config.DS_TEST + '_' + len_cycle_tes + '.txt')

features0_test = LinearDiscriminantAnalysis().fit_transform(features0_test, targets_test)
features1_test = LinearDiscriminantAnalysis().fit_transform(features1_test, targets_test)
features2_test = LinearDiscriminantAnalysis().fit_transform(features2_test, targets_test)

'''
Treinamento dos Classificadores
'''
clf0 = AdaBoostClassifier(base_estimator=GaussianNB(), n_estimators=3, random_state=0)
clf1 = AdaBoostClassifier(base_estimator=GaussianNB(), n_estimators=3, random_state=0)
clf2 = AdaBoostClassifier(base_estimator=GaussianNB(), n_estimators=3, random_state=0)

clf0.fit(features0_train, targets_train)
clf1.fit(features1_train, targets_train)
clf2.fit(features2_train, targets_train)
    
predicted_values0 = clf0.predict(features0_test)
predicted_values1 = clf1.predict(features1_test)
predicted_values2 = clf2.predict(features2_test)

'''
Computa as medidas de performance para cada classificador
'''
M0 = measures_performance.every(predicted_values0, targets_test, config.POSITIVE_CLASS, config.NEGATIVE_CLASS)
M1 = measures_performance.every(predicted_values1, targets_test, config.POSITIVE_CLASS, config.NEGATIVE_CLASS)
M2 = measures_performance.every(predicted_values2, targets_test, config.POSITIVE_CLASS, config.NEGATIVE_CLASS)


M = [M0, M1, M2]

print('Performances considerando duas máquinas distintas (três componentes)')

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

'''
Agrega as duas componentes
''' 
features_train = []
for i in range(0, features0_train.shape[0]):
    features_train.append(np.concatenate((features0_train[i,:], features1_train[i,:], features2_train[i,:])))

features_train = np.array(features_train)

del features0_train, features1_train, features2_train

features_test = []
for i in range(0, features0_test.shape[0]):
    features_test.append(np.concatenate((features0_test[i,:], features1_test[i,:], features2_test[i,:])))

features_test = np.array(features_test)

del features0_test, features1_test, features2_test


print('Performances considerando componentes agregadas')

'''
Treinamento dos Classificadores
'''
clf = AdaBoostClassifier(base_estimator=GaussianNB(), n_estimators=3, random_state=0)
clf.fit(features_train, targets_train)
predicted_values = clf.predict(features_test)

m = measures_performance.every(predicted_values, targets_test, config.POSITIVE_CLASS, config.NEGATIVE_CLASS)

print(' Acc: ', round(m[0], 4))
print(' F+:  ', round(m[1], 4))
print(' F-:  ', round(m[2], 4))
print(' Se:  ', round(m[3], 4))
print(' Sp:  ', round(m[4], 4))
print(' P+:  ', round(m[5], 4))
print(' P-:  ', round(m[6], 4))
