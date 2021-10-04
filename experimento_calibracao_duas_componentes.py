# -*- coding: utf-8 -*-
"""
Utiliza apenas duas compoenetes do AMUSE
"""
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import numpy as np
import config
################# PLOT #################
import matplotlib 
import matplotlib.pyplot as plt
matplotlib.rc('xtick', labelsize=9) 
matplotlib.rc('ytick', labelsize=9)
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

'''
Carrega os dados de treinamento
'''
features0_train = np.loadtxt(config.DIR_FILES + config.FILE_FEATURES_TRAINING + '_' + config.DS_TRAINING + '_0.txt')
features1_train = np.loadtxt(config.DIR_FILES + config.FILE_FEATURES_TRAINING + '_' + config.DS_TRAINING + '_1.txt')
targets_train   = np.loadtxt(config.DIR_FILES + config.FILE_TARGET_TRAINING + '_' + config.DS_TRAINING + '.txt')

'''
Carrega os dados de teste
'''
features0_test = np.loadtxt(config.DIR_FILES + config.FILE_FEATURES_TEST + '_' + config.DS_TEST + '_0.txt')
features1_test = np.loadtxt(config.DIR_FILES + config.FILE_FEATURES_TEST + '_' + config.DS_TEST + '_1.txt')
targets_test   = np.loadtxt(config.DIR_FILES + config.FILE_TARGET_TEST + '_' + config.DS_TEST + '.txt')

'''
Treinamento dos Classificadores
'''
clf0 = GaussianNB()
clf1 = GaussianNB()

'''
Ajuste dos classificadores
'''
clf0.fit(features0_train, targets_train)
clf1.fit(features1_train, targets_train)

'''
Previsao dos valores reais das classes, os quais são 0 ou 1
'''
predicted_values0_proba = clf0.predict_proba(features0_test)
predicted_values1_proba = clf1.predict_proba(features1_test)

'''
Curva do Diagrama de confiabilidade
'''
fop0, mpv0 = calibration_curve(targets_test, predicted_values0_proba[:,1], n_bins=10)
fop1, mpv1 = calibration_curve(targets_test, predicted_values1_proba[:,1], n_bins=10)

'''
Calibração
'''
calibrator0 = CalibratedClassifierCV(clf0, method='sigmoid', cv=3)
calibrator0.fit(features0_train, targets_train)
predicted_values0_proba_cal = calibrator0.predict_proba(features0_test)

calibrator1 = CalibratedClassifierCV(clf1, method='sigmoid', cv=3)
calibrator1.fit(features1_train, targets_train)
predicted_values1_proba_cal = calibrator1.predict_proba(features1_test)

'''
Curva do Diagrama de confiabilidade, após calibração
'''
fop0_cal, mpv0_cal = calibration_curve(targets_test, predicted_values0_proba_cal[:,1], n_bins=10)
fop1_cal, mpv1_cal = calibration_curve(targets_test, predicted_values1_proba_cal[:,1], n_bins=10)

'''
Diagrama de confiabilidade
'''
plt.figure(1)
plt.subplot(121)
plt.plot([0, 1], [0, 1], linestyle='--', color='k')
plt.plot(mpv0, fop0, marker='.', color='r')
plt.plot(mpv0_cal, fop0_cal, marker='.', color='g')
plt.title('(a)')
plt.xlabel('Probabilidades previstas')
plt.ylabel('Frequência observada')
plt.yticks([0.00, 0.25, 0.50, 0.75, 1.00])
plt.subplot(122)
plt.plot([0, 1], [0, 1], linestyle='--', color='k')
plt.plot(mpv1, fop1, marker='.', color='r')
plt.plot(mpv1_cal, fop1_cal, marker='.', color='g')
plt.title('(b)')
plt.xlabel('Probabilidades previstas')
plt.ylabel('Frequência observada')
plt.yticks([0.00, 0.25, 0.50, 0.75, 1.00])
plt.tight_layout()
plt.show()

del clf0, clf1
del features0_train, features1_train
del features0_test, features1_test
del targets_test, targets_train

