# -*- coding: utf-8 -*-
"""
Neste experimento são utilizadas a base DS1 para treinamento e DS2 para teste e duas componentes ocultas da abordagem proposta.
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import config
import tools
import timeit

len_cycle_tra = str(config.LEN_CARDIAC_CYCLE(config.DB_TRAINING))
len_cycle_tes = str(config.LEN_CARDIAC_CYCLE(config.DB_TEST))

'''
Carrega os dados de treinamento
'''
features0_train = tools.rectify(np.loadtxt(config.DIR_FILES + '2componentes/' + config.FILE_FEATURES_TRAINING + '_' + config.DS_TRAINING + '_' + len_cycle_tra + '_0.txt'))
features1_train = tools.rectify(np.loadtxt(config.DIR_FILES + '2componentes/' + config.FILE_FEATURES_TRAINING + '_' + config.DS_TRAINING + '_' + len_cycle_tra + '_1.txt'))
targets_train   = np.loadtxt(config.DIR_FILES + config.FILE_TARGET_TRAINING + '_' + config.DS_TRAINING + '_' + len_cycle_tra + '.txt')

features0_train = LinearDiscriminantAnalysis().fit_transform(features0_train, targets_train)
features1_train = LinearDiscriminantAnalysis().fit_transform(features1_train, targets_train)

'''
Carrega os dados de teste
'''
features0_test = tools.rectify(np.loadtxt(config.DIR_FILES + '2componentes/' + config.FILE_FEATURES_TEST + '_' + config.DS_TEST + '_' + len_cycle_tes + '_0.txt'))
features1_test = tools.rectify(np.loadtxt(config.DIR_FILES + '2componentes/' + config.FILE_FEATURES_TEST + '_' + config.DS_TEST + '_' + len_cycle_tes + '_1.txt'))
targets_test   = np.loadtxt(config.DIR_FILES + config.FILE_TARGET_TEST + '_' + config.DS_TEST + '_' + len_cycle_tes + '.txt')

features0_test = LinearDiscriminantAnalysis().fit_transform(features0_test, targets_test)
features1_test = LinearDiscriminantAnalysis().fit_transform(features1_test, targets_test)

'''
Treinamento dos Classificadores
'''
clf0_KNN = KNeighborsClassifier()
clf1_KNN = KNeighborsClassifier()

clf0_GNB = GaussianNB()
clf1_GNB = GaussianNB()


'''
Para excutar no prompt python3 -m memory_profiler experimento8_GNB_2_componentes_tempo_memoria.py
'''
@profile
def compute_memory_time():   
	start_KNN = timeit.default_timer()

	clf0_KNN.fit(features0_train, targets_train)
	clf1_KNN.fit(features1_train, targets_train) 
	predicted_values0 = clf0_KNN.predict(features0_test)
	predicted_values1 = clf1_KNN.predict(features1_test)

	stop_KNN = timeit.default_timer()

	start_GNB = timeit.default_timer()

	clf0_GNB.fit(features0_train, targets_train)
	clf1_GNB.fit(features1_train, targets_train) 
	predicted_values0 = clf0_GNB.predict(features0_test)
	predicted_values1 = clf1_GNB.predict(features1_test)

	stop_GNB = timeit.default_timer()

	print('Time KNN: ', stop_KNN - start_KNN)
	print('Time GNB: ', stop_GNB - start_GNB)

if __name__ == '__main__':
	compute_memory_time()
