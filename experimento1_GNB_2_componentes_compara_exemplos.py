# -*- coding: utf-8 -*-
"""
Neste experimento são utilizadas a base DS1 para treinamento e DS2 para teste e duas componentes ocultas da abordagem proposta.

Calcula os acordos e desacordos entre os classificadores
"""
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import config
import tools

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
clf0 = GaussianNB()
clf1 = GaussianNB()

'''
Ajuste dos classificadores
'''
clf0.fit(features0_train, targets_train)
clf1.fit(features1_train, targets_train)
   
'''
Calcula os acordos e desacordos
'''
ACORDOS_P  = 0
ACORDOS_N  = 0
DESACORDOS = 0

ACORDOS_CORRETOS_P = 0
ACORDOS_CORRETOS_N = 0

DESACORDOS_CORRETOS_clf0 = 0
DESACORDOS_CORRETOS_clf1 = 0

for k in range(0, targets_test.shape[0]):

    #Predicao
    predicted_clf0 = clf0.predict(features0_test[k].reshape(-1, 1).T)
    predicted_clf1 = clf1.predict(features1_test[k].reshape(-1, 1).T)
            
    target = int(targets_test[k])
    
    if predicted_clf0 == 0:
        if predicted_clf1 == 0:
            ACORDOS_N += 1
            
            if target == 0:
                ACORDOS_CORRETOS_N += 1
        else:
            DESACORDOS += 1
            
            if target == 1:
                DESACORDOS_CORRETOS_clf1 += 1
            else:
                DESACORDOS_CORRETOS_clf0 += 1
    else:
        if predicted_clf1 == 0:
            
            DESACORDOS += 1
            
            if target == 1:
                DESACORDOS_CORRETOS_clf0 += 1
            else:
                DESACORDOS_CORRETOS_clf1 += 1
                
        else:
            ACORDOS_P += 1
            
            if target == 1:
                ACORDOS_CORRETOS_P += 1
    
print('Acordos')
print('  Total Positivo: ', ACORDOS_P)
print('  Corretos Positivo: ', ACORDOS_CORRETOS_P)
print('  Total Negativo: ', ACORDOS_N)
print('  Corretos Negativo: ', ACORDOS_CORRETOS_N)
print('')
print('Desacordos')
print('  Total: ', DESACORDOS)
print('Desacordos Corretos')
print('  Clf1: ', DESACORDOS_CORRETOS_clf0)
print('  Clf2: ', DESACORDOS_CORRETOS_clf1)

'''
# create plot
fig, ax = plt.subplots()
n_groups = 3
index = np.arange(n_groups)
bar_width = 0.35
opacity = 1.0

rects1 = plt.bar(index[0], ACORDOS_P, bar_width, alpha=opacity, color='k', label='Acordos Positivo')
rects2 = plt.bar(index[1], ACORDOS_N, bar_width, alpha=opacity, color='r', label='Acordos Negativo')
rects3 = plt.bar(index[2], DESACORDOS, bar_width, alpha=opacity, color='b', label='Desacordos')

plt.xlabel('Batimento Cardíaco')
plt.ylabel('Quantidade')
plt.xticks(index, ('PVC', 'Normal','PVC/Normal'))
plt.legend()

plt.tight_layout()
plt.show()

plt.savefig('experimento1_compara_exemplos.png', format='png', dpi=300)

del clf0, clf1
del features0_train, features1_train
del features0_test, features1_test
del targets_test, targets_train
'''
