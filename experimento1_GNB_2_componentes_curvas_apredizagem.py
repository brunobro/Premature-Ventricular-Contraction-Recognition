# -*- coding: utf-8 -*-
"""
Neste experimento são utilizadas a base DS1 para treinamento e DS2 para teste e duas componentes ocultas da abordagem proposta.
Plota as curvas de aprendizagem das máquinas
"""
from sklearn.naive_bayes import GaussianNB
import numpy as np
import config
import tools
from sklearn.metrics import accuracy_score

################# PLOT #################
import matplotlib 
import matplotlib.pyplot as plt
matplotlib.rc('xtick', labelsize=9) 
matplotlib.rc('ytick', labelsize=9)
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

len_cycle_tra = str(config.LEN_CARDIAC_CYCLE(config.DB_TRAINING))
len_cycle_tes = str(config.LEN_CARDIAC_CYCLE(config.DB_TEST))

'''
Carrega os dados de treinamento
'''
features0_train = tools.rectify(np.loadtxt(config.DIR_FILES + '2componentes/' + config.FILE_FEATURES_TRAINING + '_' + config.DS_TRAINING + '_' + len_cycle_tra + '_0.txt'))
features1_train = tools.rectify(np.loadtxt(config.DIR_FILES + '2componentes/' + config.FILE_FEATURES_TRAINING + '_' + config.DS_TRAINING + '_' + len_cycle_tra + '_1.txt'))
targets_train   = np.loadtxt(config.DIR_FILES + config.FILE_TARGET_TRAINING + '_' + config.DS_TRAINING + '_' + len_cycle_tra + '.txt')

'''
Carrega os dados de teste
'''
features0_test = tools.rectify(np.loadtxt(config.DIR_FILES + '2componentes/' + config.FILE_FEATURES_TEST + '_' + config.DS_TEST + '_' + len_cycle_tes + '_0.txt'))
features1_test = tools.rectify(np.loadtxt(config.DIR_FILES + '2componentes/' + config.FILE_FEATURES_TEST + '_' + config.DS_TEST + '_' + len_cycle_tes + '_1.txt'))
targets_test   = np.loadtxt(config.DIR_FILES + config.FILE_TARGET_TEST + '_' + config.DS_TEST + '_' + len_cycle_tes + '.txt')

'''
Altera as instâncias para que haja maior distribuição dos exemplos positivos
'''
N = features0_train.shape[0]
I = np.random.randint(0, N, size=N)
features0_train_ = []
features1_train_ = []
targets_train_   = []
for i in I:
    features0_train_.append(features0_train[i,:])
    features1_train_.append(features1_train[i,:])
    targets_train_.append(targets_train[i])
    
del features0_train, features1_train, targets_train

features0_train_ = np.array(features0_train_)
features1_train_ = np.array(features1_train_)
targets_train_   = np.array(targets_train_)

'''
Treinamento dos Classificadores
'''
clf0 = GaussianNB()
clf1 = GaussianNB()

'''
Ajuste dos classificadores
'''
ACC_0 = []
ACC_1 = []

STEP  = 1000 #step
instances = np.arange(STEP, features0_train_.shape[0], STEP)

for LEN in instances:
    
    print('Instâncias: 0 a ', LEN)
        
    clf0.fit(features0_train_[0:LEN,:], targets_train_[0:LEN])
    clf1.fit(features1_train_[0:LEN,:], targets_train_[0:LEN])
        
    acc0 = np.round(accuracy_score(targets_test, clf0.predict(features0_test)) * 100, 4)
    acc1 = np.round(accuracy_score(targets_test, clf1.predict(features1_test)) * 100, 4)
    
    print('Acc 0: ', acc0)
    print('Acc 1: ', acc1, '\n')
    
    ACC_0.append(acc0)
    ACC_1.append(acc1)
    

plt.figure(1)
plt.plot(instances, ACC_0, label=r'$Clf_0$')
plt.plot(instances, ACC_1, label=r'$Clf_1$')
plt.xlabel('Instâncias')
plt.ylabel('Acurácia (%)')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig('experimento1_cruvas_aprendizagem.png', format='png', dpi=300)
#del clf0, clf1
#del features0_train, features1_train
#del features0_test, features1_test
#del targets_test, targets_train