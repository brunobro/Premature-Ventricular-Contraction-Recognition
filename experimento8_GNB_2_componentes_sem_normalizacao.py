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
features0_train = tools.rectify(np.loadtxt(config.DIR_FILES + '2componentes/sem_normalizacao/' + config.FILE_FEATURES_TRAINING + '_' + config.DS_TRAINING + '_' + len_cycle_tra + '_0.txt'))
features1_train = tools.rectify(np.loadtxt(config.DIR_FILES + '2componentes/sem_normalizacao/' + config.FILE_FEATURES_TRAINING + '_' + config.DS_TRAINING + '_' + len_cycle_tra + '_1.txt'))
targets_train   = np.loadtxt(config.DIR_FILES + config.FILE_TARGET_TRAINING + '_' + config.DS_TRAINING + '_' + len_cycle_tra + '.txt')

features0_train = LinearDiscriminantAnalysis().fit_transform(features0_train, targets_train)
features1_train = LinearDiscriminantAnalysis().fit_transform(features1_train, targets_train)

'''
Carrega os dados de teste
'''
features0_test = tools.rectify(np.loadtxt(config.DIR_FILES + '2componentes/sem_normalizacao/' + config.FILE_FEATURES_TEST + '_' + config.DS_TEST + '_' + len_cycle_tes + '_0.txt'))
features1_test = tools.rectify(np.loadtxt(config.DIR_FILES + '2componentes/sem_normalizacao/' + config.FILE_FEATURES_TEST + '_' + config.DS_TEST + '_' + len_cycle_tes + '_1.txt'))
targets_test   = np.loadtxt(config.DIR_FILES + config.FILE_TARGET_TEST + '_' + config.DS_TEST + '_' + len_cycle_tes + '.txt')

features0_test = LinearDiscriminantAnalysis().fit_transform(features0_test, targets_test)
features1_test = LinearDiscriminantAnalysis().fit_transform(features1_test, targets_test)

'''
Treinamento dos Classificadores
'''
clf0 = GaussianNB()
clf1 = GaussianNB()
    
clf0.fit(features0_train, targets_train)
clf1.fit(features1_train, targets_train)
    
predicted_values0 = clf0.predict(features0_test)
predicted_values1 = clf1.predict(features1_test)

predicted_values0_proba = clf0.predict_proba(features0_test)
predicted_values1_proba = clf1.predict_proba(features1_test)

'''
Computa as medidas de performance para cada classificador
'''
M0 = measures_performance.every(predicted_values0, targets_test, config.POSITIVE_CLASS, config.NEGATIVE_CLASS)
M1 = measures_performance.every(predicted_values1, targets_test, config.POSITIVE_CLASS, config.NEGATIVE_CLASS)

'''
Experimento 1: Performance individual de cada classificador
'''

M = [M0, M1]

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
 
print('######################## EXP2 #####################')
print('Voto majoritário suave')
    
TP = 0.0
FP = 0.0
FN = 0.0
TN = 0.0
for k in range(0, targets_test.shape[0]):

    # Probabilidade de predição ponderada
    predicted_proba_clf0 = clf0.predict_proba(features0_test[k].reshape(-1, 1).T)
    predicted_proba_clf1 = clf1.predict_proba(features1_test[k].reshape(-1, 1).T)
    
    #Calcula a media ponderada das probabilidades M_proba[0] classe negativa, M_proba[1] classe positiva
    M_proba = predicted_proba_clf0[0] + predicted_proba_clf1[0]
    
    M = 0
    if  M_proba[1] > M_proba[0]:
        M = 1 #classe postiva foi a maioria
                        
    target = int(targets_test[k])
    
    #Computa as medidas
    if M == 0 and target == 0:
        TN += 1
    if M == 1 and target == 1:
        TP += 1   
    if M == 0 and target == 1:
        FN += 1
    if M == 1 and target == 0:
        FP += 1

#Retorna as medidas de performance
print_measures(TP, TN, FP, FN)

print('######################## EXP3 #####################')
print('Voto majoritário rígido - padrão negativo')
    
TP = 0.0
FP = 0.0
FN = 0.0
TN = 0.0
for k in range(0, targets_test.shape[0]):

    #Predicao
    predicted_clf0 = clf0.predict(features0_test[k].reshape(-1, 1).T)
    predicted_clf1 = clf1.predict(features1_test[k].reshape(-1, 1).T)
    
    ps = np.array([predicted_clf0, predicted_clf1])
    total_pos = len(np.where(ps == 1)[0])
    total_neg = len(np.where(ps == 0)[0])
    M = 0
    if  total_pos > total_neg:
        M = 1 #classe postiva foi a maioria
                        
    target = int(targets_test[k])
    
    #Computa as medidas
    if M == 0 and target == 0:
        TN += 1
    if M == 1 and target == 1:
        TP += 1   
    if M == 0 and target == 1:
        FN += 1
    if M == 1 and target == 0:
        FP += 1

#Retorna as medidas de performance
print_measures(TP, TN, FP, FN)

print('######################## EXP4 #####################')
print('Voto majoritário rígido - padrão positivo')
    
TP = 0.0
FP = 0.0
FN = 0.0
TN = 0.0
for k in range(0, targets_test.shape[0]):

    #Predicao
    predicted_clf0 = clf0.predict(features0_test[k].reshape(-1, 1).T)
    predicted_clf1 = clf1.predict(features1_test[k].reshape(-1, 1).T)
    
    ps = np.array([predicted_clf0, predicted_clf1])
    total_pos = len(np.where(ps == 1)[0])
    total_neg = len(np.where(ps == 0)[0])
    M = 1
    if  total_pos < total_neg:
        M = 0 #classe postiva foi a maioria
        
    target = int(targets_test[k])
    
    #Computa as medidas
    if M == 0 and target == 0:
        TN += 1
    if M == 1 and target == 1:
        TP += 1   
    if M == 0 and target == 1:
        FN += 1
    if M == 1 and target == 0:
        FP += 1

#Retorna as medidas de performance
print_measures(TP, TN, FP, FN)

'''
Emprega o voto majoritario ponderados pelas prioridades globais obtidas pelo AHP rígido
'''
print('######################## EXP5 #####################')
print('Voto AHP')

#vetor de pesos para as medidas (criterios), utilizado na AHP
w  = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
w  = w / np.linalg.norm(w, 1)

#Resultados
Acc = [M0[0], M1[0]]
F_P = [M0[1], M1[1]]
F_N = [M0[2], M1[2]]
Se  = [M0[3], M1[3]]
Sp  = [M0[4], M1[4]]
Pr_P = [M0[5], M1[5]]
Pr_N = [M0[6], M1[6]]

measures_arr = np.array([Acc, F_P, F_N, Se, Sp, Pr_P, Pr_N]).T
clf0_results = measures_arr[0,]
clf1_results = measures_arr[1,]

measures_clfs = np.array([clf0_results, clf1_results])

#Matrizes de comparações paritárias
m_Acc, m_F1_P, m_F1_N, m_Se, m_Sp, m_Pr_P, m_Pr_N = AHP.pairwiseMatrix(measures_clfs, config.KAPPA_AHP) 

#Obtém os vetores de prioridades locais
w1, cr = AHP.localVector(m_Acc)
w2, cr = AHP.localVector(m_F1_P)
w3, cr = AHP.localVector(m_F1_N)
w4, cr = AHP.localVector(m_Se)
w5, cr = AHP.localVector(m_Sp)
w6, cr = AHP.localVector(m_Pr_P)
w7, cr = AHP.localVector(m_Pr_N)

#obtém o vetor de prioridade global
ws = np.array([w1, w2, w3, w4, w5, w6, w7])
v  = AHP.globalVector(w, ws)

TP = 0.0
FP = 0.0
FN = 0.0
TN = 0.0
for k in range(0, targets_test.shape[0]):

    # Probabilidade de predição ponderada
    predicted_proba_clf0 = v[0] * clf0.predict_proba(features0_test[k].reshape(-1, 1).T)
    predicted_proba_clf1 = v[1] * clf1.predict_proba(features1_test[k].reshape(-1, 1).T)
    
    #Calcula a media ponderada das probabilidades [0] classe negativa, [1] classe positiva
    M_proba = (predicted_proba_clf0[0] + predicted_proba_clf1[0])/(v[0] + v[1])
        
    M = 0
    if M_proba[1] > 0.5: #cut-off limit
        M = 1
                        
    target = int(targets_test[k])
    
    #Computa as medidas
    if M == 0 and target == 0:
        TN += 1
    if M == 1 and target == 1:
        TP += 1   
    if M == 0 and target == 1:
        FN += 1
    if M == 1 and target == 0:
        FP += 1

#Retorna as medidas de performance
print_measures(TP, TN, FP, FN)
#print Cm

print('\nDados AHP')
print('clf0, clf1: ', np.round(v, 4))

#del clf0, clf1
#del features0_train, features1_train
#del features0_test, features1_test
#del targets_test, targets_train