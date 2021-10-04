# -*- coding: utf-8 -*-
"""
Utiliza apenas duas compoenetes do AMUSE
"""
from sklearn.naive_bayes import GaussianNB
import numpy as np
import config
import measures_performance
import AHP

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
predicted_values0 = clf1.predict(features0_test)
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
    print(' AUC: ', round(m[5], 4))
    print(' Se:  ', round(m[9], 4))
    print(' Sp:  ', round(m[10], 4))
    i += 1
 
print('######################## EXP2 #####################')
print('Performance utilizando voto majoritário')
    
'''
Emprega o voto majoritario ponderados pelas prioridades globais obtidas pelo AHP
'''
TP = 0.0
FP = 0.0
FN = 0.0
TN = 0.0
predicted_proba  = []
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
                
    predicted_proba.append(M)
        
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
Acc, Pr, Se, Sp, F1, Cm = measures_performance.calc(TP, TN, FP, FN)
auc = measures_performance.AUC(np.rint(targets_test), np.array(predicted_proba))

print(' Acc: ', round(Acc, 4))
print(' AUC: ', round(auc, 4))
print(' Se:  ', round(Se, 4))
print(' Sp:  ', round(Sp, 4))

'''
Experimento 3: Comite de classificadores utilizando o método AHP
'''

print('######################## EXP3 #####################')

#vetor de pesos para as medidas (criterios), utilizado na AHP
w  = [1.0, 1.0, 1.0, 1.0]
w  = w / np.linalg.norm(w, 1)

#Resultados para cada classificador
Acc = [M0[0], M1[0]]
AUC = [M0[5], M1[5]]
Se  = [M0[9], M1[9]]
Pr  = [M0[10], M1[10]]

measures_arr = np.array([Acc, AUC, Se, Pr]).T
clf0_results = measures_arr[0,]
clf1_results = measures_arr[1,]

measures_clfs = np.array([clf0_results, clf1_results])

#Matrizes de comparações paritárias
m_Acc, m_AUC, m_Se, m_Pr = AHP.pairwiseMatrix(measures_clfs) 

#Obtém os vetores de prioridades locais
w1, cr = AHP.localVector(m_Acc)
w2, cr = AHP.localVector(m_AUC)
w3, cr = AHP.localVector(m_Se)
w4, cr = AHP.localVector(m_Pr)

#obtém o vetor de prioridade global
ws = np.array([w1, w2, w3, w4])
v  = AHP.globalVector(w, ws)

print('AHP - clf0, clf1: ', np.round(v, 4))

'''
Emprega o voto majoritario ponderados pelas prioridades globais obtidas pelo AHP
'''
TP = 0.0
FP = 0.0
FN = 0.0
TN = 0.0
predicted_proba  = []
total_proba      = []
for k in range(0, targets_test.shape[0]):

    # Probabilidade de predição ponderada
    predicted_proba_clf0 = v[0] * clf0.predict_proba(features0_test[k].reshape(-1, 1).T)
    predicted_proba_clf1 = v[1] * clf1.predict_proba(features1_test[k].reshape(-1, 1).T)
    
    #Calcula a media ponderada das probabilidades [0] classe negativa, [1] classe positiva
    M_proba = (predicted_proba_clf0[0] + predicted_proba_clf1[0])/(v[0] + v[1])
    
    total_proba.append(M_proba[1])
    
    M = 0
    if M_proba[1] > 0.5: #cut-off limit
        M = 1
        
    predicted_proba.append(M)
                
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
Acc, Pr, Se, Sp, F1, Cm = measures_performance.calc(TP, TN, FP, FN)
auc = measures_performance.AUC(np.rint(targets_test), np.array(predicted_proba))

print('Performance usando voto AHP rígido')
print(' Acc: ', round(Acc, 4))
print(' AUC: ', round(auc, 4))
print(' Se:  ', round(Se, 4))
print(' Sp:  ', round(Sp, 4))
#print Cm

del clf0, clf1
del features0_train, features1_train
del features0_test, features1_test
del targets_test, targets_train

