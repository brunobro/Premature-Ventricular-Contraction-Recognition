# -*- coding: utf-8 -*-
"""
Implementa o ensemble utilizando subconjuntos do espaço de características
Utiliza apenas Regressão Logística
"""
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import sklearn.metrics as metrics
import numpy as np
import config
import measures_performance
import AHP

'''
Carrega os dados de treinamento
'''
features1_train = np.loadtxt(config.DIR_FILES + config.FILE_FEATURES_TRAINING + '_' + config.DS_TRAINING + '_1.txt')
features2_train = np.loadtxt(config.DIR_FILES + config.FILE_FEATURES_TRAINING + '_' + config.DS_TRAINING + '_2.txt')
features3_train = np.loadtxt(config.DIR_FILES + config.FILE_FEATURES_TRAINING + '_' + config.DS_TRAINING + '_3.txt')
targets_train   = np.loadtxt(config.DIR_FILES + config.FILE_TARGET_TRAINING + '_' + config.DS_TRAINING + '.txt')

'''
Carrega os dados de teste
'''
features1_test = np.loadtxt(config.DIR_FILES + config.FILE_FEATURES_TEST + '_' + config.DS_TEST + '_1.txt')
features2_test = np.loadtxt(config.DIR_FILES + config.FILE_FEATURES_TEST + '_' + config.DS_TEST + '_2.txt')
features3_test = np.loadtxt(config.DIR_FILES + config.FILE_FEATURES_TEST + '_' + config.DS_TEST + '_3.txt')
targets_test   = np.loadtxt(config.DIR_FILES + config.FILE_TARGET_TEST + '_' + config.DS_TEST + '.txt')

'''
Treinamento dos Classificadores
'''
#regularization = 'l1'
#max_iter       = 1e3
#C              = 0.1
#clf1 = LogisticRegression(penalty=regularization, solver='saga', C=C, max_iter=max_iter)
#clf2 = LogisticRegression(penalty=regularization, solver='saga', C=C, max_iter=max_iter)
#clf3 = LogisticRegression(penalty=regularization, solver='saga', C=C, max_iter=max_iter)

clf1 = GaussianNB()
clf2 = GaussianNB()
clf3 = GaussianNB()

#Coeficientes de regressão
#clf1.coef_ #beta1, ..., betaN
#clf1.intercept_ #beta0

'''
Indução dos classificadores
'''
clf1.fit(features1_train, targets_train)
clf2.fit(features2_train, targets_train)
clf3.fit(features3_train, targets_train)

'''
Previsao dos valores reais das classes, os quais são 0 ou 1
'''
predicted_values1 = clf1.predict(features1_test)
predicted_values2 = clf2.predict(features2_test)
predicted_values3 = clf3.predict(features3_test)

predicted_values1_proba = clf1.predict_proba(features1_test)
predicted_values2_proba = clf2.predict_proba(features2_test)
predicted_values3_proba = clf3.predict_proba(features3_test)

'''
Computa as medidas de performance para cada classificador
'''
M1 = measures_performance.every(predicted_values1, targets_test, config.POSITIVE_CLASS, config.NEGATIVE_CLASS)
M2 = measures_performance.every(predicted_values2, targets_test, config.POSITIVE_CLASS, config.NEGATIVE_CLASS)
M3 = measures_performance.every(predicted_values3, targets_test, config.POSITIVE_CLASS, config.NEGATIVE_CLASS)

'''
Experimento 1: Performance individual de cada classificador
'''

M = [M1, M2, M3]

print('######################## EXP1 #####################')
print('RESULTADOS: Individual para cada classificador')

i = 1
for m in M:
    print('Clf', i)
    print(' Acc: ', round(m[0], 4))
    print(' AUC: ', round(m[5], 4))
    print(' Se:  ', round(m[9], 4))
    print(' Sp:  ', round(m[10], 4))
    i += 1
 
print('######################## EXP2 #####################')
print('RESULTADOS: voto majoritário ponderado pelo AHP')
    
'''
Emprega o voto majoritario ponderados pelas prioridades globais obtidas pelo AHP
'''

new_dataset_train_samples_VOT = []
new_dataset_train_targets_VOT = []
new_dataset_test_samples_VOT  = []
new_dataset_test_targets_VOT  = []

for l in range(0, 2):
    
    #escolhe a base de dados para testar
    if l == 0:
        features1 = features1_train
        features2 = features2_train
        features3 = features3_train
    else:
        features1 = features1_test
        features2 = features2_test
        features3 = features3_test
    
    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0
    predicted_proba  = []
    
    for k in range(0, targets_test.shape[0]):
    
        #Predicao
        predicted_clf1 = clf1.predict(features1[k].reshape(-1, 1).T)
        predicted_clf2 = clf2.predict(features2[k].reshape(-1, 1).T)
        predicted_clf3 = clf3.predict(features3[k].reshape(-1, 1).T)
                
        s = [predicted_clf1[0], predicted_clf2[0], predicted_clf3[0]]
        
        if l == 0:
            new_dataset_train_samples_VOT.append(s)
        else:
            new_dataset_test_samples_VOT.append(s)
        
        ps = np.array([predicted_clf1, predicted_clf2, predicted_clf3])
        total_pos = len(np.where(ps == 1)[0])
        total_neg = len(np.where(ps == 0)[0])
        M = 0
        if  total_pos > total_neg:
            M = 1 #classe postiva foi a maioria
                    
        predicted_proba.append(M)
            
        target = int(targets_test[k])
        
        if l == 0:
            new_dataset_train_targets_VOT.append(target)
        else:
            new_dataset_test_targets_VOT.append(target)
        
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

    label_dados = 'Teste'    
    if l == 0:
        label_dados = 'Treinamento'
        
    print('Performance sobre os dados de ', label_dados)
    print(' Acc: ', round(Acc, 4))
    print(' AUC: ', round(auc, 4))
    print(' Se:  ', round(Se, 4))
    print(' Sp:  ', round(Sp, 4))

'''
Experimento 3: Comite de classificadores utilizando o método AHP
'''

print('######################## EXP3 #####################')
print('RESULTADOS: voto majoritário ponderado pelo AHP')

#vetor de pesos para as medidas (criterios), utilizado na AHP
w  = [1.0, 1.0, 1.0, 1.0]
w  = w / np.linalg.norm(w, 1)

#Resultados para cada classificador
Acc = [M1[0], M2[0], M3[0]]
AUC = [M1[5], M2[5], M3[5]]
Se  = [M1[9], M2[9], M3[9]]
Pr  = [M1[10], M2[10], M3[10]]

measures_arr = np.array([Acc, AUC, Se, Pr]).T
clf1_results = measures_arr[0,]
clf2_results = measures_arr[1,]
clf3_results = measures_arr[2,]

measures_clfs = np.array([clf1_results, clf2_results, clf3_results])

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

print('AHP - clf1, clf2, clf3, clf4: ', np.round(v, 4))

'''
Emprega o voto majoritario ponderados pelas prioridades globais obtidas pelo AHP
'''

CUT_OFF = 0.5

new_dataset_train_samples_AHP = []
new_dataset_train_targets_AHP = []
new_dataset_test_samples_AHP  = []
new_dataset_test_targets_AHP  = []

for l in range(0, 2):
    
    #escolhe a base de dados para testar
    if l == 0:
        features1 = features1_train
        features2 = features2_train
        features3 = features3_train
    else:
        features1 = features1_test
        features2 = features2_test
        features3 = features3_test
            
    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0
    predicted_proba  = []
    total_proba      = []
    
    for k in range(0, targets_test.shape[0]):
    
        # Probabilidade de predição ponderada
        predicted_proba_clf1 = v[0] * clf1.predict_proba(features1[k].reshape(-1, 1).T)
        predicted_proba_clf2 = v[1] * clf2.predict_proba(features2[k].reshape(-1, 1).T)
        predicted_proba_clf3 = v[2] * clf3.predict_proba(features3[k].reshape(-1, 1).T)
                
        #Calcula a media ponderada das probabilidades [0] classe negativa, [1] classe positiva
        M_proba = (predicted_proba_clf1[0] + predicted_proba_clf2[0] + predicted_proba_clf3[0])/(v[0] + v[1] + v[2])
        
        total_proba.append(M_proba[1])
        
        M = 0
        if M_proba[1] > CUT_OFF: #cut-off limit
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
    
    label_dados = 'Teste'    
    if l == 0:
        label_dados = 'Treinamento'
        
    print('Performance sobre os dados de ', label_dados)
    print(' Acc: ', round(Acc, 4))
    print(' AUC: ', round(auc, 4))
    print(' Se:  ', round(Se, 4))
    print(' Sp:  ', round(Sp, 4))

#del clf1, clf2, clf3, clf4
del features1_train, features2_train, features3_train
del features1_test, features2_test, features3_test
del targets_test, targets_train