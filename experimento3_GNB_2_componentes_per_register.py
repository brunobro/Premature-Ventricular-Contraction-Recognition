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
import os.path
from config import RECORDS_DS1, RECORDS_DS2

def print_measures(TP, TN, FP, FN):
    r = ''
    Acc, Pr_P, Pr_N, Se, Sp, F_P, F_N, Cm = measures_performance.calc(TP, TN, FP, FN)
    r += '\n Acc: ' + str(round(Acc, 4))
    r += '\n F+:  ' + str(round(F_P, 4))
    r += '\n F-:  ' + str(round(F_N, 4))
    r += '\n Se:  ' + str(round(Se, 4))
    r += '\n Sp:  ' + str(round(Sp, 4))
    r += '\n P+:  ' + str(round(Pr_P, 4))
    r += '\n P-:  ' + str(round(Pr_N, 4))
    r += '\n TP:  ' + str(TP) 
    r += '\n TN:  ' + str(TN)
    r += '\n FP:  ' + str(FP)
    r += '\n FN:  ' + str(FN)
    return r

len_cycle_tra = str(config.LEN_CARDIAC_CYCLE(config.DB_TRAINING))
len_cycle_tes = str(config.LEN_CARDIAC_CYCLE(config.DB_TEST))

def change_shape(features):
    if len(features) == features.size:
        return features.reshape(-1, 1).T
    else:
        return features

def classification(clf0, clf1, features0_test, features1_test, targets_test):

    RESULTS = ''
    
    '''
    Caso exista somente um atributo para o registro
    '''
    if targets_test.size == 1:
       targets_test = np.array([targets_test])
        
    '''
    Previsao dos valores reais das classes, os quais são 0 ou 1
    '''
    predicted_values0 = clf0.predict(features0_test)
    predicted_values1 = clf1.predict(features1_test)
        
    '''
    Computa as medidas de performance para cada classificador
    '''
    M0 = measures_performance.every(predicted_values0, targets_test, config.POSITIVE_CLASS, config.NEGATIVE_CLASS)
    M1 = measures_performance.every(predicted_values1, targets_test, config.POSITIVE_CLASS, config.NEGATIVE_CLASS)
    
    '''
    Experimento 1: Performance individual de cada classificador
    '''
    
    M = [M0, M1]
    '''
    RESULTS += '\n######################## EXP1 #####################'
    RESULTS += '\nPerformance individual para cada classificador'
    
    i = 0
    r = ''
    for m in M:
        r += '\nClf' + str(i)
        r += '\n Acc: ' + str(round(m[0], 4))
        r += '\n F+:  ' + str(round(m[1], 4))
        r += '\n F-:  ' + str(round(m[2], 4))
        r += '\n Se:  ' + str(round(m[3], 4))
        r += '\n Sp:  ' + str(round(m[4], 4))
        r += '\n P+:  ' + str(round(m[5], 4))
        r += '\n P-:  ' + str(round(m[6], 4))
        r += '\n TP:  ' + str(m[7]) 
        r += '\n TN:  ' + str(m[8])
        r += '\n FP:  ' + str(m[9])
        r += '\n FN:  ' + str(m[10])
        i += 1
    
    RESULTS += r
     
    RESULTS += '\n######################## EXP2 #####################'
    RESULTS += '\nVoto majoritário suave'
        
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
    RESULTS += print_measures(TP, TN, FP, FN)
    '''
    #RESULTS += '\n######################## EXP3 #####################'
    RESULTS += '\nVoto majoritário rígido - padrão negativo'
        
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
    RESULTS += print_measures(TP, TN, FP, FN)
    '''
    RESULTS += '\n######################## EXP4 #####################'
    RESULTS += '\nVoto majoritário rígido - padrão positivo'
        
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
    RESULTS += print_measures(TP, TN, FP, FN)
    '''
    '''
    Emprega o voto majoritario ponderados pelas prioridades globais obtidas pelo AHP rígido
    '''
    #RESULTS += '\n######################## EXP5 #####################'
    RESULTS += '\nVoto AHP'
    
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
        if M_proba[1] > 0.7: #cut-off limit
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
    RESULTS += print_measures(TP, TN, FP, FN)
    #print Cm
    
    RESULTS += '\nDados AHP'
    RESULTS += 'clf0, clf1: ' + str(np.round(v, 4))
        
    return RESULTS + '\n#####################################################\n'


'''
DS1: treinamento
DS2: teste
'''
print('************** Teste 1 - DS1 para treinamento **************')

'''
Carrega os dados de treinamento
'''
features0_trai = tools.rectify(np.loadtxt(config.DIR_FILES + '2componentes/' + config.FILE_FEATURES_TRAINING + '_' + config.DS_TRAINING + '_' + len_cycle_tra + '_0.txt'))
features1_trai = tools.rectify(np.loadtxt(config.DIR_FILES + '2componentes/' + config.FILE_FEATURES_TRAINING + '_' + config.DS_TRAINING + '_' + len_cycle_tra + '_1.txt'))
targets_trai   = np.loadtxt(config.DIR_FILES + config.FILE_TARGET_TRAINING + '_' + config.DS_TRAINING + '_' + len_cycle_tra + '.txt')

features0_trai = LinearDiscriminantAnalysis().fit_transform(features0_trai, targets_trai)
features1_trai = LinearDiscriminantAnalysis().fit_transform(features1_trai, targets_trai)

'''
Treinamento dos Classificadores
'''
clf0 = GaussianNB()
clf1 = GaussianNB()

'''
Ajuste dos classificadores
'''    
clf0.fit(features0_trai, targets_trai)
clf1.fit(features1_trai, targets_trai)

del features0_trai, features1_trai, targets_trai

RESULTS = '\n\n>>> Resultados sobre o conjunto de treinamento <<<'
for RECORD in eval(config.DS_TRAINING):
    
    print('#Registro ', RECORD)
    RESULTS += '\n#Registro ' + str(RECORD)
    
    '''
    Carrega os dados de teste
    '''    
    rec0 = config.DIR_FILES + '2componentes/' + config.FILE_FEATURES_TRAINING + '_' + config.DS_TRAINING + '_' + len_cycle_tra + '_' + str(RECORD) + '_0.txt'
    
    '''
    Verifica se o arquivo do registro existe. Pois caso não haja batimentos das
    arritmias no ECG ele não vai gerar atributos.
    Verifica somente a componente 0, pois se existe para ela existe para as outras também
    '''
    if os.path.exists(rec0):
        data0 = np.loadtxt(rec0)
        features0_test = tools.rectify(change_shape(data0))
        
        data1 = np.loadtxt(config.DIR_FILES + '2componentes/' + config.FILE_FEATURES_TRAINING + '_' + config.DS_TRAINING + '_' + len_cycle_tra + '_' + str(RECORD) + '_1.txt')
        features1_test = tools.rectify(change_shape(data1))
        
        targets_test = np.loadtxt(config.DIR_FILES_RECORDS + config.FILE_TARGET_TRAINING + '_' + config.DS_TRAINING + '_' + len_cycle_tra + '_' + str(RECORD) + '.txt')
        
        features0_test = LinearDiscriminantAnalysis().fit_transform(features0_test, targets_test)
        features1_test = LinearDiscriminantAnalysis().fit_transform(features1_test, targets_test)
    
        RESULTS += classification(clf0, clf1, features0_test, features1_test, targets_test)

del features0_test, features1_test, targets_test

'''
Salva os resultados sobre o conjunto de treinamento
'''
f = open(config.DIR_FILES + '2componentes/RESULTADOS_DS1TRAI_R_TRAI.txt', 'w')
f.write(RESULTS)
f.close()
 
RESULTS = '\n\n>>> Resultados sobre o conjunto de teste <<<'
for RECORD in eval(config.DS_TEST):
    
    print('#Registro ', RECORD)
    RESULTS += '\n#Registro ' + str(RECORD)
    
    '''
    Carrega os dados de teste
    '''
    rec0 = config.DIR_FILES + '2componentes/' + config.FILE_FEATURES_TEST + '_' + config.DS_TEST + '_' + len_cycle_tes + '_' + str(RECORD) + '_0.txt'
    
    '''
    Verifica se o arquivo do registro existe. Pois caso não haja batimentos das
    arritmias no ECG ele não vai gerar atributos.
    Verifica somente a componente 0, pois se existe para ela existe para as outras também
    '''
    if os.path.exists(rec0):
        data0 = np.loadtxt(rec0)
        features0_test = tools.rectify(change_shape(data0))
        
        data1 = np.loadtxt(config.DIR_FILES + '2componentes/' + config.FILE_FEATURES_TEST + '_' + config.DS_TEST + '_' + len_cycle_tes + '_' + str(RECORD) + '_1.txt')
        features1_test = tools.rectify(change_shape(data1))
        
        targets_test   = np.loadtxt(config.DIR_FILES_RECORDS + config.FILE_TARGET_TEST + '_' + config.DS_TEST + '_' + len_cycle_tes + '_' + str(RECORD) + '.txt')
        
        features0_test = LinearDiscriminantAnalysis().fit_transform(features0_test, targets_test)
        features1_test = LinearDiscriminantAnalysis().fit_transform(features1_test, targets_test)
        
        if features0_test.shape[1] > 0 and features1_test.shape[1] > 0:
            RESULTS += classification(clf0, clf1, features0_test, features1_test, targets_test)

del features0_test, features1_test, targets_test

'''
Salva os resultados sobre o conjunto de teste
'''
f = open(config.DIR_FILES + '2componentes/RESULTADOS_DS1TRAI_R_TEST.txt', 'w')
f.write(RESULTS)
f.close()

'''
DS2: treinamento
DS1: teste
'''
print('************** Teste 2 - DS2 para treinamento **************')

'''
Carrega os dados de treinamento
'''
features0_trai = tools.rectify(np.loadtxt(config.DIR_FILES + '2componentes/' + config.FILE_FEATURES_TEST + '_' + config.DS_TEST + '_' + len_cycle_tes + '_0.txt'))
features1_trai = tools.rectify(np.loadtxt(config.DIR_FILES + '2componentes/' + config.FILE_FEATURES_TEST + '_' + config.DS_TEST + '_' + len_cycle_tes + '_1.txt'))
targets_trai   = np.loadtxt(config.DIR_FILES + config.FILE_TARGET_TEST + '_' + config.DS_TEST + '_' + len_cycle_tes + '.txt')

features0_trai = LinearDiscriminantAnalysis().fit_transform(features0_trai, targets_trai)
features1_trai = LinearDiscriminantAnalysis().fit_transform(features1_trai, targets_trai)

'''
Treinamento dos Classificadores
'''
clf0 = GaussianNB()
clf1 = GaussianNB()

'''
Ajuste dos classificadores
'''    
clf0.fit(features0_trai, targets_trai)
clf1.fit(features1_trai, targets_trai)

del features0_trai, features1_trai, targets_trai

RESULTS = '\n\n>>> Resultados sobre o conjunto de treinamento <<<'
for RECORD in eval(config.DS_TEST):
    
    print('#Registro ', RECORD)
    RESULTS += '\n#Registro ' + str(RECORD)
    
    '''
    Carrega os dados de teste
    '''
    rec0 = config.DIR_FILES + '2componentes/' + config.FILE_FEATURES_TEST + '_' + config.DS_TEST + '_' + len_cycle_tes + '_' + str(RECORD) + '_0.txt'
    
    if os.path.exists(rec0):
        data0 = np.loadtxt(rec0)
        features0_test = tools.rectify(change_shape(data0))
        
        data1 = np.loadtxt(config.DIR_FILES + '2componentes/' + config.FILE_FEATURES_TEST + '_' + config.DS_TEST + '_' + len_cycle_tes + '_' + str(RECORD) + '_1.txt')
        features1_test = tools.rectify(change_shape(data1))
        
        targets_test   = np.loadtxt(config.DIR_FILES_RECORDS + config.FILE_TARGET_TEST + '_' + config.DS_TEST + '_' + len_cycle_tes + '_' + str(RECORD) + '.txt')
        
        features0_test = LinearDiscriminantAnalysis().fit_transform(features0_test, targets_test)
        features1_test = LinearDiscriminantAnalysis().fit_transform(features1_test, targets_test)
        
        if features0_test.shape[1] > 0 and features1_test.shape[1] > 0:
            RESULTS += classification(clf0, clf1, features0_test, features1_test, targets_test)

del features0_test, features1_test, targets_test
 
'''
Salva os resultados sobre o conjunto de treinamento
'''
f = open(config.DIR_FILES + '2componentes/RESULTADOS_DS2TRAI_R_TRAI.txt', 'w')
f.write(RESULTS)
f.close()


RESULTS = '\n\n>>> Resultados sobre o conjunto de teste <<<'
    
for RECORD in eval(config.DS_TRAINING):
    
    print('\n#Registro ', RECORD)
    RESULTS += '\n#Registro ' + str(RECORD)
    
    '''
    Carrega os dados de teste
    '''
    rec0 = config.DIR_FILES + '2componentes/' + config.FILE_FEATURES_TRAINING + '_' + config.DS_TRAINING + '_' + len_cycle_tra + '_' + str(RECORD) + '_0.txt'
    
    if os.path.exists(rec0):
        data0 = np.loadtxt(rec0)
        features0_test = tools.rectify(change_shape(data0))
        
        data1 = np.loadtxt(config.DIR_FILES + '2componentes/' + config.FILE_FEATURES_TRAINING + '_' + config.DS_TRAINING + '_' + len_cycle_tra + '_' + str(RECORD) + '_1.txt')
        features1_test = tools.rectify(change_shape(data0))
        
        targets_test   = np.loadtxt(config.DIR_FILES_RECORDS + config.FILE_TARGET_TRAINING + '_' + config.DS_TRAINING + '_' + len_cycle_tra + '_' + str(RECORD) + '.txt')
        
        features0_test = LinearDiscriminantAnalysis().fit_transform(features0_test, targets_test)
        features1_test = LinearDiscriminantAnalysis().fit_transform(features1_test, targets_test)
        
        if features0_test.shape[1] > 0 and features1_test.shape[1] > 0:
            RESULTS += classification(clf0, clf1, features0_test, features1_test, targets_test)

del features0_test, features1_test, targets_test
    
'''
Salva os resultados sobre o conjunto de teste
'''
f = open(config.DIR_FILES + '2componentes/RESULTADOS_DS2TRAI_R_TEST.txt', 'w')
f.write(RESULTS)
f.close()
