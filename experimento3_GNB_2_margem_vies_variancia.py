# -*- coding: utf-8 -*-
"""
Neste experimento são utilizadas a base DS1 para treinamento e DS2 para teste e duas componentes ocultas da abordagem proposta.
"""
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GroupKFold
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import config
import tools
import measures_performance
import AHP
import os.path
from config import RECORDS_DS1, RECORDS_DS2

def perda_01(y, y_hat):
    if y == y_hat:
        return 0
    else:
        return 1
    
def bias():
    return 0
   

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
    Emprega o voto majoritario ponderados pelas prioridades globais obtidas pelo AHP rígido
    '''    
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
    m_Acc, m_F1_P, m_F1_N, m_Se, m_Sp, m_Pr_P, m_Pr_N = AHP.pairwiseMatrix(measures_clfs) 
    
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
    
    MARGEM_0 = []
    MARGEM_1 = []
    
    for k in range(0, targets_test.shape[0]):
    
        # Probabilidade de predição ponderada
        predicted_proba_clf0 = int(clf0.predict_proba(features0_test[k].reshape(-1, 1).T))
        predicted_proba_clf1 = int(clf1.predict_proba(features1_test[k].reshape(-1, 1).T))
                                    
        target = int(targets_test[k])
        
        #Soma os pesos das predições corretas e incorretas
        w_label_certo  = 0
        w_label_errado = 0
        
        if target == 0:
            if predicted_proba_clf0 == 0:
                if predicted_proba_clf1 == 0:
                    w_label_certo = v[0] + v[1]
                else:
                    w_label_certo  = v[0]
                    w_label_errado = v[1]
            else:
                if predicted_proba_clf1 == 0:
                    w_label_certo  = v[1]
                    w_label_errado = v[0]
                    
            MARGEM_0.append(w_label_certo - w_label_errado)
            
        else:
            if predicted_proba_clf0 == 1:
                if predicted_proba_clf1 == 1:
                    w_label_certo = v[0] + v[1]
                else:
                    w_label_certo  = v[0]
                    w_label_errado = v[1]
            else:
                if predicted_proba_clf1 == 1:
                    w_label_certo  = v[1]
                    w_label_errado = v[0]
                    
            MARGEM_1.append(w_label_certo - w_label_errado)
            
        
    return MARGEM_0, MARGEM_1

'''
Faz a leitura de cada registro e armazenas os dados para posterior aplicação
da validação Cruzada
Serão selecionados 15 registros para treinamento e 7 para teste independente da
quantidade da batimentos em cada um
'''
FEATURES0 = []
FEATURES1 = []
LABELS    = []
GROUPS    = []

for RECORD in eval(config.DS_TRAINING) + eval(config.DS_TEST):
    
    print('#Registro ', RECORD)
          
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
        F0 = tools.rectify(change_shape(data0))
        
        
        data1 = np.loadtxt(config.DIR_FILES + '2componentes/' + config.FILE_FEATURES_TRAINING + '_' + config.DS_TRAINING + '_' + len_cycle_tra + '_' + str(RECORD) + '_1.txt')
        F1 = tools.rectify(change_shape(data1))
        
        
        L = np.loadtxt(config.DIR_FILES_RECORDS + config.FILE_TARGET_TRAINING + '_' + config.DS_TRAINING + '_' + len_cycle_tra + '_' + str(RECORD) + '.txt')
                
        for i in range(0, len(L)):
            FEATURES0.append(F0[i])
            FEATURES1.append(F1[i])
            LABELS.append(L[i])
            GROUPS.append(str(RECORD))
         
FEATURES0 = np.array(FEATURES0)
FEATURES1 = np.array(FEATURES1)
LABELS    = np.array(LABELS)
GROUPS    = np.array(GROUPS)

'''
Validação Cruzada
Ajuste e validação dos classificadores
'''
group_kfold = GroupKFold(n_splits=22)

MARGEM_0, MARGEM_1, BIAS, VARIANCE = []

for train_index, test_index in group_kfold.split(FEATURES0, LABELS, GROUPS):
    
    FEATURES0_train, FEATURES0_test = FEATURES0[train_index], FEATURES0[test_index]    
    FEATURES1_train, FEATURES1_test = FEATURES1[train_index], FEATURES1[test_index]   
    LABELS_train, LABELS_test       = LABELS[train_index], LABELS[test_index]
    
    clf0 = GaussianNB()
    clf1 = GaussianNB()
    
    clf0.fit(FEATURES0_train, LABELS_train)
    clf1.fit(FEATURES1_train, LABELS_train)
    
    m_0, m_1 = classification(clf0, clf1, FEATURES0_test, FEATURES1_test, LABELS_test)
    
    bias_0     = 0.5 * (1 - np.sign(MARGEM_0))
    variance_0 = 0.5 * (1 + np.abs(MARGEM_0))
    

'''
Salva os resultados sobre o conjunto de treinamento
'''
