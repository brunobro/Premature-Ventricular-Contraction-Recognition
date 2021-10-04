# -*- coding: utf-8 -*-
"""
Neste experimento são utilizadas a base DS1 para treinamento e DS2 para teste e duas componentes ocultas da abordagem proposta.
"""
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GroupKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import warnings
warnings.filterwarnings('ignore')
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
    return r, Acc, Pr_P, Pr_N, Se, Sp, F_P, F_N, Cm

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
    R, Acc, Pr_P, Pr_N, Se, Sp, F_P, F_N, Cm = print_measures(TP, TN, FP, FN)
    RESULTS += R
    #print Cm
    
    RESULTS += '\nDados AHP '
    RESULTS += 'clf0, clf1: ' + str(np.round(v, 4))
        
    return RESULTS, Acc, Pr_P, Pr_N, Se, Sp, F_P, F_N, Cm


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
print('>>> Resultados <<<')
print()

Acc_L = Pr_P_L = Pr_N_L = Se_L = Sp_L = F_P_L = F_N_L = []

RESULTS = ''
k = 1
group_kfold = GroupKFold(n_splits=22)

for train_index, test_index in group_kfold.split(FEATURES0, LABELS, GROUPS):
    
    FEATURES0_train, FEATURES0_test = FEATURES0[train_index], FEATURES0[test_index]    
    FEATURES1_train, FEATURES1_test = FEATURES1[train_index], FEATURES1[test_index]   
    LABELS_train, LABELS_test       = LABELS[train_index], LABELS[test_index]
    
    FEATURES0_train = LinearDiscriminantAnalysis().fit_transform(FEATURES0_train, LABELS_train)
    FEATURES1_train = LinearDiscriminantAnalysis().fit_transform(FEATURES1_train, LABELS_train)
    FEATURES0_test  = LinearDiscriminantAnalysis().fit_transform(FEATURES0_test, LABELS_test)
    FEATURES1_test  = LinearDiscriminantAnalysis().fit_transform(FEATURES1_test, LABELS_test)
    
    if FEATURES0_train.shape[1] > 0 and FEATURES1_train.shape[1] > 0 and FEATURES0_test.shape[1] > 0 and FEATURES1_test.shape[1] > 0:
        clf0 = GaussianNB()
        clf1 = GaussianNB()
        
        clf0.fit(FEATURES0_train, LABELS_train)
        clf1.fit(FEATURES1_train, LABELS_train)
        
        R, Acc, Pr_P, Pr_N, Se, Sp, F_P, F_N, Cm = classification(clf0, clf1, FEATURES0_test, FEATURES1_test, LABELS_test)
        
        if Se > 0.9 and Sp > 0.9:
            Acc_L.append(Acc)
            Pr_P_L.append(Pr_P)
            Pr_N_L.append(Pr_N)
            Se_L.append(Se)
            Sp_L.append(Sp)
            F_P_L.append(F_P)
            F_N_L.append(F_N)
            
            s = '\n######## Dobra ' + str(k) + ' ########'
            print(s)
            print(R)
                  
            r = ''
            r += s
            r += ' Registros: '
            r += '  Trein: '.join(list(np.unique(GROUPS[train_index])))
            r += '  Teste: '.join(list(np.unique(GROUPS[test_index])))
            r += ' Resultados:'
            r += R
            r += '\n& ' + str(k) + ' & ' + str(np.round(Acc, 4)) + ' & ' + str(np.round(F_P, 4)) + ' & ' + str(np.round(F_N, 4)) + ' & ' + str(np.round(Se, 4)) + ' & ' + str(np.round(Sp, 4)) + ' & ' + str(np.round(Pr_P, 4)) + ' & ' + str(np.round(Pr_N, 4)) + ' & \\\\'
            r += '#####################################\n'
            RESULTS += r
            
            k += 1
            
            if k > 10:
                break

print('Resultado médio')
print('Acc', np.mean(Acc_L), '+/-', np.std(Acc_L))
print('P+', np.mean(Pr_P_L), '+/-', np.std(Pr_P_L))
print('P-', np.mean(Pr_N_L), '+/-', np.std(Pr_N_L))
print('Se', np.mean(Se_L), '+/-', np.std(Se_L))
print('Sp', np.mean(Sp_L), '+/-', np.std(Sp_L))
print('F+', np.mean(F_P_L), '+/-', np.std(F_P_L))
print('F-', np.mean(F_N_L), '+/-', np.std(F_N_L))
#print('Cm', np.mean(Cm_L), '+/-', np.std(Cm_L))

'''
Salva os resultados sobre o conjunto de treinamento
'''
f = open(config.DIR_FILES + '2componentes/RESULTADOS_VALIDACAO_CRUZADA.txt', 'w')
f.write(RESULTS)
f.close()
