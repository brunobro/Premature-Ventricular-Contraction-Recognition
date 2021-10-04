# -*- coding: utf-8 -*-
"""
Balanceia os dados gerandos exemplos CPV artificias, com base na média de dois
outros batimentos, para cada paciente. Implementa no modo validação cruzada
"""
from sklearn.naive_bayes import GaussianNB
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
    return r, Acc, Pr_P, Pr_N, Se, Sp, F_P, F_N, Cm, TP, TN, FP, FN

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
    R, Acc, Pr_P, Pr_N, Se, Sp, F_P, F_N, Cm, TP, TN, FP, FN = print_measures(TP, TN, FP, FN)
    RESULTS += R
    #print Cm
    
    RESULTS += '\nDados AHP '
    RESULTS += 'clf0, clf1: ' + str(np.round(v, 4))
        
    return RESULTS, Acc, Pr_P, Pr_N, Se, Sp, F_P, F_N, Cm, TP, TN, FP, FN


'''
Faz a leitura de cada registro e armazenas os dados para posterior aplicação
da validação Cruzada
Serão selecionados 15 registros para treinamento e 7 para teste independente da
quantidade da batimentos em cada um
'''
#performance
Acc_L = Pr_P_L = Pr_N_L = Se_L = Sp_L = F_P_L = F_N_L = []

#resultados para arquivo txt
RESULTS = ''

#valores iniciais
v_trai = np.array(eval(config.DS_TRAINING))
v_test = np.array(eval(config.DS_TEST))

index_change = 0
fold = 0

for i in range(0, int((len(v_trai) + len(v_test))/2)):
    
    fold += 1
    
    FEATURES0_TRAI = []
    FEATURES0_TEST = []
    FEATURES1_TRAI = []
    FEATURES1_TEST = []
    LABELS_TRAI    = []
    LABELS_TEST    = []
    
    '''
    Carrega os dados de treinamento
    '''
    print('Lendo Dados Treinamento')
    for record in v_trai: 
        #print('> Registro', record)
        rec0 = config.DIR_FILES + '2componentes/records/' + str(record) + '_0.txt'
    
        '''
        Verifica se o arquivo do registro existe. Pois caso não haja batimentos das arritmias no ECG ele não vai gerar atributos.
        '''
        if os.path.exists(rec0):
            data0 = np.loadtxt(rec0)
            F0 = tools.rectify(change_shape(data0))
    
            data1 = np.loadtxt(config.DIR_FILES + '2componentes/records/' + str(record) + '_1.txt')
            F1 = tools.rectify(change_shape(data1))
            
            L_TRAI = np.loadtxt(config.DIR_FILES + '2componentes/records/' + str(record) + '.txt')
            
            '''
            Conta a quantidade de batimentos Normal e gera a mesma quantidade de batimentos CPV
            
            i_CPV = np.where(L_TRAI == config.POSITIVE_CLASS)[0] #indices que tem os batimentos CPV
            qtde_Normal = len(L_TRAI) - len(i_CPV)
            CPV_artificial = []
            for i in range(0, qtde_Normal):
                new_CPV = 
                CPV_artificial.append()
            '''
            
                    
            for i in range(0, L_TRAI.size):
                FEATURES0_TRAI.append(F0[i])
                FEATURES1_TRAI.append(F1[i])
                if L_TRAI.size == 1:
                    LABELS_TRAI.append(int(L_TRAI))
                else:
                    LABELS_TRAI.append(L_TRAI[i])
            
    '''
    Carrega os dados de teste
    '''
    print('Lendo Dados Teste')
    for record in v_test:    
        #print('> Registro', record)
        rec0 = config.DIR_FILES + '2componentes/records/' + str(record) + '_0.txt'
    
        '''
        Verifica se o arquivo do registro existe. Pois caso não haja batimentos das arritmias no ECG ele não vai gerar atributos.
        '''
        if os.path.exists(rec0):
            data0 = np.loadtxt(rec0)
            F0 = tools.rectify(change_shape(data0))
        
            data1 = np.loadtxt(config.DIR_FILES + '2componentes/records/' + str(record) + '_1.txt')
            F1 = tools.rectify(change_shape(data1))
        
            L_TEST = np.loadtxt(config.DIR_FILES + '2componentes/records/' + str(record) + '.txt')
                
            for i in range(0, L_TEST.size):
                FEATURES0_TEST.append(F0[i])
                FEATURES1_TEST.append(F1[i])
                if L_TEST.size == 1:
                    LABELS_TEST.append(int(L_TEST))
                else:
                    LABELS_TEST.append(L_TEST[i])
                
    '''
    Altera os registros
    '''
    v_change = v_test[index_change] 
    v_test[index_change] = v_trai[index_change]
    v_trai[index_change] = v_change
    index_change += 1
    
         
    FEATURES0_TRAI = np.array(FEATURES0_TRAI)
    FEATURES1_TRAI = np.array(FEATURES1_TRAI)
    LABELS_TRAI    = np.array(LABELS_TRAI)
    FEATURES0_TEST = np.array(FEATURES0_TEST)
    FEATURES1_TEST = np.array(FEATURES1_TEST)
    LABELS_TEST    = np.array(LABELS_TEST)
    
    '''
    Indução das máquina Bayesianas 
    '''
    clf0 = GaussianNB()
    clf1 = GaussianNB()
    
    clf0.fit(FEATURES0_TRAI, LABELS_TRAI)
    clf1.fit(FEATURES1_TRAI, LABELS_TRAI)
    
    '''
    Cálculo da performance
    '''
    R, Acc, Pr_P, Pr_N, Se, Sp, F_P, F_N, Cm, TP, TN, FP, FN = classification(clf0, clf1, FEATURES0_TEST, FEATURES1_TEST, LABELS_TEST)
    
    Acc_L.append(Acc)
    Pr_P_L.append(Pr_P)
    Pr_N_L.append(Pr_N)
    Se_L.append(Se)
    Sp_L.append(Sp)
    F_P_L.append(F_P)
    F_N_L.append(F_N)
    
    r = '\n######## Dobra ' + str(fold) + ' ########'
    print(r)
    print(R)
          
    r += '\n Registros:'
    r += '\n  Trein: (' + str(L_TRAI.size) + ')' + str(v_trai).replace('\n','')
    r += '\n  Teste: (' + str(L_TEST.size) + ')' + str(v_test).replace('\n','')
    r += '\n Resultados:'
    r += R
    r += '\n& ' + str(fold) + ' & ' + str(np.round(Acc, 4)) + ' & ' + str(np.round(F_P, 4)) + ' & ' + str(np.round(F_N, 4)) + ' & ' + str(np.round(Se, 4)) + ' & ' + str(np.round(Sp, 4)) + ' & ' + str(np.round(Pr_P, 4)) + ' & ' + str(np.round(Pr_N, 4)) + ' & \\\\'
    r += '\n#########################################\n'
    RESULTS += r

r = '\n>>> Resultado médio <<<\n'
r += 'Acc' + str(np.mean(Acc_L)) + '+/-' + str(np.std(Acc_L))
r += 'P+' + str(np.mean(Pr_P_L)) + '+/-' + str(np.std(Pr_P_L))
r += 'P-' + str(np.mean(Pr_N_L)) + '+/-' + str(np.std(Pr_N_L))
r += 'Se' + str(np.mean(Se_L)) + '+/-' + str(np.std(Se_L))
r += 'Sp' + str(np.mean(Sp_L)) + '+/-' + str(np.std(Sp_L))
r += 'F+' + str(np.mean(F_P_L)) + '+/-' + str(np.std(F_P_L))
r += 'F-' + str(np.mean(F_N_L)) + '+/-' + str(np.std(F_N_L))

RESULTS += r

'''
Salva os resultados sobre o conjunto de treinamento
'''
f = open(config.DIR_FILES + '2componentes/RESULTADOS_VALIDACAO_CRUZADA.txt', 'w')
f.write(RESULTS)
f.close()


