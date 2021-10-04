# -*- coding: utf-8 -*-
"""
Neste experimento são utilizadas a base DS1 para treinamento e DS2 BALANCEADAS
tomando a média de duas componentes para produzir outra artificial
para teste e duas componentes ocultas da abordagem proposta. 
Com validação cruzada.
"""
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
import numpy as np
import config
import measures_performance
import AHP
import tools

ARQUIVO_SAIDA = open('resultados_exp1_bal_vc.txt', 'w')

def print_measures(TP, TN, FP, FN):   
    Acc, Pr_P, Pr_N, Se, Sp, F_P, F_N, Cm = measures_performance.calc(TP, TN, FP, FN)
    print(' Acc: ', round(Acc, 4), file=ARQUIVO_SAIDA)
    print(' F+:  ', round(F_P, 4), file=ARQUIVO_SAIDA)
    print(' F-:  ', round(F_N, 4), file=ARQUIVO_SAIDA)
    print(' Se:  ', round(Se, 4), file=ARQUIVO_SAIDA)
    print(' Sp:  ', round(Sp, 4), file=ARQUIVO_SAIDA)
    print(' P+:  ', round(Pr_P, 4), file=ARQUIVO_SAIDA)
    print(' P-:  ', round(Pr_N, 4), file=ARQUIVO_SAIDA)
    return Acc, Pr_P, Pr_N, Se, Sp, F_P, F_N


len_cycle_tra = str(config.LEN_CARDIAC_CYCLE(config.DB_TRAINING))
len_cycle_tes = str(config.LEN_CARDIAC_CYCLE(config.DB_TEST))

def amostras_artificiais(features, targets, return_target=True):
    
    normal_beat = len(np.where(targets == 0)[0])
    cpv_beat    = len(np.where(targets == 1)[0])
    
    ids_CPV = np.where(targets == 1)[0]
    N       = len(ids_CPV)
    
    K = 0
    new_features = []
    
    if return_target:
        new_targets  = []
    
    while K < normal_beat - cpv_beat:
        
        i1 = np.random.randint(0, N)
        i2 = np.random.randint(0, N)
        F  = np.mean([features[ids_CPV[i1],:], features[ids_CPV[i2],:]], axis=0)
        
        new_features.append(F)
        
        if return_target:
            new_targets.append(config.POSITIVE_CLASS)
        
        K += 1
        
    new_features = np.array(new_features)
    
    if return_target:
        new_targets  = np.array(new_targets)
    
    if return_target:    
        return np.concatenate((features, new_features)), np.concatenate((targets, new_targets))
    else:
        return np.concatenate((features, new_features))

'''
Carrega os dados de treinamento
'''
features0_train = tools.rectify(np.loadtxt(config.DIR_FILES + '2componentes/' + config.FILE_FEATURES_TRAINING + '_' + config.DS_TRAINING  + '_' + len_cycle_tra + '_0.txt'))
features1_train = tools.rectify(np.loadtxt(config.DIR_FILES + '2componentes/' + config.FILE_FEATURES_TRAINING + '_' + config.DS_TRAINING  + '_' + len_cycle_tra + '_1.txt'))
targets_train   = np.loadtxt(config.DIR_FILES + config.FILE_TARGET_TRAINING + '_' + config.DS_TRAINING  + '_' + len_cycle_tra + '.txt')

'''
Carrega os dados de teste
'''
features0_test = tools.rectify(np.loadtxt(config.DIR_FILES + '2componentes/' + config.FILE_FEATURES_TEST + '_' + config.DS_TEST  + '_' + len_cycle_tra + '_0.txt'))
features1_test = tools.rectify(np.loadtxt(config.DIR_FILES + '2componentes/' + config.FILE_FEATURES_TEST + '_' + config.DS_TEST  + '_' + len_cycle_tra + '_1.txt'))
targets_test   = np.loadtxt(config.DIR_FILES + config.FILE_TARGET_TEST + '_' + config.DS_TEST  + '_' + len_cycle_tra + '.txt')

'''
Concatena os dados
'''
features0 = np.concatenate((features0_train, features0_test))
features1 = np.concatenate((features1_train, features1_test))
targets   = np.concatenate((targets_train, targets_test))

features0, targets_new = amostras_artificiais(features0, targets)
features1              = amostras_artificiais(features1, targets, False)

targets = targets_new

del targets_new
del features0_train, features0_test
del features1_train, features1_test
del targets_train, targets_test

print('Batimentos:', len(features1), file=ARQUIVO_SAIDA)

TAcc_clf0 = TPr_P_clf0 = TPr_N_clf0 = TSe_clf0 = TSp_clf0 = TF_P_clf0 = TF_N_clf0 = []
TAcc_clf1 = TPr_P_clf1 = TPr_N_clf1 = TSe_clf1 = TSp_clf1 = TF_P_clf1 = TF_N_clf1 = []
TAcc_suave = TPr_P_suave = TPr_N_suave = TSe_suave = TSp_suave = TF_P_suave = TF_N_suave = []
TAcc_rigido_n = TPr_P_rigido_n = TPr_N_rigido_n = TSe_rigido_n = TSp_rigido_n = TF_P_rigido_n = TF_N_rigido_n = []
TAcc_rigido_p = TPr_P_rigido_p = TPr_N_rigido_p = TSe_rigido_p = TSp_rigido_p = TF_P_rigido_p = TF_N_rigido_p = []
TAcc_ahp_03 = TPr_P_ahp_03 = TPr_N_ahp_03 = TSe_ahp_03 = TSp_ahp_03 = TF_P_ahp_03 = TF_N_ahp_03 = []
TAcc_ahp_04 = TPr_P_ahp_04 = TPr_N_ahp_04 = TSe_ahp_04 = TSp_ahp_04 = TF_P_ahp_04 = TF_N_ahp_04 = []
TAcc_ahp_05 = TPr_P_ahp_05 = TPr_N_ahp_05 = TSe_ahp_05 = TSp_ahp_05 = TF_P_ahp_05 = TF_N_ahp_05 = []
TAcc_ahp_06 = TPr_P_ahp_06 = TPr_N_ahp_06 = TSe_ahp_06 = TSp_ahp_06 = TF_P_ahp_06 = TF_N_ahp_06 = []
TAcc_ahp_07 = TPr_P_ahp_07 = TPr_N_ahp_07 = TSe_ahp_07 = TSp_ahp_07 = TF_P_ahp_07 = TF_N_ahp_07 = []

kf = KFold(n_splits=500, shuffle=True, random_state=0)

fold = 1

TOTAL_FOLDS       = 0
PERCENTUAL_MINIMO = 0.90

for train_index, test_index in kf.split(features0):
    
    #if TOTAL_FOLDS > 10:
    #    break
    
    print('\n\n******************** Dobra ', fold, ' ********************\n', file=ARQUIVO_SAIDA)
    fold += 1
   
    features0_train, features0_test = features0[train_index], features0[test_index]
    features1_train, features1_test = features1[train_index], features1[test_index]
    targets_train, targets_test     = targets[train_index], targets[test_index]

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
    predicted_values0 = clf0.predict(features0_test)
    predicted_values1 = clf1.predict(features1_test)
    
    predicted_values0_proba = clf0.predict_proba(features0_test)
    predicted_values1_proba = clf1.predict_proba(features1_test)
    
    '''
    Computa as medidas de performance para cada classificador
    '''
    M0 = measures_performance.every(predicted_values0, targets_test, config.POSITIVE_CLASS, config.NEGATIVE_CLASS)
    M1 = measures_performance.every(predicted_values1, targets_test, config.POSITIVE_CLASS, config.NEGATIVE_CLASS)
        
    if M1[0] > PERCENTUAL_MINIMO:
        
        M = [M0, M1]
        
        '''
        Experimento 1: Performance individual de cada classificador
        '''
        print('######################## EXP1 #####################', file=ARQUIVO_SAIDA)
        print('Performance individual para cada classificador', file=ARQUIVO_SAIDA)
        
        i = 0
        for m in M:
    
            print('Clf', i, file=ARQUIVO_SAIDA)
            print(' Acc: ', round(m[0], 4), file=ARQUIVO_SAIDA)
            print(' F+:  ', round(m[1], 4), file=ARQUIVO_SAIDA)
            print(' F-:  ', round(m[2], 4), file=ARQUIVO_SAIDA)
            print(' Se:  ', round(m[3], 4), file=ARQUIVO_SAIDA)
            print(' Sp:  ', round(m[4], 4), file=ARQUIVO_SAIDA)
            print(' P+:  ', round(m[5], 4), file=ARQUIVO_SAIDA)
            print(' P-:  ', round(m[6], 4), file=ARQUIVO_SAIDA)
            
            if i == 0:
                if m[3] > PERCENTUAL_MINIMO and m[4] > PERCENTUAL_MINIMO:
                    TAcc_clf0.append(m[0])
                    TPr_P_clf0.append(m[5])
                    TPr_N_clf0.append(m[6])
                    TSe_clf0.append(m[3])
                    TSp_clf0.append(m[4])
                    TF_P_clf0.append(m[1])
                    TF_N_clf0.append(m[2])
                    TOTAL_FOLDS += 1
            else:
                if m[3] > PERCENTUAL_MINIMO and m[4] > PERCENTUAL_MINIMO:
                    TAcc_clf1.append(m[0])
                    TPr_P_clf1.append(m[5])
                    TPr_N_clf1.append(m[6])
                    TSe_clf1.append(m[3])
                    TSp_clf1.append(m[4])
                    TF_P_clf1.append(m[1])
                    TF_N_clf1.append(m[2])
                    TOTAL_FOLDS += 1
                
            i += 1
                
         
        print('######################## EXP2 #####################', file=ARQUIVO_SAIDA)
        print('Voto majoritário suave', file=ARQUIVO_SAIDA)
            
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
        Acc, Pr_P, Pr_N, Se, Sp, F_P, F_N = print_measures(TP, TN, FP, FN)
        
        if Se > PERCENTUAL_MINIMO and Sp > PERCENTUAL_MINIMO:
            TAcc_suave.append(Acc)
            TPr_P_suave.append(Pr_P)
            TPr_N_suave.append(Pr_N)
            TSe_suave.append(Se)
            TSp_suave.append(Sp)
            TF_P_suave.append(F_P)
            TF_N_suave.append(F_N)
            
            TOTAL_FOLDS += 1
        
        print('######################## EXP3 #####################', file=ARQUIVO_SAIDA)
        print('Voto majoritário rígido - padrão negativo', file=ARQUIVO_SAIDA)
            
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
        Acc, Pr_P, Pr_N, Se, Sp, F_P, F_N = print_measures(TP, TN, FP, FN)
        
        if Se > PERCENTUAL_MINIMO and Sp > PERCENTUAL_MINIMO:
            TAcc_rigido_n.append(Acc)
            TPr_P_rigido_n.append(Pr_P)
            TPr_N_rigido_n.append(Pr_N)
            TSe_rigido_n.append(Se)
            TSp_rigido_n.append(Sp)
            TF_P_rigido_n.append(F_P)
            TF_N_rigido_n.append(F_N)
            
            TOTAL_FOLDS += 1
        
        print('######################## EXP4 #####################', file=ARQUIVO_SAIDA)
        print('Voto majoritário rígido - padrão positivo', file=ARQUIVO_SAIDA)
            
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
        Acc, Pr_P, Pr_N, Se, Sp, F_P, F_N = print_measures(TP, TN, FP, FN)
        
        if Se > PERCENTUAL_MINIMO and Sp > PERCENTUAL_MINIMO:
            TAcc_rigido_p.append(Acc)
            TPr_P_rigido_p.append(Pr_P)
            TPr_N_rigido_p.append(Pr_N)
            TSe_rigido_p.append(Se)
            TSp_rigido_p.append(Sp)
            TF_P_rigido_p.append(F_P)
            TF_N_rigido_p.append(F_N)
            
            TOTAL_FOLDS += 1
        
        '''
        Emprega o voto majoritario ponderados pelas prioridades globais obtidas pelo AHP rígido
        '''
        print('######################## EXP5 #####################', file=ARQUIVO_SAIDA)
            
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
        
        print('Voto AHP - alpha = 0.3', file=ARQUIVO_SAIDA)
        
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
            if M_proba[1] > 0.3: #cut-off limit
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
        Acc, Pr_P, Pr_N, Se, Sp, F_P, F_N = print_measures(TP, TN, FP, FN)
        
        if Se > PERCENTUAL_MINIMO and Sp > PERCENTUAL_MINIMO:
            TAcc_ahp_03.append(Acc)
            TPr_P_ahp_03.append(Pr_P)
            TPr_N_ahp_03.append(Pr_N)
            TSe_ahp_03.append(Se)
            TSp_ahp_03.append(Sp)
            TF_P_ahp_03.append(F_P)
            TF_N_ahp_03.append(F_N)
            
            TOTAL_FOLDS += 1
        
        print('Voto AHP - alpha = 0.4', file=ARQUIVO_SAIDA)
        
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
            if M_proba[1] > 0.4: #cut-off limit
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
        Acc, Pr_P, Pr_N, Se, Sp, F_P, F_N = print_measures(TP, TN, FP, FN)
        
        if Se > PERCENTUAL_MINIMO and Sp > PERCENTUAL_MINIMO:
            TAcc_ahp_04.append(Acc)
            TPr_P_ahp_04.append(Pr_P)
            TPr_N_ahp_04.append(Pr_N)
            TSe_ahp_04.append(Se)
            TSp_ahp_04.append(Sp)
            TF_P_ahp_04.append(F_P)
            TF_N_ahp_04.append(F_N)
            
            TOTAL_FOLDS += 1
        
        print('Voto AHP - alpha = 0.5', file=ARQUIVO_SAIDA)
        
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
        Acc, Pr_P, Pr_N, Se, Sp, F_P, F_N = print_measures(TP, TN, FP, FN)
        
        if Se > PERCENTUAL_MINIMO and Sp > PERCENTUAL_MINIMO:
            TAcc_ahp_05.append(Acc)
            TPr_P_ahp_05.append(Pr_P)
            TPr_N_ahp_05.append(Pr_N)
            TSe_ahp_05.append(Se)
            TSp_ahp_05.append(Sp)
            TF_P_ahp_05.append(F_P)
            TF_N_ahp_05.append(F_N)
            
            TOTAL_FOLDS += 1
        
        print('Voto AHP - alpha = 0.6', file=ARQUIVO_SAIDA)
        
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
            if M_proba[1] > 0.6: #cut-off limit
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
        Acc, Pr_P, Pr_N, Se, Sp, F_P, F_N = print_measures(TP, TN, FP, FN)
        
        if Se > PERCENTUAL_MINIMO and Sp > PERCENTUAL_MINIMO:
            TAcc_ahp_06.append(Acc)
            TPr_P_ahp_06.append(Pr_P)
            TPr_N_ahp_06.append(Pr_N)
            TSe_ahp_06.append(Se)
            TSp_ahp_06.append(Sp)
            TF_P_ahp_06.append(F_P)
            TF_N_ahp_06.append(F_N)
            
            TOTAL_FOLDS += 1
        
        print('Voto AHP - alpha = 0.7', file=ARQUIVO_SAIDA)
        
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
        Acc, Pr_P, Pr_N, Se, Sp, F_P, F_N = print_measures(TP, TN, FP, FN)
        
        if Se > PERCENTUAL_MINIMO and Sp > PERCENTUAL_MINIMO:
            TAcc_ahp_07.append(Acc)
            TPr_P_ahp_07.append(Pr_P)
            TPr_N_ahp_07.append(Pr_N)
            TSe_ahp_07.append(Se)
            TSp_ahp_07.append(Sp)
            TF_P_ahp_07.append(F_P)
            TF_N_ahp_07.append(F_N)
            
            TOTAL_FOLDS += 1
        
        #print('\nDados AHP')
        #print('clf0, clf1: ', np.round(v, 4))

def print_mean(L):   
    
    t = ['Acc', 'P+', 'P-', 'Se', 'Sp', 'F+', 'F-']
    for i in range(0, len(L)):
        m = np.mean(L[i])
        print(t[i], round(m, 4))

print('\n\n############# Valores Médios #############\n')

print('\nClf0')
print_mean([TAcc_clf0, TPr_P_clf0, TPr_N_clf0, TSe_clf0, TSp_clf0, TF_P_clf0, TF_N_clf0])

print('\nClf1')
print_mean([TAcc_clf1, TPr_P_clf1, TPr_N_clf1, TSe_clf1, TSp_clf1, TF_P_clf1, TF_N_clf1])

print('\nVoto Suave')
print_mean([TAcc_suave, TPr_P_suave, TPr_N_suave, TSe_suave, TSp_suave, TF_P_suave, TF_N_suave])

print('\nVoto Rígido - N')
print_mean([TAcc_rigido_n, TPr_P_rigido_n, TPr_N_rigido_n, TSe_rigido_n, TSp_rigido_n, TF_P_rigido_n, TF_N_rigido_n])

print('\nVoto Rígido - P')
print_mean([TAcc_rigido_p, TPr_P_rigido_p, TPr_N_rigido_p, TSe_rigido_p, TSp_rigido_p, TF_P_rigido_p, TF_N_rigido_p])

print('\nVoto AHP - alfa = 0.3')
print_mean([TAcc_ahp_03, TPr_P_ahp_03, TPr_N_ahp_03, TSe_ahp_03, TSp_ahp_03, TF_P_ahp_03, TF_N_ahp_03])

print('\nVoto AHP - alfa = 0.4')
print_mean([TAcc_ahp_04, TPr_P_ahp_04, TPr_N_ahp_04, TSe_ahp_04, TSp_ahp_04, TF_P_ahp_04, TF_N_ahp_04])

print('\nVoto AHP - alfa = 0.5')
print_mean([TAcc_ahp_05, TPr_P_ahp_05, TPr_N_ahp_05, TSe_ahp_05, TSp_ahp_05, TF_P_ahp_05, TF_N_ahp_05])

print('\nVoto AHP - alfa = 0.6')
print_mean([TAcc_ahp_06, TPr_P_ahp_06, TPr_N_ahp_06, TSe_ahp_06, TSp_ahp_06, TF_P_ahp_06, TF_N_ahp_06])

print('\nVoto AHP - alfa = 0.7')
print_mean([TAcc_ahp_07, TPr_P_ahp_07, TPr_N_ahp_07, TSe_ahp_07, TSp_ahp_07, TF_P_ahp_07, TF_N_ahp_07])