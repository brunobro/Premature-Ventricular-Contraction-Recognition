# -*- coding: utf-8 -*-
"""
Neste experimento são utilizadas a base DS1 para treinamento e DS2 para teste e duas componentes ocultas da abordagem proposta.
"""
from sklearn.externals import joblib
import numpy as np
import config
import tools
import measures_performance
import AHP

len_cycle_tra = str(config.LEN_CARDIAC_CYCLE(config.DB_TRAINING))
len_cycle_tes = str(config.LEN_CARDIAC_CYCLE(config.DB_TEST))

'''
Carrega os dados de teste
'''
features0_test = tools.rectify(np.loadtxt(config.DIR_FILES + '2componentes/' + config.FILE_FEATURES_TEST + '_' + config.DS_TEST + '_' + len_cycle_tes + '_0.txt'))
features1_test = tools.rectify(np.loadtxt(config.DIR_FILES + '2componentes/' + config.FILE_FEATURES_TEST + '_' + config.DS_TEST + '_' + len_cycle_tes + '_1.txt'))
targets_test   = np.loadtxt(config.DIR_FILES + config.FILE_TARGET_TEST + '_' + config.DS_TEST + '_' + len_cycle_tes + '.txt')

'''
Carrega as maquinas
'''
clf0 = joblib.load(config.DIR_FILES_ALL_MACHINES + 'clf_I0.sav')
clf1 = joblib.load(config.DIR_FILES_ALL_MACHINES + 'clf_I1.sav')
    
predicted_values0 = clf0.predict(features0_test)
predicted_values1 = clf1.predict(features1_test)

predicted_values0_proba = clf0.predict_proba(features0_test)
predicted_values1_proba = clf1.predict_proba(features1_test)

'''
Computa as medidas de performance para cada máquina
para utilizar os valores no voto AHP
'''
M0 = measures_performance.every(predicted_values0, targets_test, config.POSITIVE_CLASS, config.NEGATIVE_CLASS)
M1 = measures_performance.every(predicted_values1, targets_test, config.POSITIVE_CLASS, config.NEGATIVE_CLASS)

'''
AHP
'''
#vetor de pesos para as medidas (criterios), utilizado na AHP
w  = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
w  = w / np.linalg.norm(w, 1)

#Resultados
M = [M0, M1]
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
     
'''
Compara as máquinas
'''
C_01  = np.zeros((2, 2)) #máquinas Clf0, Clf1
C_0S  = np.zeros((2, 2)) #máquinas Clf0, Voto Suave
C_0HN = np.zeros((2, 2)) #máquinas Clf0, Voto Rígido N
C_0HP = np.zeros((2, 2)) #máquinas Clf0, Voto Rígido P
C_0A1 = np.zeros((2, 2)) #máquinas Clf0, Voto AHP
C_0A2 = np.zeros((2, 2)) #máquinas Clf0, Voto AHP
C_0A3 = np.zeros((2, 2)) #máquinas Clf0, Voto AHP

C_1S  = np.zeros((2, 2)) #máquinas Clf1, Voto Suave
C_1HN = np.zeros((2, 2)) #máquinas Clf1, Voto Rígido N
C_1HP = np.zeros((2, 2)) #máquinas Clf1, Voto Rígido P
C_1A1 = np.zeros((2, 2)) #máquinas Clf0, Voto AHP
C_1A2 = np.zeros((2, 2)) #máquinas Clf0, Voto AHP
C_1A3 = np.zeros((2, 2)) #máquinas Clf0, Voto AHP

C_SHN = np.zeros((2, 2)) #máquinas Voto Suave, Voto Rígido N
C_SHP = np.zeros((2, 2)) #máquinas Voto Suave, Voto Rígido P
C_SA1 = np.zeros((2, 2)) #máquinas Clf0, Voto AHP
C_SA2 = np.zeros((2, 2)) #máquinas Clf0, Voto AHP
C_SA3 = np.zeros((2, 2)) #máquinas Clf0, Voto AHP

C_HNHP = np.zeros((2, 2)) #máquinas Voto Rígido N, Voto Rígido P
C_HNA  = np.zeros((2, 2)) #máquinas Voto Rígido N, Voto AHP
C_HNA1 = np.zeros((2, 2)) #máquinas Clf0, Voto AHP
C_HNA2 = np.zeros((2, 2)) #máquinas Clf0, Voto AHP
C_HNA3 = np.zeros((2, 2)) #máquinas Clf0, Voto AHP

C_HPA  = np.zeros((2, 2)) #máquinas Voto Rígido P, Voto AHP
C_HPA1 = np.zeros((2, 2)) #máquinas Clf0, Voto AHP
C_HPA2 = np.zeros((2, 2)) #máquinas Clf0, Voto AHP
C_HPA3 = np.zeros((2, 2)) #máquinas Clf0, Voto AHP

for k in range(0, targets_test.shape[0]):

    '''
    Predições
    '''
    predicted_clf0 = clf0.predict(features0_test[k].reshape(-1, 1).T)
    predicted_clf1 = clf1.predict(features1_test[k].reshape(-1, 1).T)
    
    predicted_proba_clf0 = clf0.predict_proba(features0_test[k].reshape(-1, 1).T)
    predicted_proba_clf1 = clf1.predict_proba(features1_test[k].reshape(-1, 1).T)
    
    '''
    Voto majoritário Suave
    '''
    M_proba = predicted_proba_clf0[0] + predicted_proba_clf1[0]
    predicted_soft = 0
    if  M_proba[1] > M_proba[0]:
        predicted_soft = 1   

    '''
    Voto majoritário Rígido
    '''
    ps = np.array([predicted_clf0, predicted_clf1])
    total_pos = len(np.where(ps == 1)[0])
    total_neg = len(np.where(ps == 0)[0])
    '''
    Voto majoritário Rígido - N
    '''
    predicted_hard_N = 0
    if  total_pos > total_neg:
        predicted_hard_N = 1
    '''
    Voto majoritário Rígido - P
    '''
    predicted_hard_P = 1
    if  total_pos < total_neg:
        predicted_hard_P = 0
        
    '''
    Voto AHP
    '''
    M_proba_AHP = (v[0] * predicted_proba_clf0[0] + v[1] * predicted_proba_clf1[0])/(v[0] + v[1])
    predicted_AHP1 = 0
    predicted_AHP2 = 0
    predicted_AHP3 = 0
    if M_proba_AHP[1] > 0.3:
        predicted_AHP1 = 1
    if M_proba_AHP[1] > 0.5:
        predicted_AHP2 = 1
    if M_proba_AHP[1] > 0.7:
        predicted_AHP3 = 1
    
    '''
    Rótulo correto
    '''            
    target_TRUE = int(targets_test[k])
    
    
    '''
    Comparação Clf0, Clf1
    '''
    if predicted_clf0 != target_TRUE:
        if predicted_clf1 != target_TRUE:
            C_01[0,0] += 1
        else:
            C_01[0,1] += 1
    else:
        if predicted_clf1 != target_TRUE:
            C_01[1,0] += 1
        else:
            C_01[1,1] += 1
    
    '''
    Comparação Clf0, voto Suave
    '''
    if predicted_clf0 != target_TRUE:
        if predicted_soft != target_TRUE:
            C_0S[0,0] += 1
        else:
            C_0S[0,1] += 1
    else:
        if predicted_soft != target_TRUE:
            C_0S[1,0] += 1
        else:
            C_0S[1,1] += 1
            
    '''
    Comparação Clf0, voto Rígido - N
    '''
    if predicted_clf0 != target_TRUE:
        if predicted_hard_N != target_TRUE:
            C_0HN[0,0] += 1
        else:
            C_0HN[0,1] += 1
    else:
        if predicted_hard_N != target_TRUE:
            C_0HN[1,0] += 1
        else:
            C_0HN[1,1] += 1
            
    '''
    Comparação Clf0, voto Rígido - P
    '''
    if predicted_clf0 != target_TRUE:
        if predicted_hard_P != target_TRUE:
            C_0HP[0,0] += 1
        else:
            C_0HP[0,1] += 1
    else:
        if predicted_hard_P != target_TRUE:
            C_0HP[1,0] += 1
        else:
            C_0HP[1,1] += 1
            
    '''
    Comparação Clf0, voto AHP
    '''
    if predicted_clf0 != target_TRUE:
        if predicted_AHP1 != target_TRUE:
            C_0A1[0,0] += 1
        else:
            C_0A1[0,1] += 1
    else:
        if predicted_AHP1 != target_TRUE:
            C_0A1[1,0] += 1
        else:
            C_0A1[1,1] += 1
            
    if predicted_clf0 != target_TRUE:
        if predicted_AHP2 != target_TRUE:
            C_0A2[0,0] += 1
        else:
            C_0A2[0,1] += 1
    else:
        if predicted_AHP2 != target_TRUE:
            C_0A2[1,0] += 1
        else:
            C_0A2[1,1] += 1
            
    if predicted_clf0 != target_TRUE:
        if predicted_AHP3 != target_TRUE:
            C_0A3[0,0] += 1
        else:
            C_0A3[0,1] += 1
    else:
        if predicted_AHP3 != target_TRUE:
            C_0A3[1,0] += 1
        else:
            C_0A3[1,1] += 1
            
    '''
    Comparação Clf1, voto Suave
    '''
    if predicted_clf1 != target_TRUE:
        if predicted_soft != target_TRUE:
            C_1S[0,0] += 1
        else:
            C_1S[0,1] += 1
    else:
        if predicted_soft != target_TRUE:
            C_1S[1,0] += 1
        else:
            C_1S[1,1] += 1
            
    '''
    Comparação Clf1, voto Rígido - N
    '''
    if predicted_clf1 != target_TRUE:
        if predicted_hard_N != target_TRUE:
            C_1HN[0,0] += 1
        else:
            C_1HN[0,1] += 1
    else:
        if predicted_hard_N != target_TRUE:
            C_1HN[1,0] += 1
        else:
            C_1HN[1,1] += 1
            
    '''
    Comparação Clf1, voto Rígido - p
    '''
    if predicted_clf1 != target_TRUE:
        if predicted_hard_P != target_TRUE:
            C_1HP[0,0] += 1
        else:
            C_1HP[0,1] += 1
    else:
        if predicted_hard_P != target_TRUE:
            C_1HP[1,0] += 1
        else:
            C_1HP[1,1] += 1
            
    '''
    Comparação Clf1, voto AHP
    '''
    if predicted_clf1 != target_TRUE:
        if predicted_AHP1 != target_TRUE:
            C_1A1[0,0] += 1
        else:
            C_1A1[0,1] += 1
    else:
        if predicted_AHP1 != target_TRUE:
            C_1A1[1,0] += 1
        else:
            C_1A1[1,1] += 1
            
    if predicted_clf1 != target_TRUE:
        if predicted_AHP2 != target_TRUE:
            C_1A2[0,0] += 1
        else:
            C_1A2[0,1] += 1
    else:
        if predicted_AHP2 != target_TRUE:
            C_1A2[1,0] += 1
        else:
            C_1A2[1,1] += 1
            
    if predicted_clf1 != target_TRUE:
        if predicted_AHP3 != target_TRUE:
            C_1A3[0,0] += 1
        else:
            C_1A3[0,1] += 1
    else:
        if predicted_AHP3 != target_TRUE:
            C_1A3[1,0] += 1
        else:
            C_1A3[1,1] += 1
            
    '''
    Comparação voto Suave, voto Rígido - N
    '''
    if predicted_soft != target_TRUE:
        if predicted_hard_N != target_TRUE:
            C_SHN[0,0] += 1
        else:
            C_SHN[0,1] += 1
    else:
        if predicted_hard_N != target_TRUE:
            C_SHN[1,0] += 1
        else:
            C_SHN[1,1] += 1
            
    '''
    Comparação voto Suave, voto Rígido - P
    '''
    if predicted_soft != target_TRUE:
        if predicted_hard_P != target_TRUE:
            C_SHP[0,0] += 1
        else:
            C_SHP[0,1] += 1
    else:
        if predicted_hard_P != target_TRUE:
            C_SHP[1,0] += 1
        else:
            C_SHP[1,1] += 1
            
    '''
    Comparação voto Suave, voto AHP
    '''
    if predicted_soft != target_TRUE:
        if predicted_AHP1 != target_TRUE:
            C_SA1[0,0] += 1
        else:
            C_SA1[0,1] += 1
    else:
        if predicted_AHP1 != target_TRUE:
            C_SA1[1,0] += 1
        else:
            C_SA1[1,1] += 1
            
    if predicted_soft != target_TRUE:
        if predicted_AHP2 != target_TRUE:
            C_SA2[0,0] += 1
        else:
            C_SA2[0,1] += 1
    else:
        if predicted_AHP2 != target_TRUE:
            C_SA2[1,0] += 1
        else:
            C_SA2[1,1] += 1
            
    if predicted_soft != target_TRUE:
        if predicted_AHP3 != target_TRUE:
            C_SA3[0,0] += 1
        else:
            C_SA3[0,1] += 1
    else:
        if predicted_AHP3 != target_TRUE:
            C_SA3[1,0] += 1
        else:
            C_SA3[1,1] += 1
            
    '''
    Comparação voto Rígido - N, voto Rígido - P
    '''
    if predicted_hard_N != target_TRUE:
        if predicted_hard_P != target_TRUE:
            C_HNHP[0,0] += 1
        else:
            C_HNHP[0,1] += 1
    else:
        if predicted_hard_P != target_TRUE:
            C_HNHP[1,0] += 1
        else:
            C_HNHP[1,1] += 1
            
    '''
    Comparação voto Rígido - N, voto AHP
    '''
    if predicted_hard_N != target_TRUE:
        if predicted_AHP1 != target_TRUE:
            C_HNA1[0,0] += 1
        else:
            C_HNA1[0,1] += 1
    else:
        if predicted_AHP1 != target_TRUE:
            C_HNA1[1,0] += 1
        else:
            C_HNA1[1,1] += 1
            
    if predicted_hard_N != target_TRUE:
        if predicted_AHP2 != target_TRUE:
            C_HNA2[0,0] += 1
        else:
            C_HNA2[0,1] += 1
    else:
        if predicted_AHP2 != target_TRUE:
            C_HNA2[1,0] += 1
        else:
            C_HNA2[1,1] += 1
            
    if predicted_hard_N != target_TRUE:
        if predicted_AHP3 != target_TRUE:
            C_HNA3[0,0] += 1
        else:
            C_HNA3[0,1] += 1
    else:
        if predicted_AHP3 != target_TRUE:
            C_HNA3[1,0] += 1
        else:
            C_HNA3[1,1] += 1
            
    '''
    Comparação voto Rígido - P, voto AHP
    '''
    if predicted_hard_P != target_TRUE:
        if predicted_AHP1 != target_TRUE:
            C_HPA1[0,0] += 1
        else:
            C_HPA1[0,1] += 1
    else:
        if predicted_AHP1 != target_TRUE:
            C_HPA1[1,0] += 1
        else:
            C_HPA1[1,1] += 1
            
    if predicted_hard_P != target_TRUE:
        if predicted_AHP2 != target_TRUE:
            C_HPA2[0,0] += 1
        else:
            C_HPA2[0,1] += 1
    else:
        if predicted_AHP2 != target_TRUE:
            C_HPA2[1,0] += 1
        else:
            C_HPA2[1,1] += 1
            
    if predicted_hard_P != target_TRUE:
        if predicted_AHP3 != target_TRUE:
            C_HPA3[0,0] += 1
        else:
            C_HPA3[0,1] += 1
    else:
        if predicted_AHP3 != target_TRUE:
            C_HPA3[1,0] += 1
        else:
            C_HPA3[1,1] += 1 


'''
Calcula as estatisticas
'''
z_01 = (abs(C_01[0,1] - C_01[1,0]) - 1)/np.sqrt(C_01[0,1] + C_01[1,0])
z_0S = (abs(C_0S[0,1] - C_0S[1,0]) - 1)/np.sqrt(C_0S[0,1] + C_0S[1,0])
z_0HN = (abs(C_0HN[0,1] - C_0HN[1,0]) - 1)/np.sqrt(C_0HN[0,1] + C_0HN[1,0])
z_0HP = (abs(C_0HP[0,1] - C_0HP[1,0]) - 1)/np.sqrt(C_0HP[0,1] + C_0HP[1,0])
z_0A1 = (abs(C_0A1[0,1] - C_0A1[1,0]) - 1)/np.sqrt(C_0A1[0,1] + C_0A1[1,0])
z_0A2 = (abs(C_0A2[0,1] - C_0A2[1,0]) - 1)/np.sqrt(C_0A2[0,1] + C_0A2[1,0])
z_0A3 = (abs(C_0A3[0,1] - C_0A3[1,0]) - 1)/np.sqrt(C_0A3[0,1] + C_0A3[1,0])

z_1S  = (abs(C_1S[0,1] - C_1S[1,0]) - 1)/np.sqrt(C_1S[0,1] + C_1S[1,0])
z_1HN = (abs(C_1HN[0,1] - C_1HN[1,0]) - 1)/np.sqrt(C_1HN[0,1] + C_1HN[1,0])
z_1HP = (abs(C_1HP[0,1] - C_1HP[1,0]) - 1)/np.sqrt(C_1HP[0,1] + C_1HP[1,0])
z_1A1  = (abs(C_1A1[0,1] - C_1A1[1,0]) - 1)/np.sqrt(C_1A1[0,1] + C_1A1[1,0])
z_1A2  = (abs(C_1A2[0,1] - C_1A2[1,0]) - 1)/np.sqrt(C_1A2[0,1] + C_1A2[1,0])
z_1A3  = (abs(C_1A3[0,1] - C_1A3[1,0]) - 1)/np.sqrt(C_1A3[0,1] + C_1A3[1,0])

z_SHN = (abs(C_SHN[0,1] - C_SHN[1,0]) - 1)/np.sqrt(C_SHN[0,1] + C_SHN[1,0])
z_SHP = (abs(C_SHP[0,1] - C_SHP[1,0]) - 1)/np.sqrt(C_SHP[0,1] + C_SHP[1,0])
z_SA1  = (abs(C_SA1[0,1] - C_SA1[1,0]) - 1)/np.sqrt(C_SA1[0,1] + C_SA1[1,0])
z_SA2  = (abs(C_SA2[0,1] - C_SA2[1,0]) - 1)/np.sqrt(C_SA2[0,1] + C_SA2[1,0])
z_SA3  = (abs(C_SA3[0,1] - C_SA3[1,0]) - 1)/np.sqrt(C_SA3[0,1] + C_SA3[1,0])

z_HNHP = (abs(C_HNHP[0,1] - C_HNHP[1,0]) - 1)/np.sqrt(C_HNHP[0,1] + C_HNHP[1,0])
z_HNA1  = (abs(C_HNA1[0,1] - C_HNA1[1,0]) - 1)/np.sqrt(C_HNA1[0,1] + C_HNA1[1,0])
z_HNA2  = (abs(C_HNA2[0,1] - C_HNA2[1,0]) - 1)/np.sqrt(C_HNA2[0,1] + C_HNA2[1,0])
z_HNA3  = (abs(C_HNA3[0,1] - C_HNA3[1,0]) - 1)/np.sqrt(C_HNA3[0,1] + C_HNA3[1,0])

z_HPA1  = (abs(C_HPA1[0,1] - C_HPA1[1,0]) - 1)/np.sqrt(C_HPA1[0,1] + C_HPA1[1,0])
z_HPA2  = (abs(C_HPA2[0,1] - C_HPA2[1,0]) - 1)/np.sqrt(C_HPA2[0,1] + C_HPA2[1,0])
z_HPA3  = (abs(C_HPA3[0,1] - C_HPA3[1,0]) - 1)/np.sqrt(C_HPA3[0,1] + C_HPA3[1,0])

H0 = '\tMesma taxa de Erro'
H1 = '\tDiferentes taxas de Erro'

#chi = 1.323  #75%
#chi = 3.841 #95%
chi = 6.635  #99%
#chi = 7.879  #99,5%

if z_01 > chi:
    print('Clf0-Clf1: ', H1)
    print('z: ', z_01)
else:
    print('Clf0-Clf1: ', H0)
    print('z: ', z_01)
    
if z_0S > chi:
    print('Clf0-Suave: ', H1)
    print('z: ', z_0S)
else:
    print('Clf0-Suave: ', H0)
    print('z: ', z_0S)
    
if z_0HN > chi:
    print('Clf0-HN: ', H1)
    print('z: ', z_0HN)
else:
    print('Clf0-HN: ', H0)
    print('z: ', z_0HN)
    
if z_0HP > chi:
    print('Clf0-HP: ', H1)
    print('z: ', z_0HP)
else:
    print('Clf0-HP: ', H0)
    print('z: ', z_0HP)
    
if z_0A1 > chi:
    print('Clf0-AHP: ', H1)
    print('z: ', z_0A1)
else:
    print('Clf0-AHP: ', H0)
    print('z: ', z_0A1)
    
if z_0A2 > chi:
    print('Clf0-AHP: ', H1)
    print('z: ', z_0A2)
else:
    print('Clf0-AHP: ', H0)
    print('z: ', z_0A2)
    
if z_0A3 > chi:
    print('Clf0-AHP: ', H1)
    print('z: ', z_0A3)
else:
    print('Clf0-AHP: ', H0)
    print('z: ', z_0A3)
    
if z_1S > chi:
    print('Clf1-Suave: ', H1)
    print('z: ', z_1S)
else:
    print('Clf1-Suave: ', H0)
    print('z: ', z_1S)
    
if z_1HN > chi:
    print('Clf1-HN: ', H1)
    print('z: ', z_1HN)
else:
    print('Clf1-HN: ', H0)
    print('z: ', z_1HN)
    
if z_1HP > chi:
    print('Clf1-HP: ', H1)
    print('z: ', z_1HP)
else:
    print('Clf1-HP: ', H0)
    print('z: ', z_1HP)
    
if z_1A1 > chi:
    print('Clf1-AHP: ', H1)
    print('z: ', z_1A1)
else:
    print('Clf1-AHP: ', H0)
    print('z: ', z_1A1)
    
if z_1A2 > chi:
    print('Clf1-AHP: ', H1)
    print('z: ', z_1A2)
else:
    print('Clf1-AHP: ', H0)
    print('z: ', z_1A2)
    
if z_1A3 > chi:
    print('Clf1-AHP: ', H1)
    print('z: ', z_1A3)
else:
    print('Clf1-AHP: ', H0)
    print('z: ', z_1A3)
    
if z_SHN > chi:
    print('Suave-HN: ', H1)
    print('z: ', z_SHN)
else:
    print('Suave-HN: ', H0)
    print('z: ', z_SHN)
    
if z_SHP > chi:
    print('Suave-HP: ', H1)
    print('z: ', z_SHP)
else:
    print('Suave-HP: ', H0)
    print('z: ', z_SHP)
    
if z_SA1 > chi:
    print('Suave-AHP: ', H1)
    print('z: ', z_SA1)
else:
    print('Suave-AHP: ', H0)
    print('z: ', z_SA1)
    
if z_SA2 > chi:
    print('Suave-AHP: ', H1)
    print('z: ', z_SA2)
else:
    print('Suave-AHP: ', H0)
    print('z: ', z_SA2)
    
if z_SA3 > chi:
    print('Suave-AHP: ', H1)
    print('z: ', z_SA3)
else:
    print('Suave-AHP: ', H0)
    print('z: ', z_SA3)
    
if z_HNHP > chi:
    print('HN-HP: ', H1)
    print('z: ', z_HNHP)
else:
    print('HN-HP: ', H0)
    print('z: ', z_HNHP)
    
if z_HNA1 > chi:
    print('HN-AHP: ', H1)
    print('z: ', z_HNA1)
else:
    print('HN-AHP: ', H0)
    print('z: ', z_HNA1)
    
if z_HNA2 > chi:
    print('HN-AHP: ', H1)
    print('z: ', z_HNA2)
else:
    print('HN-AHP: ', H0)
    print('z: ', z_HNA2)
    
if z_HNA3 > chi:
    print('HN-AHP: ', H1)
    print('z: ', z_HNA3)
else:
    print('HN-AHP: ', H0)
    print('z: ', z_HNA3)
    
if z_HPA1 > chi:
    print('HP-AHP: ', H1)
    print('z: ', z_HPA1)
else:
    print('HP-AHP: ', H0)
    print('z: ', z_HPA1)
    
if z_HPA2 > chi:
    print('HP-AHP: ', H1)
    print('z: ', z_HPA2)
else:
    print('HP-AHP: ', H0)
    print('z: ', z_HPA2)
    
if z_HPA3 > chi:
    print('HP-AHP: ', H1)
    print('z: ', z_HPA3)
else:
    print('HP-AHP: ', H0)
    print('z: ', z_HPA3)
#del clf0, clf1
#del features0_test, features1_test, targets_test