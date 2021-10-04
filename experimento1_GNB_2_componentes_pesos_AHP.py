# -*- coding: utf-8 -*-
"""
Neste experimento são utilizadas a base DS1 para treinamento e DS2 para teste e duas componentes ocultas da abordagem proposta. Os parâmetros da Rede Neural são ajustado utilizando o esquema Grid Search. Também são utilizados diferentes valores de subamostragem das componentes.
"""
from sklearn.naive_bayes import GaussianNB
import numpy as np
import config
import measures_performance
import AHP
import tools

################# PLOT #################
import matplotlib 
import matplotlib.pyplot as plt
matplotlib.rc('xtick', labelsize=9) 
matplotlib.rc('ytick', labelsize=9)
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

'''
Carrega os dados de treinamento
'''
features0_train = tools.rectify(np.loadtxt(config.DIR_FILES + config.FILE_FEATURES_TRAINING + '_' + config.DS_TRAINING + '_0.txt'))
features1_train = tools.rectify(np.loadtxt(config.DIR_FILES + config.FILE_FEATURES_TRAINING + '_' + config.DS_TRAINING + '_1.txt'))
targets_train   = np.loadtxt(config.DIR_FILES + config.FILE_TARGET_TRAINING + '_' + config.DS_TRAINING + '.txt')

'''
Carrega os dados de teste
'''
features0_test = tools.rectify(np.loadtxt(config.DIR_FILES + config.FILE_FEATURES_TEST + '_' + config.DS_TEST + '_0.txt'))
features1_test = tools.rectify(np.loadtxt(config.DIR_FILES + config.FILE_FEATURES_TEST + '_' + config.DS_TEST + '_1.txt'))
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

'''
Comite de classificadores utilizando o método AHP
'''      
ACCs  = []
SEs   = []
SPs   = []
PR_Ps = []
PR_Ns = []
F_Ps  = []
F_Ns  = []
Ws    = []

for WEIGHT in [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]:
    #vetor de pesos para as medidas (criterios), utilizado na AHP
    w  = [1.0, WEIGHT, 1.0, WEIGHT, 1.0, WEIGHT, 1.0]
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
    
    Ws.append(w[1])
    
    print('Pesos - medidas classe prositiva: \t', w[1])
    print('Importância classificadores - clf0, clf1: \t', np.round(v, 4))
    
    '''
    Emprega o voto majoritario ponderados pelas prioridades globais obtidas pelo AHP
    '''
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
    Acc, Pr_P, Pr_N, Se, Sp, F_P, F_N, Cm = measures_performance.calc(TP, TN, FP, FN)
    print(' Acc: ', round(Acc, 4))
    print(' F+:  ', round(F_P, 4))
    print(' F-:  ', round(F_N, 4))
    print(' Se:  ', round(Se, 4))
    print(' Sp:  ', round(Sp, 4))
    print(' P+:  ', round(Pr_P, 4))
    print(' P-:  ', round(Pr_N, 4))
    
    ACCs.append(Acc)
    PR_Ps.append(Pr_P)
    PR_Ns.append(Pr_N)
    F_Ps.append(F_P)
    F_Ns.append(F_N)
    SEs.append(Se)
    SPs.append(Sp)

del clf0, clf1
del features0_train, features1_train
del features0_test, features1_test
del targets_test, targets_train

plt.figure(1)
plt.plot(Ws, ACCs, marker='o', label=r"$A_{cc}$")
plt.plot(Ws, SEs, marker='o', label=r'$S_e$')
plt.plot(Ws, SPs, marker='o', label=r'$S_p$')
plt.plot(Ws, PR_Ps, marker='o', label=r'$P^{+}$')
plt.plot(Ws, PR_Ns, marker='o', label=r'$P^{-}$')
plt.plot(Ws, F_Ps, marker='o', label=r'$F(\varsigma)^{+}$')
plt.plot(Ws, F_Ns, marker='o', label=r'$F(\varsigma)^{-}$')
plt.legend()
plt.xlabel(r'Peso relativo as medidas positivas $S_e$, $P^{+}$ e $F(\varsigma)^{+}$')
plt.ylabel('Performance')
plt.grid()
#plt.xticks(Ws)
plt.tight_layout()
plt.show()
plt.savefig('experimento1_pesos_medidas_positivas.png', format='png', dpi=300)

