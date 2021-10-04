#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compara as a performance geral das máquinas utilizando o AHP
"""
import numpy as np
import AHP
import config

'''
Resultados do Exeprimento II
'''
m_clf0   = [0.9651, 0.7654, 0.9811, 0.7018, 0.9883, 0.8417, 0.9740]
m_clf1   = [0.9654, 0.7663, 0.9813, 0.6977, 0.9891, 0.8498, 0.9737]
m_clf2   = [0.9777, 0.8691, 0.9878, 0.9130, 0.9834, 0.8293, 0.9922]
m_suave  = [0.9728, 0.8181, 0.8517, 0.7540, 0.9921, 0.8943, 0.9786]
m_rigida = [0.9694, 0.7956, 0.8378, 0.7335, 0.9903, 0.8693, 0.9768]
m_ahp_03 = [0.9759, 0.8608, 0.9548, 0.9195, 0.9808, 0.8092, 0.9928]
m_ahp_04 = [0.9786, 0.8737, 0.9494, 0.9102, 0.9847, 0.8400, 0.9920]
m_ahp_05 = [0.9797, 0.8784, 0.9441, 0.9012, 0.9867, 0.8568, 0.9912]
m_ahp_06 = [0.9756, 0.8444, 0.8916, 0.8152, 0.9898, 0.8758, 0.9838]
m_ahp_07 = [0.9741, 0.8240, 0.8459, 0.7453, 0.9944, 0.9213, 0.9779]

w  = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
w  = w / np.linalg.norm(w, 1)

#Matrizes de comparações paritárias
m_Acc, m_F1_P, m_F1_N, m_Se, m_Sp, m_Pr_P, m_Pr_N = AHP.pairwiseMatrix(np.array([m_clf0, m_clf1, m_clf2, m_suave, m_rigida, m_ahp_03, m_ahp_04, m_ahp_05, m_ahp_06, m_ahp_07]), kappa=config.KAPPA_AHP)

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

names = ['Clf0    ', 'Clf1    ', 'Clf2    ', 'Suave  ', 'Rígido  ', 'AHP 0,3', 'AHP 0,4', 'AHP 0,5', 'AHP 0,6', 'AHP 0,7']
items = np.flip(np.argsort(v))
o = 1
for i in items:
    print(o, ' - ', names[i], '\t', np.round(v[i]* 100, 2), '%')
    o += 1