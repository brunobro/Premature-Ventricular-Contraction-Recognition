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
m_clf0   = [0.9427, 0.5690, 0.9693, 0.4660, 0.9848, 0.7306, 0.9543]
m_clf1   = [0.9521, 0.6339, 0.9743, 0.5113, 0.9910, 0.8338, 0.9582]
m_clf2   = [0.9318, 0.5906, 0.9628, 0.6058, 0.9606, 0.5761, 0.9650]
m_clf3   = [0.9499, 0.6941, 0.9727, 0.6999, 0.9720, 0.6884, 0.9734]
m_suave  = [0.9730, 0.8089, 0.8167, 0.7030, 0.9969, 0.9524, 0.9744]
m_rigida = [0.9501, 0.5633, 0.5590, 0.3961, 0.9991, 0.9748, 0.9493]
m_ahp_03 = [0.9615, 0.7808, 0.9093, 0.8437, 0.9719, 0.7266, 0.9860]
m_ahp_04 = [0.9725, 0.8254, 0.8830, 0.8018, 0.9875, 0.8504, 0.9826]
m_ahp_05 = [0.9730, 0.8116, 0.8254, 0.7154, 0.9958, 0.9377, 0.9754]
m_ahp_06 = [0.9558, 0.6323, 0.6283, 0.4682, 0.9989, 0.9735, 0.9551]
m_ahp_07 = [0.9491, 0.5469, 0.5409, 0.3784, 0.9995, 0.9862, 0.9479]

w  = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
w  = w / np.linalg.norm(w, 1)

#Matrizes de comparações paritárias
m_Acc, m_F1_P, m_F1_N, m_Se, m_Sp, m_Pr_P, m_Pr_N = AHP.pairwiseMatrix(np.array([m_clf0, m_clf1, m_clf2, m_clf3, m_suave, m_rigida, m_ahp_03, m_ahp_04, m_ahp_05, m_ahp_06, m_ahp_07]), kappa=config.KAPPA_AHP)

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

names = ['Clf0    ', 'Clf1    ', 'Clf2    ', 'Clf3    ', 'Suave  ', 'Rígido  ', 'AHP 0,3', 'AHP 0,4', 'AHP 0,5', 'AHP 0,6', 'AHP 0,7']
items = np.flip(np.argsort(v))
o = 1
for i in items:
    print(o, ' - ', names[i], '\t', np.round(v[i]* 100, 2), '%')
    o += 1