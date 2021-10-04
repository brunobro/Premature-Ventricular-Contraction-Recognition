#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Analityc Hierarchy Process

@author: Bruno R. de Oliveira <bruno@cerradosites.com>
"""

import numpy as np
from fractions import Fraction
np.set_printoptions(formatter={'all': lambda x: format(x, '8.4f')})

'''
Converte a diferença dos resultados para a escala de Saaty
'''
def scale(c1, c2, kappa):
    
    s = []
    
    for i in range(0, len(c1)):
    
        diff  = (c1[i] - c2[i]) * kappa
        
        if abs(diff) < 1.0:
            delta = 1.0
        elif abs(diff) > 9.0:
            delta = 9.0
        else:
            delta = np.ceil(abs(diff))
            
        if diff > 0:
            DELTA = delta
        else:
            DELTA = 1.0 / delta
        
        s.append(DELTA)
    
    return s

def localVector(matrix):    
    '''
    Calcula vetor de prioridades
    '''
    e_vals, e_vecs = np.linalg.eig(matrix)
    lamb = np.real(max(e_vals))
    w = np.abs(np.real(e_vecs[:, e_vals.argmax()]))
    w = w / np.linalg.norm(w, 1)
    
    #Obtém a razão de consistência
    n = matrix.shape[0]
    ri = {1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49, 11: 1.51}
    ci = (lamb - n) / (n - 1)
    cr = ci / (ri[n] + 1e-10)
    
    return w, cr

def globalVector(w, ws):
    #Calcula o vetor de prioridades global
    ws = ws.T
    V = []
    for p in range(0, ws.shape[0]):
        v = 0
        for j in range(0, len(w)):
            v += w[j] * ws[p,j]
        V.append(v)
    
    V = np.array(V)
    V = V / np.linalg.norm(V, 1)
    return V

def pairwiseMatrix(measures, kappa = 20):
    '''
    Calcula as matrizes de comparações paritárias
    '''
    M       = measures.shape[0]
    m_Acc   = np.identity(M)
    m_F1_P  = np.identity(M)
    m_F1_N  = np.identity(M)
    m_Se    = np.identity(M)
    m_Sp    = np.identity(M)
    m_Pr_P  = np.identity(M)
    m_Pr_N  = np.identity(M)

    i = 0
    J = 1
    for k in range(0, M - 1):
        
        j = J
        for q in range(k + 1, M):
            s = scale(measures[k], measures[q], kappa)
            
            m_Acc[i, j]  = s[0]
            m_F1_P[i, j] = s[1]
            m_F1_N[i, j] = s[2]
            m_Se[i, j]   = s[3]
            m_Sp[i, j]   = s[4]
            m_Pr_P[i, j] = s[5]
            m_Pr_N[i, j] = s[6]
            
            m_Acc[j, i]  = 1.0/s[0]
            m_F1_P[j, i] = 1.0/s[1]
            m_F1_N[j, i] = 1.0/s[2]
            m_Se[j, i]   = 1.0/s[3]
            m_Sp[j, i]   = 1.0/s[4]
            m_Pr_P[j, i] = 1.0/s[5]
            m_Pr_N[j, i] = 1.0/s[6]
            
            j += 1
            
        J += 1
        i += 1
    
    return m_Acc, m_F1_P, m_F1_N, m_Se, m_Sp, m_Pr_P, m_Pr_N
    
def printMatrix(pcm):   
    '''
    Imprime matrix de comparações paraitárias
    '''
    for r in range(0, len(pcm)):
        i = 1 
        for l in pcm[r,:]:
            char = '&'
            if i == len(pcm):
                char = ''
            print((Fraction.from_float(l)).limit_denominator(10), char)
            i += 1
        print('\\\\')
        
def printVector(w):
    for r in range(0, len(w)):
        print(np.round(w[r], 4), '\\\\')

