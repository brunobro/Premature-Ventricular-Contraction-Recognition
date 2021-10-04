#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Implementa o método AHP sobre os resultados
"""

import numpy as np
from fractions import Fraction
np.set_printoptions(formatter={'all': lambda x: format(x, '8.4f')})

def scale(c1, c2, kappa = 20):
    '''
    Converte a diferença dos resultados para a escala de Saaty
    '''
    s = []
    for i in range(0, len(c1)):
        diff  = c1[i] - c2[i]
        delta = np.ceil(np.abs(diff) * kappa)
            
        if delta > 9:
            delta = 9.0
        elif delta < 1:
            delta = 1.0
            
        if diff < 0:
            delta = 1.0 / delta
        
        s.append(delta)
    
    return s

def AHP(matrix):    
    '''
    Calcula vetor de prioridades
    '''
    e_vals, e_vecs = np.linalg.eig(matrix)
    lamb = np.real(max(e_vals))
    w = np.real(e_vecs[:, e_vals.argmax()])
    w = w / np.sum(w)
    
    #Obtém a razão de consistência
    n = matrix.shape[0]
    ri = {1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49, 11: 1.51}
    ci = (lamb - n) / (n - 1)
    cr = ci / ri[n]
    
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
    V = V / np.sum(V)
    return V

def pairwiseMatrix(measures, M):
    '''
    Calcula as matrizes de comparações paritárias
    '''
    m_Acc = np.identity(M)
    m_Pp  = np.identity(M)
    m_Pnp = np.identity(M)
    m_Fp  = np.identity(M)
    m_Fnp = np.identity(M)
    m_Se  = np.identity(M)
    m_Sp  = np.identity(M)
    m_AUC = np.identity(M)

    i = 0
    J = 1
    for k in range(0, measures.shape[0] - 1):
        
        j = J
        for q in range(k + 1, measures.shape[0]):
            s = scale(measures[k], measures[q])
            
            m_Acc[i, j] = s[0]
            m_Pp[i, j]  = s[1]
            m_Pnp[i, j] = s[2]
            m_Fp[i, j]  = s[3]
            m_Fnp[i, j] = s[4]
            m_Se[i, j]  = s[5]
            m_Sp[i, j]  = s[6]
            m_AUC[i, j] = s[7]
            
            m_Acc[j, i] = 1.0/s[0]
            m_Pp[j, i]  = 1.0/s[1]
            m_Pnp[j, i] = 1.0/s[2]
            m_Fp[j, i]  = 1.0/s[3]
            m_Fnp[j, i] = 1.0/s[4]
            m_Se[j, i]  = 1.0/s[5]
            m_Sp[j, i]  = 1.0/s[6]
            m_AUC[j, i] = 1.0/s[7]
            
            j += 1
            
        J += 1
        i += 1
    
    return m_Acc, m_Pp, m_Pnp, m_Fp, m_Fnp, m_Se, m_Sp, m_AUC

def pairwiseMatrix1(measures, M):
    '''
    Calcula as matrizes de comparações paritárias, sem as medidas F1
    '''
    m_Acc = np.identity(M)
    m_Pp  = np.identity(M)
    m_Pnp = np.identity(M)
    m_Se  = np.identity(M)
    m_Sp  = np.identity(M)
    m_AUC = np.identity(M)

    i = 0
    J = 1
    for k in range(0, measures.shape[0] - 1):
        
        j = J
        for q in range(k + 1, measures.shape[0]):
            s = scale(measures[k], measures[q])
            
            m_Acc[i, j] = s[0]
            m_Pp[i, j]  = s[1]
            m_Pnp[i, j] = s[2]
            m_Se[i, j]  = s[3]
            m_Sp[i, j]  = s[4]
            m_AUC[i, j] = s[5]
            
            m_Acc[j, i] = 1.0/s[0]
            m_Pp[j, i]  = 1.0/s[1]
            m_Pnp[j, i] = 1.0/s[2]
            m_Se[j, i]  = 1.0/s[3]
            m_Sp[j, i]  = 1.0/s[4]
            m_AUC[j, i] = 1.0/s[5]
            
            j += 1
            
        J += 1
        i += 1
    
    return m_Acc, m_Pp, m_Pnp, m_Se, m_Sp, m_AUC
    
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
            print (Fraction.from_float(l)).limit_denominator(10), char,
            i += 1
        print '\\\\'
        
def printVector(w):
    for r in range(0, len(w)):
        print np.round(w[r], 4), '\\\\'

def confidenceInterval(measure, level, n):
    error = np.sqrt(measure * (1 - measure) / n)
    if level == 90:
        z = 1.65
    elif level == 95:
        z = 1.96
    elif level == 98:
        z = 2.33
    elif level == 99:
        z = 2.58
        
    return np.round([measure - z * error, measure + z * error], 3)

'''
Variáveis Globais
'''
measures_labels = ['Acc', 'Pp', 'Pnp', 'Fp', 'Fnp', 'Se', 'Sp', 'AUC']
labels          = ['RF', 'KNN', 'MNB', 'SVM', 'MLP', 'RBF', 'VP', 'AIS']
w  = [1.0, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0] #vetor de pesos para as medidas, utilizado na AHP
w  = w / np.sum(w)

'''
Resultados Gerais, por medida
'''
Acc = [0.972, 0.968, 0.970, 0.976, 0.967, 0.971, 0.970, 0.984]
Pp  = [0.827, 0.804, 0.819, 0.978, 0.778, 0.912, 0.953, 0.857]
Pnp = [0.985, 0.983, 0.984, 0.976, 0.985, 0.976, 0.971, 0.992]
Fp  = [0.830, 0.806, 0.823, 0.830, 0.806, 0.807, 0.783, 0.883]
Fnp = [0.985, 0.983, 0.984, 0.987, 0.982, 0.985, 0.984, 0.989]
Se  = [0.833, 0.805, 0.821, 0.721, 0.835, 0.724, 0.665, 0.911]
Sp  = [0.985, 0.983, 0.984, 0.999, 0.985, 0.994, 0.997, 0.987]
AUC = [0.963, 0.928, 0.945, 0.860, 0.955, 0.977, 0.831, 0.949]

#Resultados Gerais, por classificador
measures_arr = np.array([Acc, Pp, Pnp, Fp, Fnp, Se, Sp, AUC]).T

#Mostra a média e desvio padrão
print '################ Resultados Gerais #######################'
print '\n>> Média e Desvio padrão <<\n'
i = 0
for m in measures_arr.T:
    print '{:>4}'.format(measures_labels[i]), ' {:.3f} +/- {:.3f}'.format(np.mean(m), np.std(m))
    i += 1
'''    
print '\n>> Intervalos de Confiança <<\n'
print 'Acc'
print '{:>3}'.format(90), confidenceInterval(m, 90)
'''

RF  = measures_arr[0,]
KNN = measures_arr[1,]
MNB = measures_arr[2,]
SVM = measures_arr[3,]
MLP = measures_arr[4,]
RBF = measures_arr[5,]
VP  = measures_arr[6,]
AIS = measures_arr[7,]
measures = np.array([RF, KNN, MNB, SVM, MLP, RBF, VP, AIS])
M = measures.shape[0]

#Matrizes de comparações paritárias
m_Acc, m_Pp, m_Pnp, m_Fp, m_Fnp, m_Se, m_Sp, m_AUC = pairwiseMatrix(measures, M) 

#Obtém os vetores de prioridades locais
w1, cr = AHP(m_Acc)
w2, cr = AHP(m_Pp)
w3, cr = AHP(m_Pnp)
w4, cr = AHP(m_Fp)
w5, cr = AHP(m_Fnp)
w6, cr = AHP(m_Se)
w7, cr = AHP(m_Sp)
w8, cr = AHP(m_AUC)

#obtém o vetor de prioridade global
ws = np.array([w1, w2, w3, w4, w5, w6, w7, w8])
v  = globalVector(w, ws)

print 'Vetor de Prioridades Global - Resultados Gerais'
print v

'''
Resultados selecionando algumas características (features). Para tres classificadores apenas
'''

def AHPFeaturesSelecionadas(measures_arr):
    measures_arr = measures_arr.T
    AIS = measures_arr[0,]
    SVM = measures_arr[1,]
    RBF = measures_arr[2,]
    measures = np.array([AIS, SVM, RBF])
    M = measures.shape[0]
    m_Acc, m_Pp, m_Pnp, m_Fp, m_Fnp, m_Se, m_Sp, m_AUC = pairwiseMatrix(measures, M)
    w1, cr = AHP(m_Acc)
    w2, cr = AHP(m_Pp)
    w3, cr = AHP(m_Pnp)
    w4, cr = AHP(m_Fp)
    w5, cr = AHP(m_Fnp)
    w6, cr = AHP(m_Se)
    w7, cr = AHP(m_Sp)
    w8, cr = AHP(m_AUC)
    ws = np.array([w1, w2, w3, w4, w5, w6, w7, w8])
    v  = globalVector(w, ws)
    return v
    
print '\n################ Resultados Para algumas Features #######################\n'
print 'Vetor de Prioridades Global\n'

sel_features = []

Acc = [0.658, 0.949, 0.975]
Pp  = [0.114, 0.775, 0.914]
Pnp = [0.936, 0.960, 0.981]
Fp  = [0.184, 0.633, 0.840]
Fnp = [0.784, 0.973, 0.987]
Se  = [0.474, 0.535, 0.777]
Sp  = [0.675, 0.986, 0.994]
AUC = [0.574, 0.761, 0.978]
a = np.array([Acc, Pp, Pnp, Fp, Fnp, Se, Sp, AUC])
sel_features.append(a.T)
v = AHPFeaturesSelecionadas(a)
print 'Features a6, a3: ', v

Acc = [0.932, 0.970, 0.975]
Pp  = [0.560, 0.964, 0.883]
Pnp = [0.979, 0.971, 0.983]
Fp  = [0.649, 0.784, 0.840]
Fnp = [0.963, 0.984, 0.987]
Se  = [0.772, 0.660, 0.801]
Sp  = [0.946, 0.998, 0.991]
AUC = [0.859, 0.829, 0.981]
a = np.array([Acc, Pp, Pnp, Fp, Fnp, Se, Sp, AUC])
sel_features.append(a.T)
v = AHPFeaturesSelecionadas(a)
print 'Features a6, a3, a12: ', v

Acc = [0.900, 0.700, 0.973]
Pp  = [0.436, 0.116, 0.866]
Pnp = [0.977, 0.933, 0.982]
Fp  = [0.552, 0.181, 0.831]
Fnp = [0.944, 0.817, 0.986]
Se  = [0.751, 0.408, 0.798]
Sp  = [0.914, 0.726, 0.989]
AUC = [0.833, 0.567, 0.981]
a = np.array([Acc, Pp, Pnp, Fp, Fnp, Se, Sp, AUC])
sel_features.append(a.T)
v = AHPFeaturesSelecionadas(a)
print 'Features a6, a3, a12, a11: ', v

Acc = [0.944, 0.868, 0.966]
Pp  = [0.613, 0.326, 0.858]
Pnp = [0.987, 0.960, 0.975]
Fp  = [0.716, 0.418, 0.777]
Fnp = [0.969, 0.926, 0.982]
Se  = [0.859, 0.580, 0.710]
Sp  = [0.952, 0.994, 0.990]
AUC = [0.806, 0.737, 0.980]
a = np.array([Acc, Pp, Pnp, Fp, Fnp, Se, Sp, AUC])
sel_features.append(a.T)
v = AHPFeaturesSelecionadas(a)
print 'Features a6, a3, a12, a11, a2: ', v

Acc = [0.932, 0.818, 0.976]
Pp  = [0.557, 0.199, 0.969]
Pnp = [0.983, 0.942, 0.977]
Fp  = [0.661, 0.267, 0.837]
Fnp = [0.962, 0.897, 0.987]
Se  = [0.811, 0.406, 0.737]
Sp  = [0.943, 0.855, 0.998]
AUC = [0.877, 0.631, 0.962]
a = np.array([Acc, Pp, Pnp, Fp, Fnp, Se, Sp, AUC])
sel_features.append(a.T)
v = AHPFeaturesSelecionadas(a)
print 'Features a6, a3, a12, a11, a2, a4: ', v

Acc = [0.962, 0.955, 0.978]
Pp  = [0.736, 0.949, 0.927]
Pnp = [0.985, 0.956, 0.982]
Fp  = [0.781, 0.639, 0.857]
Fnp = [0.979, 0.976, 0.988]
Se  = [0.832, 0.482, 0.797]
Sp  = [0.974, 0.998, 0.994]
AUC = [0.903, 0.740, 0.975]
a = np.array([Acc, Pp, Pnp, Fp, Fnp, Se, Sp, AUC])
sel_features.append(a.T)
v = AHPFeaturesSelecionadas(a)
print 'Features a6, a3, a12, a11, a2, a4, a8: ', v

Acc = [0.962, 0.950, 0.971]
Pp  = [0.738, 0.957, 0.890]
Pnp = [0.985, 0.950, 0.978]
Fp  = [0.781, 0.568, 0.811]
Fnp = [0.979, 0.974, 0.985]
Se  = [0.829, 0.404, 0.745]
Sp  = [0.979, 0.998, 0.992]
AUC = [0.902, 0.701, 0.977]
a = np.array([Acc, Pp, Pnp, Fp, Fnp, Se, Sp, AUC])
sel_features.append(a.T)
v = AHPFeaturesSelecionadas(a)
print 'Features a6, a3, a12, a11, a2, a4, a8, a9: ', v

Acc = [0.974, 0.951, 0.959]
Pp  = [0.825, 0.945, 0.871]
Pnp = [0.989, 0.952, 0.964]
Fp  = [0.849, 0.593, 0.701]
Fnp = [0.986, 0.974, 0.978]
Se  = [0.874, 0.432, 0.587]
Sp  = [0.984, 0.998, 0.992]
AUC = [0.929, 0.715, 0.973]
a = np.array([Acc, Pp, Pnp, Fp, Fnp, Se, Sp, AUC])
sel_features.append(a.T)
v = AHPFeaturesSelecionadas(a)
print 'Features a6, a3, a12, a11, a2, a4, a8, a9, a10: ', v

Acc = [0.950, 0.953, 0.975]
Pp  = [0.649, 0.991, 0.918]
Pnp = [0.987, 0.952, 0.979]
Fp  = [0.739, 0.598, 0.832]
Fnp = [0.973, 0.975, 0.987]
Se  = [0.857, 0.428, 0.761]
Sp  = [0.959, 0.999, 0.994]
AUC = [0.908, 0.714, 0.981]
a = np.array([Acc, Pp, Pnp, Fp, Fnp, Se, Sp, AUC])
sel_features.append(a.T)
v = AHPFeaturesSelecionadas(a)
print 'Features a6, a3, a12, a11, a2, a4, a8, a9, a10, a5: ', v

Acc = [0.960, 0.960, 0.972]
Pp  = [0.703, 0.711, 0.950]
Pnp = [0.990, 0.987, 0.974]
Fp  = [0.784, 0.777, 0.808]
Fnp = [0.978, 0.978, 0.985]
Se  = [0.886, 0.857, 0.704]
Sp  = [0.967, 0.969, 0.997]
AUC = [0.926, 0.913, 0.977]
a = np.array([Acc, Pp, Pnp, Fp, Fnp, Se, Sp, AUC])
sel_features.append(a.T)
v = AHPFeaturesSelecionadas(a)
print 'Features a6, a3, a12, a11, a2, a4, a8, a9, a10, a5, a7: ', v

'''
Verifica se existe algum resultado utilizando menos features que seja melhor que utilizar todas
'''
r_ais = 0
r_svm = 0
r_rbf = 0
for F in sel_features:
    if any(x > 0 for x in F[0] - AIS):
        r_ais += 1
    if any(x > 0 for x in F[1] - SVM):
        r_svm += 1
    if any(x > 0 for x in F[2] - RBF):
        r_rbf += 1
        
print '\nDiferença entre todas features e as selecionadas'
print 'AIS: ', r_ais, ' SVM: ', r_svm, ' RBF: ', r_rbf

'''
Resultados - Validação Cruzada
'''
print '\n\nResultados - Validação Cruzada'

Acc = [0.982, 0.990, 0.987]
Pp  = [0.981, 0.995, 0.987]
Pnp = [0.984, 0.985, 0.988]
Fp  = [0.982, 0.990, 0.988]
Fnp = [0.982, 0.990, 0.988]
Se  = [0.984, 0.985, 0.988]
Sp  = [0.981, 0.995, 0.987]
AUC = [0.982, 0.990, 0.998]
a = np.array([Acc, Pp, Pnp, Fp, Fnp, Se, Sp, AUC])
sel_features.append(a.T)
v = AHPFeaturesSelecionadas(a)

print '\nPrioridade Global '
print v
'''
n_acc = 36428 + 3219 #quantidade de batimentos totais
print '\nIntervalos de Confiança'
print '\nAIS classifier'
print 'Acc - 90: ', confidenceInterval(Acc[0], 90, n_acc)
print 'Acc - 95: ', confidenceInterval(Acc[0], 95, n_acc)
print 'Acc - 98: ', confidenceInterval(Acc[0], 98, n_acc)
print 'Acc - 99: ', confidenceInterval(Acc[0], 99, n_acc)
print 'Pp - 90: ', confidenceInterval(Acc[0], 90, n_acc)
print 'Pp - 95: ', confidenceInterval(Acc[0], 95, n_acc)
print 'Pp - 98: ', confidenceInterval(Acc[0], 98, n_acc)
print 'Pp - 99: ', confidenceInterval(Acc[0], 99, n_acc)
'''

'''
Resultados - Desvio picos-R
'''
print '\n\nResultados - Desvio picos-R'
w  = [1.0, 1.0, 0.5, 1.0, 0.5, 1.0]
w  = w / np.sum(w)

#Resultados
Acc = [0.976, 0.967, 0.945]
Pp  = [0.977, 0.853, 0.622]
Pnp = [0.976, 0.976, 0.986]
Se  = [0.721, 0.724, 0.857]
Sp  = [0.999, 0.989, 0.981]
AUC = [0.860, 0.958, 0.901]

measures_arr = np.array([Acc, Pp, Pnp, Se, Sp, AUC]).T

SVM = measures_arr[0,]
RBF = measures_arr[1,]
AIS = measures_arr[2,]
measures = np.array([SVM, RBF, AIS])
M = measures.shape[0]

#Matrizes de comparações paritárias
m_Acc, m_Pp, m_Pnp, m_Se, m_Sp, m_AUC = pairwiseMatrix1(measures, M) 

#Obtém os vetores de prioridades locais
w1, cr = AHP(m_Acc)
w2, cr = AHP(m_Pp)
w3, cr = AHP(m_Pnp)
w4, cr = AHP(m_Se)
w5, cr = AHP(m_Sp)
w6, cr = AHP(m_AUC)

#obtém o vetor de prioridade global
ws = np.array([w1, w2, w3, w4, w5, w6])
v  = globalVector(w, ws)

print '\nPrioridade Global '
print v

'''
Resultados - Desvio picos-R, DS3 dataset
'''
print '\n\nResultados - Desvio picos-R, DS3 dataset'

#Resultados
Acc = [0.942, 0.977, 0.945]
Pp  = [0.907, 0.975, 0.953]
Pnp = [0.985, 0.980, 0.938]
Se  = [0.987, 0.980, 0.937]
Sp  = [0.899, 0.975, 0.954]
AUC = [0.943, 0.978, 0.945]

measures_arr = np.array([Acc, Pp, Pnp, Se, Sp, AUC]).T

SVM = measures_arr[0,]
RBF = measures_arr[1,]
AIS = measures_arr[2,]
measures = np.array([SVM, RBF, AIS])
M = measures.shape[0]

#Matrizes de comparações paritárias
m_Acc, m_Pp, m_Pnp, m_Se, m_Sp, m_AUC = pairwiseMatrix1(measures, M) 

#Obtém os vetores de prioridades locais
w1, cr = AHP(m_Acc)
w2, cr = AHP(m_Pp)
w3, cr = AHP(m_Pnp)
w4, cr = AHP(m_Se)
w5, cr = AHP(m_Sp)
w6, cr = AHP(m_AUC)

#obtém o vetor de prioridade global
ws = np.array([w1, w2, w3, w4, w5, w6])
v  = globalVector(w, ws)

print '\nPrioridade Global '
print v