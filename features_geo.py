# -*- coding: utf-8 -*-
"""
Extrai as características utilizando DWT
"""
import sys, os
sys.path.append(os.getcwd())
sys.path.append('/media/bruno/DADOS/BRUNO/PESQUISAS/Python')

import numpy as np
import config
from config import RECORDS_DS1, RECORDS_DS2

def featuresVector(p0, p1, p2=[0,0]):
    v0 = np.array(p0) - np.array(p2)
    v1 = np.array(p1) - np.array(p2)

    prodv = np.dot(v0, v1)
    a0 = np.linalg.norm(v0) #lado 1
    a1 = np.linalg.norm(v1) #lado 2
    a2 = np.linalg.norm(v0 - v1) #lado 3
    p = a0 + a1 + a2
    a3 = np.arccos(np.clip(prodv/(a0 * a1), -1, 1)) #angulo theta
    a4 = a1 * a2 * np.sin(a3)/2 #area do triangulo
    a5 = np.linalg.norm(np.array([(v0[0] + v1[0])/3, (v0[1] + v1[1])/3])) #vetor do baricentro
    a6 = (2 * a4)/(a1 + a2 + a3) #raio do incírculo
    a7 = np.linalg.norm(np.array([(a2 * v0[0] + a3 * v1[0])/p, (a2 * v0[1] + a3 * v1[1])/p])) #incentro
    a8 = 2 * np.pi * a6 #comprimento da cincrunferencia
    a9 = np.pi * a6**2 #area da cincrunferencia
    a10 = p #perimetro do triangulo
    #a10 = (2 * a4) / (np.linalg.norm(v0 - v1) + 1e-10)
    #a11 = (2 * a4) / n0
    #a12 = (2 * a4) / n1

    return [a0, a1, a2, a5, a3, a4, a6, a7, a8, a9, a10]
    #return [a1, a5, a2, a3, a7, a8, a9, a4, a6]
    
'''
Obtem os vetores de características
Antes os dados são ESCALONADOS
'''

def __getFeatures(data_cycle, DATABASE):

    '''
    Sinal em baixa resolução
    '''
    features_set0 = []
    
    '''
    Grupo com todas os atributos
    '''
    features_set1 = []
    
    '''
    Atributos relativos ao triângulo somente: a1, a2, a3, a4, a5, a6, a7, a8, a9, a14, a15
    '''
    features_set2 = []
    
    '''
    Atributos relativos ao incírculo somente: a10, a11, a12, a13
    '''
    features_set3 = []    

    for cycle in data_cycle:

        '''
        Recebe um ciclo cardiaco aproximado e segmenta o QRS para calcular os atributos
        '''

        center = int(len(cycle)/2) #pico R no segmento do ciclo cardiaco
        qrs    = np.array(cycle[center - config.LEN_QRS(DATABASE) : center + config.LEN_QRS(DATABASE)])
        center = int(len(qrs)/2)

        #minimo e máximo
        mi = np.min(qrs[center : len(qrs)]) #mínimo após a onda R, para evitar dizer que o mínimo é o da onda Q
        ma = qrs[center]

        #retorna os indices do maximo e minimo
        i_mi = np.where(qrs == mi)[0][0] + center
        i_ma = center

        #calcula as projecoes
        dist_projx = abs(i_mi - i_ma) #distancia entre as projeções
        dist_projy = abs(mi - ma) #distancia entre as projeções

        #pontos de minimo e maximo e suas localizacoes
        p1 = [i_ma, ma]
        p2 = [i_mi, mi]

        #obtem todos os atributos
        fv = featuresVector(p1, p2)
        
        #atributos conforme texto da tese
        a1 = fv[0]
        a2 = fv[1]
        a3 = fv[2]
        a4 = ma
        a5 = mi
        a6 = fv[3]
        a7 = fv[4]
        a8 = fv[5]
        a9 = fv[10]
        a10 = fv[6]
        a11 = fv[7]
        a12 = fv[8]
        a13 = fv[9]
        a14 = dist_projx
        a15 = dist_projy
        
        features_set0.append(cycle)
        features_set1.append([a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15]) #todas
        features_set2.append([a1, a2, a3, a4, a5, a6, a7, a8, a9, a14, a15])
        features_set3.append([a10, a11, a12, a13]) #a3, a5, a6, a7, a8, a9

    return np.array(features_set0), np.array(features_set1), np.array(features_set2), np.array(features_set3)


def run(DS_TRAINING, DS_TEST):

    LABEL_RECORDS_TRA = DS_TRAINING
    LABEL_RECORDS_TES = DS_TEST

    '''
    Lê os dados de treinamento e extrai as características
    '''
    features_TRAINING0, features_TRAINING1, features_TRAINING2, features_TRAINING3 = __getFeatures(np.loadtxt(config.DIR_FILES + config.FILE_SAMPLES_TRAINING + '_' + LABEL_RECORDS_TRA + '.txt'), config.DB_TRAINING)
    #target_TRAINING   = np.loadtxt(config.DIR_FILES + config.FILE_TARGET_TRAINING + '_' + LABEL_RECORDS_TRA + '.txt')
    
    features_TRAINING0_balanced, features_TRAINING1_balanced, features_TRAINING2_balanced, features_TRAINING3_balanced = __getFeatures(np.loadtxt(config.DIR_FILES + config.FILE_SAMPLES_TRAINING + '_' + LABEL_RECORDS_TRA + '_' + config.FILE_SAMPLES_BALANCED + '.txt'), config.DB_TRAINING)
    #target_TRAINING_balanced = np.loadtxt(config.DIR_FILES + config.FILE_TARGET_TRAINING + '_' + LABEL_RECORDS_TRA + '_' + config.FILE_SAMPLES_BALANCED + '.txt')
        
    '''
    Lê os dados de teste e extrai as características
    '''
    features_TEST0, features_TEST1, features_TEST2, features_TEST3  = __getFeatures(np.loadtxt(config.DIR_FILES + config.FILE_SAMPLES_TEST + '_' + LABEL_RECORDS_TES + '.txt'), config.DB_TEST)
    #target_TEST   = np.loadtxt(config.DIR_FILES + config.FILE_TARGET_TEST + '_' + LABEL_RECORDS_TES + '.txt')
    
    features_TEST0_balanced, features_TEST1_balanced, features_TEST2_balanced, features_TEST3_balanced = __getFeatures(np.loadtxt(config.DIR_FILES + config.FILE_SAMPLES_TEST + '_' + LABEL_RECORDS_TES + '_' + config.FILE_SAMPLES_BALANCED + '.txt'), config.DB_TEST)
    #target_TEST_balanced = np.loadtxt(config.DIR_FILES + config.FILE_TARGET_TEST + '_' + LABEL_RECORDS_TES + '_' + config.FILE_SAMPLES_BALANCED + '.txt')
        
    '''
    Salva as informações das características para utilização em outro script
    '''
    np.savetxt(config.DIR_FILES + config.FILE_FEATURES_TRAINING + '_' + LABEL_RECORDS_TRA + '_0.txt', features_TRAINING0, delimiter=' ', fmt='%1.4f')
    np.savetxt(config.DIR_FILES + config.FILE_FEATURES_TEST + '_' + LABEL_RECORDS_TES + '_0.txt', features_TEST0, delimiter=' ', fmt='%1.4f')
    
    np.savetxt(config.DIR_FILES + config.FILE_FEATURES_TRAINING + '_' + LABEL_RECORDS_TRA + '_1.txt', features_TRAINING1, delimiter=' ', fmt='%1.4f')
    np.savetxt(config.DIR_FILES + config.FILE_FEATURES_TEST + '_' + LABEL_RECORDS_TES + '_1.txt', features_TEST1, delimiter=' ', fmt='%1.4f')
    
    np.savetxt(config.DIR_FILES + config.FILE_FEATURES_TRAINING + '_' + LABEL_RECORDS_TRA + '_2.txt', features_TRAINING2, delimiter=' ', fmt='%1.4f')
    np.savetxt(config.DIR_FILES + config.FILE_FEATURES_TEST + '_' + LABEL_RECORDS_TES + '_2.txt', features_TEST2, delimiter=' ', fmt='%1.4f')
    
    np.savetxt(config.DIR_FILES + config.FILE_FEATURES_TRAINING + '_' + LABEL_RECORDS_TRA + '_3.txt', features_TRAINING3, delimiter=' ', fmt='%1.4f')
    np.savetxt(config.DIR_FILES + config.FILE_FEATURES_TEST + '_' + LABEL_RECORDS_TES + '_3.txt', features_TEST3, delimiter=' ', fmt='%1.4f')

    np.savetxt(config.DIR_FILES + config.FILE_FEATURES_TRAINING + '_' + LABEL_RECORDS_TRA + '_' + config.FILE_SAMPLES_BALANCED + '_0.txt', features_TRAINING0_balanced, delimiter=' ', fmt='%1.4f')
    np.savetxt(config.DIR_FILES + config.FILE_FEATURES_TEST + '_' + LABEL_RECORDS_TES + '_' + config.FILE_SAMPLES_BALANCED + '_0.txt', features_TEST0_balanced, delimiter=' ', fmt='%1.4f')
        
    np.savetxt(config.DIR_FILES + config.FILE_FEATURES_TRAINING + '_' + LABEL_RECORDS_TRA + '_' + config.FILE_SAMPLES_BALANCED + '_1.txt', features_TRAINING1_balanced, delimiter=' ', fmt='%1.4f')
    np.savetxt(config.DIR_FILES + config.FILE_FEATURES_TEST + '_' + LABEL_RECORDS_TES + '_' + config.FILE_SAMPLES_BALANCED + '_1.txt', features_TEST1_balanced, delimiter=' ', fmt='%1.4f')
    
    np.savetxt(config.DIR_FILES + config.FILE_FEATURES_TRAINING + '_' + LABEL_RECORDS_TRA + '_' + config.FILE_SAMPLES_BALANCED + '_2.txt', features_TRAINING2_balanced, delimiter=' ', fmt='%1.4f')
    np.savetxt(config.DIR_FILES + config.FILE_FEATURES_TEST + '_' + LABEL_RECORDS_TES + '_' + config.FILE_SAMPLES_BALANCED + '_2.txt', features_TEST2_balanced, delimiter=' ', fmt='%1.4f')
    
    np.savetxt(config.DIR_FILES + config.FILE_FEATURES_TRAINING + '_' + LABEL_RECORDS_TRA + '_' + config.FILE_SAMPLES_BALANCED + '_3.txt', features_TRAINING3_balanced, delimiter=' ', fmt='%1.4f')
    np.savetxt(config.DIR_FILES + config.FILE_FEATURES_TEST + '_' + LABEL_RECORDS_TES + '_' + config.FILE_SAMPLES_BALANCED + '_3.txt', features_TEST3_balanced, delimiter=' ', fmt='%1.4f')
    
    del features_TRAINING0
    del features_TRAINING1
    del features_TRAINING2
    del features_TRAINING3
    del features_TRAINING0_balanced
    del features_TRAINING1_balanced
    del features_TRAINING2_balanced
    del features_TRAINING3_balanced
    del features_TEST0
    del features_TEST1
    del features_TEST2
    del features_TEST3
    del features_TEST0_balanced
    del features_TEST1_balanced
    del features_TEST2_balanced
    del features_TEST3_balanced


if __name__ == '__main__':
    print('Aguarde...')
    run(config.DS_TRAINING, config.DS_TEST)
    print('Caracteristicas geradas!')
