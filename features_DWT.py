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
import pywt
    
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

        #center = int(len(cycle)/2) #pico R no segmento do ciclo cardiaco
        #qrs    = np.array(cycle[center - config.LEN_QRS(DATABASE) : center + config.LEN_QRS(DATABASE)])
        coeffs = pywt.wavedec(cycle, pywt.Wavelet('db4'), level=config.LEVEL_DEC)
        f0 = coeffs[0]
        f1 = coeffs[1]
        f2 = coeffs[2]
        f3 = coeffs[3]
        
        features_set0.append(f0)
        features_set1.append(f1)
        features_set2.append(f2)
        features_set3.append(f3)

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
