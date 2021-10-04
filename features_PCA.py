# -*- coding: utf-8 -*-
"""
Extrai as características utilizando DWT
"""
import sys, os
sys.path.append(os.getcwd())

import numpy as np
import config
import AMUSE
from config import RECORDS_DS1, RECORDS_DS2

'''
Subamostra e normaliza
'''
def __sub(s, n):
    S = s[0:len(s):n]
    return S/np.linalg.norm(S, 1)
 
'''
Obtem os vetores de características
Antes os dados são ESCALONADOS
'''

def __getFeatures(data_cycle, N_COMPONENTS, N_SUB, DATABASE):
    
    features_set0 = []
    features_set1 = []
    features_set2 = []
    features_set3 = []
    
    for cycle in data_cycle:

        '''
        Recebe um ciclo cardiaco aproximado e segmenta o QRS para calcular os atributos
        '''
        
        if N_COMPONENTS == 2:
            comp = AMUSE.delay(cycle, 1)
            s    = np.array([comp[0], comp[1]])
        if N_COMPONENTS == 3:
            comp  = AMUSE.delay(cycle, 2)
            s     = np.array([comp[0], comp[1], comp[2]])
        if N_COMPONENTS == 4:
            comp  = AMUSE.delay(cycle, 3)
            s     = np.array([comp[0], comp[1], comp[2], comp[3]])
                
        am = AMUSE.calc(s, s.shape[0], 1)
        features = am.sources
        
        if N_COMPONENTS >= 2:
            features_set0.append(list(__sub(features[0], N_SUB)))
            features_set1.append(list(__sub(features[1], N_SUB)))
        if N_COMPONENTS >= 3:
            features_set2.append(list(__sub(features[2], N_SUB)))
        if N_COMPONENTS >= 4:
            features_set3.append(list(__sub(features[3], N_SUB)))
    
    print('Número de atributos: ', __sub(features[0], N_SUB).shape)
        
    if N_COMPONENTS == 2:    
        return np.array(features_set0), np.array(features_set1)
    if N_COMPONENTS == 3:    
        return np.array(features_set0), np.array(features_set1), np.array(features_set2)
    if N_COMPONENTS == 4:
        return np.array(features_set0), np.array(features_set1), np.array(features_set2), np.array(features_set3)
    

def run(DS_TRAINING, DS_TEST, N_COMPONENTS, N_SUB, DATABASE_TRAINING, DATABASE_TEST):

    LABEL_RECORDS_TRA = DS_TRAINING
    LABEL_RECORDS_TES = DS_TEST

    len_cycle_tra = str(config.LEN_CARDIAC_CYCLE(DATABASE_TRAINING))
    len_cycle_tes = str(config.LEN_CARDIAC_CYCLE(DATABASE_TEST))

    '''
    Lê os dados de treinamento e extrai as características
    '''
    if N_COMPONENTS == 2:
        features_TRAINING0, features_TRAINING1 = __getFeatures(np.loadtxt(config.DIR_FILES + config.FILE_SAMPLES_TRAINING + '_' + LABEL_RECORDS_TRA + '_' + len_cycle_tra + '.txt'), N_COMPONENTS, N_SUB, config.DB_TRAINING)
        features_TRAINING0_BAL, features_TRAINING1_BAL = __getFeatures(np.loadtxt(config.DIR_FILES + config.FILE_SAMPLES_TRAINING + '_' + LABEL_RECORDS_TRA + '_' + config.FILE_SAMPLES_BALANCED + '_' + len_cycle_tra + '.txt'), N_COMPONENTS, N_SUB, config.DB_TRAINING)
        
    if N_COMPONENTS == 3:
        features_TRAINING0, features_TRAINING1, features_TRAINING2 = __getFeatures(np.loadtxt(config.DIR_FILES + config.FILE_SAMPLES_TRAINING + '_' + LABEL_RECORDS_TRA + '_' + len_cycle_tra + '.txt'), N_COMPONENTS, N_SUB, config.DB_TRAINING)
        features_TRAINING0_BAL, features_TRAINING1_BAL, features_TRAINING2_BAL = __getFeatures(np.loadtxt(config.DIR_FILES + config.FILE_SAMPLES_TRAINING + '_' + LABEL_RECORDS_TRA + '_' + config.FILE_SAMPLES_BALANCED + '_' + len_cycle_tra + '.txt'), N_COMPONENTS, N_SUB, config.DB_TRAINING)
        
    if N_COMPONENTS == 4:
        features_TRAINING0, features_TRAINING1, features_TRAINING2, features_TRAINING3 = __getFeatures(np.loadtxt(config.DIR_FILES + config.FILE_SAMPLES_TRAINING + '_' + LABEL_RECORDS_TRA + '_' + len_cycle_tra + '.txt'), N_COMPONENTS, N_SUB, config.DB_TRAINING)
        features_TRAINING0_BAL, features_TRAINING1_BAL, features_TRAINING2_BAL, features_TRAINING3_BAL = __getFeatures(np.loadtxt(config.DIR_FILES + config.FILE_SAMPLES_TRAINING + '_' + LABEL_RECORDS_TRA + '_' + config.FILE_SAMPLES_BALANCED + '_' + len_cycle_tra + '.txt'), N_COMPONENTS, N_SUB, config.DB_TRAINING)
    
    '''
    Salva as informações das características para utilização em outro script
    '''
    if N_COMPONENTS >= 2:
        np.savetxt(config.DIR_FILES + config.FILE_FEATURES_TRAINING + '_' + LABEL_RECORDS_TRA + '_' + len_cycle_tra + '_0.txt', features_TRAINING0, delimiter=' ', fmt='%1.4f')
        np.savetxt(config.DIR_FILES + config.FILE_FEATURES_TRAINING + '_' + LABEL_RECORDS_TRA + '_' + len_cycle_tra + '_1.txt', features_TRAINING1, delimiter=' ', fmt='%1.4f')
        np.savetxt(config.DIR_FILES + config.FILE_FEATURES_TRAINING + '_' + LABEL_RECORDS_TRA + '_' + config.FILE_SAMPLES_BALANCED + '_' + len_cycle_tra + '_0.txt', features_TRAINING0_BAL, delimiter=' ', fmt='%1.4f')
        np.savetxt(config.DIR_FILES + config.FILE_FEATURES_TRAINING + '_' + LABEL_RECORDS_TRA + '_' + config.FILE_SAMPLES_BALANCED + '_' + len_cycle_tra + '_1.txt', features_TRAINING1_BAL, delimiter=' ', fmt='%1.4f')
        del features_TRAINING0, features_TRAINING0_BAL
        del features_TRAINING1, features_TRAINING1_BAL
            
    if N_COMPONENTS >= 3:
        np.savetxt(config.DIR_FILES + config.FILE_FEATURES_TRAINING + '_' + LABEL_RECORDS_TRA + '_' + len_cycle_tra + '_2.txt', features_TRAINING2, delimiter=' ', fmt='%1.4f')
        np.savetxt(config.DIR_FILES + config.FILE_FEATURES_TRAINING + '_' + LABEL_RECORDS_TRA + '_' + config.FILE_SAMPLES_BALANCED + '_' + len_cycle_tra + '_2.txt', features_TRAINING2_BAL, delimiter=' ', fmt='%1.4f')
        del features_TRAINING2, features_TRAINING2_BAL
        
    if N_COMPONENTS >= 4:
        np.savetxt(config.DIR_FILES + config.FILE_FEATURES_TRAINING + '_' + LABEL_RECORDS_TRA + '_' + len_cycle_tra + '_3.txt', features_TRAINING3, delimiter=' ', fmt='%1.4f')
        np.savetxt(config.DIR_FILES + config.FILE_FEATURES_TRAINING + '_' + LABEL_RECORDS_TRA + '_' + config.FILE_SAMPLES_BALANCED + '_' + len_cycle_tra + '_3.txt', features_TRAINING3_BAL, delimiter=' ', fmt='%1.4f')
        del features_TRAINING3, features_TRAINING3_BAL     
        
    '''
    Lê os dados de teste e extrai as características
    '''
    if N_COMPONENTS == 2:
        features_TEST0, features_TEST1  = __getFeatures(np.loadtxt(config.DIR_FILES + config.FILE_SAMPLES_TEST + '_' + LABEL_RECORDS_TES + '_' + len_cycle_tes + '.txt'), N_COMPONENTS, N_SUB, config.DB_TEST)
        features_TEST0_BAL, features_TEST1_BAL  = __getFeatures(np.loadtxt(config.DIR_FILES + config.FILE_SAMPLES_TEST + '_' + LABEL_RECORDS_TES + '_' + config.FILE_SAMPLES_BALANCED + '_' + len_cycle_tes + '.txt'), N_COMPONENTS, N_SUB, config.DB_TEST)
        
    if N_COMPONENTS == 3:
        features_TEST0, features_TEST1, features_TEST2  = __getFeatures(np.loadtxt(config.DIR_FILES + config.FILE_SAMPLES_TEST + '_' + LABEL_RECORDS_TES + '_' + len_cycle_tes + '.txt'), N_COMPONENTS, N_SUB, config.DB_TEST)
        features_TEST0_BAL, features_TEST1_BAL, features_TEST2_BAL  = __getFeatures(np.loadtxt(config.DIR_FILES + config.FILE_SAMPLES_TEST + '_' + LABEL_RECORDS_TES + '_' + config.FILE_SAMPLES_BALANCED + '_' + len_cycle_tes + '.txt'), N_COMPONENTS, N_SUB, config.DB_TEST)
        
    if N_COMPONENTS == 4:
        features_TEST0, features_TEST1, features_TEST2, features_TEST3  = __getFeatures(np.loadtxt(config.DIR_FILES + config.FILE_SAMPLES_TEST + '_' + LABEL_RECORDS_TES + '_' + len_cycle_tes + '.txt'), N_COMPONENTS, N_SUB, config.DB_TEST)
        features_TEST0_BAL, features_TEST1_BAL, features_TEST2_BAL, features_TEST3_BAL = __getFeatures(np.loadtxt(config.DIR_FILES + config.FILE_SAMPLES_TEST + '_' + LABEL_RECORDS_TES + '_' + config.FILE_SAMPLES_BALANCED + '_' + len_cycle_tes + '.txt'), N_COMPONENTS, N_SUB, config.DB_TEST)
        
    '''
    Salva as informações das características para utilização em outro script
    '''
    if N_COMPONENTS >= 2:
        np.savetxt(config.DIR_FILES + config.FILE_FEATURES_TEST + '_' + LABEL_RECORDS_TES + '_' + len_cycle_tes + '_0.txt', features_TEST0, delimiter=' ', fmt='%1.4f')
        np.savetxt(config.DIR_FILES + config.FILE_FEATURES_TEST + '_' + LABEL_RECORDS_TES + '_' + len_cycle_tes + '_1.txt', features_TEST1, delimiter=' ', fmt='%1.4f')
        np.savetxt(config.DIR_FILES + config.FILE_FEATURES_TEST + '_' + LABEL_RECORDS_TES + '_' + config.FILE_SAMPLES_BALANCED + '_' + len_cycle_tes + '_0.txt', features_TEST0_BAL, delimiter=' ', fmt='%1.4f')
        np.savetxt(config.DIR_FILES + config.FILE_FEATURES_TEST + '_' + LABEL_RECORDS_TES + '_' + config.FILE_SAMPLES_BALANCED + '_' + len_cycle_tes + '_1.txt', features_TEST1_BAL, delimiter=' ', fmt='%1.4f')
        del features_TEST0, features_TEST0_BAL
        del features_TEST1, features_TEST1_BAL
    
    if N_COMPONENTS >= 3:
        np.savetxt(config.DIR_FILES + config.FILE_FEATURES_TEST + '_' + LABEL_RECORDS_TES + '_' + len_cycle_tes + '_2.txt', features_TEST2, delimiter=' ', fmt='%1.4f')
        np.savetxt(config.DIR_FILES + config.FILE_FEATURES_TEST + '_' + LABEL_RECORDS_TES + '_' + config.FILE_SAMPLES_BALANCED + '_' + len_cycle_tes + '_2.txt', features_TEST2_BAL, delimiter=' ', fmt='%1.4f')
        del features_TEST2, features_TEST2_BAL
        
    if N_COMPONENTS >= 4:
        np.savetxt(config.DIR_FILES + config.FILE_FEATURES_TEST + '_' + LABEL_RECORDS_TES + '_' + len_cycle_tes + '_3.txt', features_TEST3, delimiter=' ', fmt='%1.4f')
        np.savetxt(config.DIR_FILES + config.FILE_FEATURES_TEST + '_' + LABEL_RECORDS_TES + '_' + config.FILE_SAMPLES_BALANCED + '_' + len_cycle_tes + '_3.txt', features_TEST3_BAL, delimiter=' ', fmt='%1.4f')
        del features_TEST3, features_TEST3_BAL
       

if __name__ == '__main__':
    N_COMPONENTS = int(input('Informe a quantidade de componentes (2, 3, 4): '))
    N_SUB = int(input('Informe a subamostragem: '))
    print('Aguarde...')
    run(config.DS_TRAINING, config.DS_TEST, N_COMPONENTS, N_SUB, config.DB_TRAINING, config.DB_TEST)
    print('Caracteristicas geradas!')
