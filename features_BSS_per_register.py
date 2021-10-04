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
        
    '''
    Gambiarra: para o caso de data_cycle for composte de um único segmento de ECG
    '''
    if data_cycle.size == len(data_cycle):
        data_cycle_ = []
        data_cycle_.append(data_cycle)
        data_cycle_.append(0)
        data_cycle = data_cycle_[:]
        del data_cycle_
        
    for cycle in data_cycle:
        
        #parte da gambiarra
        if isinstance(cycle, np.ndarray):
        
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
        
        #print('Número de atributos: ', __sub(features[0], N_SUB).shape)
            
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
    
    NoneType = type(None)

    '''
    Lê os dados de treinamento e extrai as características
    '''
    print('Atributos de Treinamento:')
    for record in eval(DS_TRAINING):
        print('Registro: ', record)
        
        data_trai = np.loadtxt(config.DIR_FILES_RECORDS + config.FILE_SAMPLES_TRAINING + '_' + LABEL_RECORDS_TRA + '_' + len_cycle_tra + '_' + str(record) + '.txt')
        
        features_TRAINING0 = features_TRAINING1 = features_TRAINING2 = features_TRAINING3 = None
        
        if N_COMPONENTS == 2 and data_trai.size > 0:
            features_TRAINING0, features_TRAINING1 = __getFeatures(data_trai , N_COMPONENTS, N_SUB, config.DB_TRAINING)
            
        if N_COMPONENTS == 3 and data_trai.size > 0:
            features_TRAINING0, features_TRAINING1, features_TRAINING2 = __getFeatures(data_trai, N_COMPONENTS, N_SUB, config.DB_TRAINING)
            
        if N_COMPONENTS == 4 and data_trai.size > 0:
            features_TRAINING0, features_TRAINING1, features_TRAINING2, features_TRAINING3 = __getFeatures(data_trai, N_COMPONENTS, N_SUB, config.DB_TRAINING)
        
        '''
        Salva as informações das características para utilização em outro script
        '''
        if N_COMPONENTS >= 2:
            if not isinstance(features_TRAINING0, NoneType):
                np.savetxt(config.DIR_FILES + '/' + str(N_COMPONENTS) + 'componentes/' + config.FILE_FEATURES_TRAINING + '_' + LABEL_RECORDS_TRA + '_' + len_cycle_tra + '_' + str(record) + '_0.txt', features_TRAINING0, delimiter=' ', fmt='%1.4f')
            
            if not isinstance(features_TRAINING1, NoneType):
                np.savetxt(config.DIR_FILES + '/' + str(N_COMPONENTS) + 'componentes/' + config.FILE_FEATURES_TRAINING + '_' + LABEL_RECORDS_TRA + '_' + len_cycle_tra + '_' + str(record) + '_1.txt', features_TRAINING1, delimiter=' ', fmt='%1.4f')
                
            del features_TRAINING0
            del features_TRAINING1
                
        if N_COMPONENTS >= 3 and not isinstance(features_TRAINING2, NoneType):
            np.savetxt(config.DIR_FILES + '/' + str(N_COMPONENTS) + 'componentes/' + config.FILE_FEATURES_TRAINING + '_' + LABEL_RECORDS_TRA + '_' + len_cycle_tra + '_' + str(record) + '_2.txt', features_TRAINING2, delimiter=' ', fmt='%1.4f')
            
            del features_TRAINING2
            
        if N_COMPONENTS >= 4  and not isinstance(features_TRAINING3, NoneType):
            np.savetxt(config.DIR_FILES + '/' + str(N_COMPONENTS) + 'componentes/' + config.FILE_FEATURES_TRAINING + '_' + LABEL_RECORDS_TRA + '_' + len_cycle_tra + '_' + str(record) + '_3.txt', features_TRAINING3, delimiter=' ', fmt='%1.4f')
            
            del features_TRAINING3       
    
    print('\nAtributos de Teste')
    for record in eval(DS_TEST):  
        print('Registro: ', record)
        
        '''
        Lê os dados de teste e extrai as características
        '''
        data_teste = np.loadtxt(config.DIR_FILES_RECORDS + config.FILE_SAMPLES_TEST + '_' + LABEL_RECORDS_TES + '_' + len_cycle_tes + '_' + str(record) + '.txt')
        
        features_TEST0 = features_TEST1 = features_TEST2 = features_TEST3 = None
        
        if N_COMPONENTS == 2 and data_teste.size > 0:
            features_TEST0, features_TEST1  = __getFeatures(data_teste, N_COMPONENTS, N_SUB, config.DB_TEST)
            
        if N_COMPONENTS == 3 and data_teste.size > 0:
            features_TEST0, features_TEST1, features_TEST2  = __getFeatures(data_teste, N_COMPONENTS, N_SUB, config.DB_TEST)
            
        if N_COMPONENTS == 4 and data_teste.size > 0:
            features_TEST0, features_TEST1, features_TEST2, features_TEST3  = __getFeatures(data_teste, N_COMPONENTS, N_SUB, config.DB_TEST)
        #target_TEST   = np.loadtxt(config.DIR_FILES + config.FILE_TARGET_TEST + '_' + LABEL_RECORDS_TES + '.txt')
            
        '''
        Salva as informações das características para utilização em outro script
        '''
        if N_COMPONENTS >= 2:
            if not isinstance(features_TEST0, NoneType):
                np.savetxt(config.DIR_FILES + '/' + str(N_COMPONENTS) + 'componentes/' + config.FILE_FEATURES_TEST + '_' + LABEL_RECORDS_TES + '_' + len_cycle_tes + '_' + str(record) + '_0.txt', features_TEST0, delimiter=' ', fmt='%1.4f')
                
            if not isinstance(features_TEST1, NoneType):
                np.savetxt(config.DIR_FILES + '/' + str(N_COMPONENTS) + 'componentes/' + config.FILE_FEATURES_TEST + '_' + LABEL_RECORDS_TES + '_' + len_cycle_tes + '_' + str(record) + '_1.txt', features_TEST1, delimiter=' ', fmt='%1.4f')
                
            del features_TEST0
            del features_TEST1
        
        if N_COMPONENTS >= 3 and not isinstance(features_TEST2, NoneType):
            np.savetxt(config.DIR_FILES + '/' + str(N_COMPONENTS) + 'componentes/' + config.FILE_FEATURES_TEST + '_' + LABEL_RECORDS_TES + '_' + len_cycle_tes + '_' + str(record) + '_2.txt', features_TEST2, delimiter=' ', fmt='%1.4f')
            
            del features_TEST2
            
        if N_COMPONENTS >= 4 and not isinstance(features_TEST3, NoneType):
            np.savetxt(config.DIR_FILES + '/' + str(N_COMPONENTS) + 'componentes/' + config.FILE_FEATURES_TEST + '_' + LABEL_RECORDS_TES + '_' + len_cycle_tes + '_' + str(record) + '_3.txt', features_TEST3, delimiter=' ', fmt='%1.4f')
            
            del features_TEST3

if __name__ == '__main__':
    N_COMPONENTS = int(input('Informe a quantidade de componentes (2, 3, 4): '))
    N_SUB = int(input('Informe a subamostragem: '))
    run(config.DS_TRAINING, config.DS_TEST, N_COMPONENTS, N_SUB, config.DB_TRAINING, config.DB_TEST)
    print('\nCaracteristicas geradas!')
