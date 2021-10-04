# -*- coding: utf-8 -*-
"""
Gera amostras de cada ciclo cardíaco para as arrmitias selecionadas
"""

import sys, os
sys.path.append(os.getcwd())

import wfdb
import numpy as np
import config
from config import RECORDS_DS1, RECORDS_DS2

def __getCardiacCycle(RECORD, DATABASE, ANN_TYPE):
    '''
    Cria uma lista onde cada elemento tem as amostras dos complexos QRS com a arritmia considerada
    Com ruído atenuado utilizando __denoising()

    Parameters
    ----------
    record: nome do arquivo do registro
    '''

    #localização do registro ECG
    dir_ECG = config.DATABASES[DATABASE]
    if DATABASE == config.DB_MITBIH:
        if RECORD >= 800:#caso esteja utilizando os dados da base SVDB juntamente com MIT/BIH
            rec = config.DATABASES[config.DB_SVDB] + str(RECORD) 
        else:
            rec = dir_ECG + str(RECORD) + '/' + str(RECORD)
    elif DATABASE == config.DB_SVDB:
        rec = dir_ECG + str(RECORD)
    elif DATABASE == config.DB_INCART:
        s = 'I'
        if len(str(RECORD)) == 1:
            s += '0'
        rec = dir_ECG + s + str(RECORD)

    #lê o sinal de ECG, na primeira derivação
    data, info = wfdb.rdsamp(rec, channels=[0])
    ecg = list(data[:,0])

    #para armazenar os QRS do tipo ANN_TYPE
    all_Cycle = []

    #lê as anotações do ECG
    ann = wfdb.rdann(rec, 'atr')
    
    #temanho do ciclo cardíaco
    len_cycle = config.LEN_CARDIAC_CYCLE(DATABASE)

    for l in range(0, len(ann.sample)):

        #obtêm a localização em amostra do pico R
        R_peak = ann.sample[l]

        #obtêm o tipo de anotação para armazenar no arquivo alvo
        ann_type = ann.symbol[l]

        #segmento do ciclo cardíaco
        cycle_seg = ecg[ int(R_peak - len_cycle/2) : int(R_peak + len_cycle/2) ]

        if sum(cycle_seg) != 0 and len(cycle_seg) == len_cycle and ann_type == ANN_TYPE:
            all_Cycle.append(cycle_seg)

    return all_Cycle


def __createSamples(RECORDS, LABEL_RECORDS, DATABASE, FILE_SAVE, TEXT):
    '''
    Cria uma lista onde cada elemento tem as amostras dos complexos QRS com a arritmia considerada
    Gera as amostras selecionando aleatoriamente a quantidade de complexos considerada nos parâmetros

    Parameters
    ----------
    RECORDS: lista com os registros de ECG
    LABEL_RECORDS: nome do dataset
    DATABASE: database escolhido
    FILE_SAVE: arquivo onde salvar os dados
    TEXT: texto a ser exibido
    '''
    print('Fase: ' + TEXT)
    print('Database: ' + DATABASE)

    for record in RECORDS:
        print('\tLendo registro ' + str(record) + '...')
        
        #para armazenar os batimentos normalmente
        beats_N  = []
        beats_P  = []
        target_N = []
        target_P = []
    
        '''
        Classe Negativa
        '''
        all_cardiac_cycle_N = __getCardiacCycle(record, DATABASE, config.LABEL_N[0])
        count_record_N = 0 

        for i, item in enumerate(all_cardiac_cycle_N):

            beats_N.append(all_cardiac_cycle_N[i])
            target_N.append(config.NEGATIVE_CLASS)
            count_record_N += 1

        '''
        Classe Positiva
        '''
        all_cardiac_cycle_P = __getCardiacCycle(record, DATABASE, config.LABEL_P[0])
        count_record_P = 0

        for i, item in enumerate(all_cardiac_cycle_P):

            beats_P.append(all_cardiac_cycle_P[i])
            target_P.append(config.POSITIVE_CLASS)
            count_record_P += 1

        print('\t\t\t Normal: ',  count_record_N, '\t PVC: ', count_record_P)
        
        '''
        Concatena os dados
        '''
        beats = beats_N  + beats_P
        target = target_N + target_P
                
        #banco de dados desbalanceado
        np.savetxt(config.DIR_FILES_RECORDS + FILE_SAVE + '_' + LABEL_RECORDS + '_' + str(config.LEN_CARDIAC_CYCLE(DATABASE)) + '_' + str(record) + '.txt', beats, delimiter=' ', fmt='%1.4f')
        np.savetxt(config.DIR_FILES_RECORDS + 'target_' + FILE_SAVE + '_' + LABEL_RECORDS + '_' + str(config.LEN_CARDIAC_CYCLE(DATABASE)) + '_' + str(record) + '.txt', target, delimiter=' ', fmt='%d')
                        
    return beats_N, beats_P


def run(DS_TRAINING, DS_TEST):
    '''
    TREINAMENTO
    '''
    beats_N, beats_P = __createSamples(eval(DS_TRAINING), DS_TRAINING, config.DB_TRAINING, config.FILE_SAMPLES_TRAINING, 'TREINAMENTO')

    '''
    TESTE
    '''
    beats_N, beats_P = __createSamples(eval(DS_TEST), DS_TEST, config.DB_TEST, config.FILE_SAMPLES_TEST, 'TESTE')


if __name__ == '__main__':
    print('Aguarde...')
    run(config.DS_TRAINING, config.DS_TEST)
    print('Arquivos de amostras geradas!')