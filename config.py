# -*- coding: utf-8 -*-

'''
Databases
'''
DB_INCART = 'INCART'
DB_MITBIH = 'MIT/BIH'
DB_SVDB   = 'SVDB'

#localização
DATABASES = {}
DATABASES[DB_MITBIH] = '/media/bruno/DADOS/BRUNO/PESQUISAS/ECG/MIT/mitdb/'
DATABASES[DB_SVDB]   = '/media/bruno/DADOS/BRUNO/PESQUISAS/ECG/MIT/svdb/'
DATABASES[DB_INCART] = '/media/bruno/DADOS/BRUNO/PESQUISAS/ECG/INCART/'

#taxas de amostragem
FS = {}
FS[DB_MITBIH] = FS[DB_SVDB] = 360.0
FS[DB_INCART] = 257.0

'''
Registros utilizados
Analisando essa tabela https://www.physionet.org/physiobank/database/html/mitdbdir/tables.htm#allbeats
'''

'''
MIT/BIH
OBS: REGISTROS 201 E 202 SÃO DO MESMO PACIENTE.
'''

#PRINCIPAL DS1, DS2
#For reference: A low-complexity data-adaptive approach for premature ventricular contraction recognition, Effective and efficient detection of premature ventricular contractions based on variation of principal directions, Geometrical features for premature ventricular contraction recognition with analytic hierarchy process based machine learning algorithms selection
RECORDS_DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
RECORDS_DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

#DS3, DS4
#For Reference: Automatic diagnosis of premature ventricular contraction based on Lyapunov exponents and LVQ neural network
#RECORDS_DS1 = [108, 109, 111, 112, 113, 115, 117, 122, 124, 200, 203, 207, 208, 209, 210, 212, 213, 214, 219, 222, 215, 220, 223, 228, 230, 231, 233, 234]
#RECORDS_DS2 = [100, 101, 102, 103, 104, 105, 106, 107, 114, 116, 118, 119, 121, 123, 201, 202, 205, 221, 223, 232]

# DS5, DS6
#For Reference: Detection of premature ventricular contractions using MLP neural networks: A comparative study
#RECORDS_DS1 = [103, 105, 106, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 200, 201, 202, 203, 205, 207, 208, 209, 210, 212, 213, 214, 215, 219, 220, 221, 222, 223, 228, 230, 231, 232, 233, 234]
#RECORDS_DS2 = [100, 101, 102, 104, 105, 106, 107]

# DS7, DS8
#For reference: Robust Neural-Network-Based Classification of Premature Ventricular Contractions Using Wavelet Transform and Timing Interval Features
#RECORDS_DS1 = [100, 102, 104, 105, 106, 107, 118, 119, 200, 201, 203, 205, 208, 212, 213, 214, 215, 217]
#RECORDS_DS2 = [111, 115, 116, 119, 221, 230, 231]

# DS9
#For Reference: Research on Premature Ventricular Contraction Real-time Detection Based Support Vector Machine 
#RECORDS_DS1 = [106, 119, 200, 201, 208, 213, 221, 223, 233]
#RECORDS_DS2 = [106, 119, 200, 201, 208, 213, 221, 223, 233]

#For Reference: Finding features for real-time premature ventricular contraction detection using a fuzzy neural network system
#RECORDS_DS1 = [100, 101, 102, 104, 105, 106, 107, 103, 105, 106, 108, 109, 111, 112, 113, 117, 122, 124, 200, 203, 207, 208, 209, 210, 212, 213, 214, 219, 222, 215, 220, 223, 228, 233, 234, 114, 118, 121, 123, 201, 202, 205, 223, 232]
#RECORDS_DS2 = [115, 116, 119, 221, 230, 231]
 

'''
SVDB
OBS: Registro 801, 822, 880 foi removido devido eo erro no pacote wfdb: IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices
'''
#RECORDS_DS1 = [800, 802, 804, 806, 808, 810, 812, 820, 824, 826, 828, 840, 842, 844, 846, 848, 850, 852, 854, 856, 858, 860, 862, 864, 866, 868, 870, 872, 874, 876, 878, 882, 884, 886, 888, 890, 892, 894]
#RECORDS_DS2 = [803, 805, 807, 809, 811, 821, 823, 825, 827, 829, 841, 843, 845, 847, 849, 851, 853, 855, 857, 859, 861, 863, 865, 867, 869, 871, 873, 875, 877, 879, 881, 883, 885, 887, 889, 891, 893]


'''
INCART
'''
#RECORDS_DS1 = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 51, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74]
#RECORDS_DS2 = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 40, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75]


DS_TRAINING = 'RECORDS_DS1'
DB_TRAINING = DB_MITBIH
#DB_TRAINING = DB_SVDB
#DB_TRAINING = DB_INCART
DS_TEST     = 'RECORDS_DS2'
DB_TEST     = DB_MITBIH
#DB_TEST     = DB_SVDB
#DB_TEST     = DB_INCART

'''
Parâmetros específicos
'''

def LEN_CARDIAC_CYCLE(DATABASE, LEN = 0.75):
    global FS
    len_cycle = int(LEN * FS[DATABASE])
    if len_cycle % 2 != 0:
        len_cycle += 1
    return len_cycle

'''
Quantidade de dígitos para representação dos números
'''
N_DIGITS = 4 #quantidade de casas decimais para qualquer valor utilizado

'''
Sensibilidade do método AHP
'''
KAPPA_AHP = 500

'''
Labels das Arritmia considerada para treinamento
'''
LABEL_N = ['N'] #Classe Negativa
LABEL_P = ['V'] #Classe Positiva

'''
Classes
'''
POSITIVE_CLASS = 1 #PVC
NEGATIVE_CLASS = 0 #NORMAL

'''
Para salvar as máquinas
'''
LEN_FEATURES = 267


'''
Parte do nome dos arquivos
'''
FILE_SAMPLES_TRAINING  = 'training'
FILE_SAMPLES_TEST      = 'teste'
FILE_SAMPLES_BALANCED  = 'balanced'
FILE_FEATURES_TEST     = 'features_test'
FILE_FEATURES_TRAINING = 'features_training'
FILE_TARGET_TRAINING   = 'target_' + FILE_SAMPLES_TRAINING
FILE_TARGET_TEST       = 'target_' + FILE_SAMPLES_TEST
FILE_TARGET_BALANCED   = 'target_' + FILE_SAMPLES_BALANCED
FILE_TARGET_CROSS      = 'crossvalidation'
DIR_FILES              = 'files/'
DIR_FILES_RECORDS      = 'files/records/'
DIR_FILES_MACHINES     = 'machines/'
DIR_FILES_ALL_MACHINES = 'all_machines/'
