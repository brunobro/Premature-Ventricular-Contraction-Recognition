#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Calcula as medidas de performance
"""
import sklearn.metrics as metrics
import numpy as np

'''
def medical(CM):
    
    Calcula medidas de performance geralmente utilizadas na pesquisa Biomedica tais como
    Sentitividade e especificidade
    CM e a matriz de confusao dada por CM = metrics.confusion_matrix(targets_test, predicted)
    
    TN = float(CM[0][0])
    FN = float(CM[1][0])
    TP = float(CM[1][1])
    FP = float(CM[0][1])
    # Sensitivity (positive class), hit rate, recall, or true positive rate
    TPR = TP/(TP + FN + 1e-10)
    # Specificity or true negative rate
    TNR = TN/(TN + FP + 1e-10)
    # Precision or positive predictive value
    PPV = TP/(TP + FP + 1e-10)
    # Negative predictive value or Sensitivity (negative class)
    NPV = TN/(TN + FN + 1e-10)
    
    return TPR, TNR, PPV, NPV
'''

def every(targets_pred, targets_true, POSITIVE_CLASS, NEGATIVE_CLASS):
    '''
    Calcula todas as medidas
    '''
    acc = metrics.accuracy_score(targets_true, targets_pred)

    f1_P = metrics.f1_score(targets_true, targets_pred, pos_label=POSITIVE_CLASS)
    f1_N = metrics.f1_score(targets_true, targets_pred, pos_label=NEGATIVE_CLASS)

    re_P = metrics.recall_score(targets_true, targets_pred, pos_label=POSITIVE_CLASS)
    re_N = metrics.recall_score(targets_true, targets_pred, pos_label=NEGATIVE_CLASS)

    pr_P = metrics.precision_score(targets_true, targets_pred, pos_label=POSITIVE_CLASS)
    pr_N = metrics.precision_score(targets_true, targets_pred, pos_label=NEGATIVE_CLASS)
    
    CM = metrics.confusion_matrix(targets_true, targets_pred)
    
    if CM.shape == (2, 2):
        TN = float(CM[0][0])
        FN = float(CM[1][0])
        TP = float(CM[1][1])
        FP = float(CM[0][1])
        
    elif CM.shape == (1, 1):
        '''
        Caso somente tenha feito uma única predição
        Nos casos de somente ter uma instância
        '''
        if np.unique(targets_pred)[0] == 1.0:
            TP = CM[0,0]
            TN = 0
            FP = 0
            FN = 0
            
        if np.unique(targets_pred)[0] == 0.0:
            TN = CM[0,0]
            TP = 0
            FP = 0
            FN = 0
        
        
    return [acc, f1_P, f1_N, re_P, re_N, pr_P, pr_N, TP, TN, FP, FN]

def calc(TP, TN, FP, FN):
    '''
    Computada as medidas quando passado a quantidade TP, TN, FP, FN
    Esta função será utiliza para o cálculo manual
    Quando o método AHP for considerado
    '''
    TN = float(TN)
    FN = float(FN)
    TP = float(TP)
    FP = float(FP)    
    
    Acc = (TP + TN)/(TP + TN + FP + FN + 1e-10)
    Pr_P  = TP/(TP + FP + 1e-10)
    Pr_N  = TN/(TN + FN + 1e-10)
    Se  = TP/(TP + FN + 1e-10)
    Sp  = TN/(TN + FP + 1e-10)
    F1_P  = 2 * Pr_P * Se / (Pr_P + Se + 1e-10)
    F1_N  = 2 * Pr_N * Se / (Pr_N + Se + 1e-10)
    Cm  = np.array([[TN, FP, FN, TP]])
    
    return Acc, Pr_P, Pr_N, Se, Sp, F1_P, F1_N, Cm

def AUC (y_true, y_scores, positive_class = 1):
    '''
    Calcula a area sobre a curva ROC
    '''
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores, pos_label=positive_class)
    auc = metrics.auc(fpr, tpr)
    
    return auc