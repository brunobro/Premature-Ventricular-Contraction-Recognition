#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resultados obtidos do experimento 6, considerando diferentes SNR
ora para treinamento, ora para teste e ora para ambos
"""
import numpy as np

################# PLOT #################
import matplotlib 
import matplotlib.pyplot as plt
matplotlib.rc('xtick', labelsize=9) 
matplotlib.rc('ytick', labelsize=9)
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

'''
SNR para ambos conjuntos
'''
SNR_trai = []
SNR_test = []
SNR_trai.append(-1.4434420750968793)
SNR_trai.append(0.4913861772393068)
SNR_trai.append(2.993642553173144)
SNR_trai.append(6.524490271569014)
#SNR_trai.append(12.5346)

SNR_test.append(-3.3489369560118236)
SNR_test.append(-0.09583114890311872)
SNR_test.append(1.0925090826356318)
SNR_test.append(4.609237302112096)
#SNR_test.append(10.6324)

'''
Ruído para os conjuntos de treinamento e teste
'''

m_clf0     = []
m_clf1     = []
m_suave    = []
m_rigida_n = []
m_rigida_p = []
m_ahp_03   = []
m_ahp_04   = []
m_ahp_05   = []
m_ahp_06   = []
m_ahp_07   = []
'''
m_clf0.append()
m_clf1.append()
m_suave.append()
m_rigida_n.append()
m_rigida_p.append()
m_ahp_03.append()
m_ahp_04.append()
m_ahp_05.append()
m_ahp_06.append()
m_ahp_07.append()'''

m_clf0.append([ 0.9711 ])
m_clf1.append([ 0.9675 ])
m_suave.append([ 0.9698 ])
m_rigida_n.append([ 0.9714 ])
m_rigida_p.append([ 0.9672 ])
m_ahp_03.append([ 0.9661 ])
m_ahp_04.append([ 0.9685 ])
m_ahp_05.append([ 0.9698 ])
m_ahp_06.append([ 0.9711 ])
m_ahp_07.append([ 0.9711 ])

m_clf0.append([ 0.9743 ])
m_clf1.append([ 0.9707 ])
m_suave.append([ 0.9727 ])
m_rigida_n.append([ 0.9747 ])
m_rigida_p.append([ 0.9703 ])
m_ahp_03.append([ 0.9698 ])
m_ahp_04.append([ 0.9712 ])
m_ahp_05.append([ 0.973 ])
m_ahp_06.append([ 0.9736 ])
m_ahp_07.append([ 0.9742 ])

m_clf0.append([ 0.9762 ])
m_clf1.append([ 0.9747 ])
m_suave.append([ 0.9756 ])
m_rigida_n.append([ 0.9772 ])
m_rigida_p.append([ 0.9737 ])
m_ahp_03.append([ 0.9733 ])
m_ahp_04.append([ 0.9745 ])
m_ahp_05.append([ 0.9757 ])
m_ahp_06.append([ 0.9765 ])
m_ahp_07.append([ 0.9766 ])

m_clf0.append([ 0.9811 ])
m_clf1.append([ 0.9787 ])
m_suave.append([ 0.9800 ])
m_rigida_n.append([ 0.9844 ])
m_rigida_p.append([ 0.9754 ])
m_ahp_03.append([ 0.9753 ])
m_ahp_04.append([ 0.9778 ])
m_ahp_05.append([ 0.9800 ])
m_ahp_06.append([ 0.9828 ])
m_ahp_07.append([ 0.9840 ])
'''
m_clf0.append([ 0.9815 ])
m_clf1.append([ 0.9786 ])
m_suave.append([ 0.9799 ])
m_rigida_n.append([ 0.9866 ])
m_rigida_p.append([ 0.9734 ])
m_ahp_03.append([ 0.9739 ])
m_ahp_04.append([ 0.9767 ])
m_ahp_05.append([ 0.9799 ])
m_ahp_06.append([ 0.9833 ])
m_ahp_07.append([ 0.9863 ])
'''
m_clf0     = np.array(m_clf0)
m_clf1     = np.array(m_clf1)
m_suave    = np.array(m_suave)
m_rigida_n = np.array(m_rigida_n)
m_rigida_p = np.array(m_rigida_p)
m_ahp_03   = np.array(m_ahp_03)
m_ahp_04   = np.array(m_ahp_04)
m_ahp_05   = np.array(m_ahp_05)
m_ahp_06   = np.array(m_ahp_06)
m_ahp_07   = np.array(m_ahp_07)

plt.figure(1)
plt.plot(SNR_test, m_clf0 * 100, marker='o', label=r"$Clf_{0}$", color='black')
plt.plot(SNR_test, m_clf1 * 100, marker='o', label=r'$Clf_{1}$', color='red')
plt.plot(SNR_test, m_suave * 100, marker='o', label='Voto Suave', color='blue')
plt.plot(SNR_test, m_rigida_n * 100, marker='o', label='Voto Rígido - N', color='green')
plt.plot(SNR_test, m_rigida_p * 100, marker='o', label='Voto Rígido - P', color='orange')
plt.plot(SNR_test, m_ahp_03 * 100, marker='o', label=r'Voto AHP $\alpha=0,3$', color='magenta')
plt.plot(SNR_test, m_ahp_04 * 100, marker='o', label=r'Voto AHP $\alpha=0,4$', color='sienna')
plt.plot(SNR_test, m_ahp_05 * 100, marker='o', label=r'Voto AHP $\alpha=0,5$', color='indigo')
plt.plot(SNR_test, m_ahp_06 * 100, marker='o', label=r'Voto AHP $\alpha=0,6$', color='pink')
plt.plot(SNR_test, m_ahp_07 * 100, marker='o', label=r'Voto AHP $\alpha=0,7$', color='brown')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xticks(SNR_test)
plt.xlabel(r'SNR (dB)')
plt.ylabel('Acurácia - $A_{cc}$ (100%)')
plt.grid()
plt.tight_layout()
plt.show()
plt.savefig('experimento4_resultados_SNR_ambos_conjuntos.png', format='png', dpi=300)

'''
Ruído apenas no conjunto de teste
'''

'''
SNR para ambos conjuntos
'''

m_clf0     = []
m_clf1     = []
m_suave    = []
m_rigida_n = []
m_rigida_p = []
m_ahp_03   = []
m_ahp_04   = []
m_ahp_05   = []
m_ahp_06   = []
m_ahp_07   = []

m_clf0.append([ 0.9706 ])
m_clf1.append([ 0.9669 ])
m_suave.append([ 0.9686 ])
m_rigida_n.append([ 0.9718 ])
m_rigida_p.append([ 0.9657 ])
m_ahp_03.append([ 0.9651 ])
m_ahp_04.append([ 0.9669 ])
m_ahp_05.append([ 0.9686 ])
m_ahp_06.append([ 0.9706 ])
m_ahp_07.append([ 0.9711 ])

m_clf0.append([ 0.9729 ])
m_clf1.append([ 0.9706 ])
m_suave.append([ 0.9715 ])
m_rigida_n.append([ 0.9749 ])
m_rigida_p.append([ 0.9687 ])
m_ahp_03.append([ 0.9684 ])
m_ahp_04.append([ 0.9704 ])
m_ahp_05.append([ 0.9715 ])
m_ahp_06.append([ 0.9733 ])
m_ahp_07.append([ 0.9740 ])

m_clf0.append([ 0.975 ])
m_clf1.append([ 0.9747 ])
m_suave.append([ 0.975 ])
m_rigida_n.append([ 0.9774 ])
m_rigida_p.append([ 0.9723 ])
m_ahp_03.append([ 0.9721 ])
m_ahp_04.append([ 0.974 ])
m_ahp_05.append([ 0.975 ])
m_ahp_06.append([ 0.9763 ])
m_ahp_07.append([ 0.9769 ])

m_clf0.append([ 0.975 ])
m_clf1.append([ 0.9747 ])
m_suave.append([ 0.975 ])
m_rigida_n.append([ 0.9774 ])
m_rigida_p.append([ 0.9723 ])
m_ahp_03.append([ 0.9721 ])
m_ahp_04.append([ 0.974 ])
m_ahp_05.append([ 0.975 ])
m_ahp_06.append([ 0.9763 ])
m_ahp_07.append([ 0.9769 ])
'''
m_clf0.append([ 0.7509 ])
m_clf1.append([ 0.706 ])
m_suave.append([ 0.9451 ])
m_rigida_n.append([ 0.7033 ])
m_rigida_p.append([ 0.7537 ])
m_ahp_03.append([ 0.7539 ])
m_ahp_04.append([ 0.7544 ])
m_ahp_05.append([ 0.7541 ])
m_ahp_06.append([ 0.7269 ])
m_ahp_07.append([ 0.7105 ])
'''
m_clf0     = np.array(m_clf0)
m_clf1     = np.array(m_clf1)
m_suave    = np.array(m_suave)
m_rigida_n = np.array(m_rigida_n)
m_rigida_p = np.array(m_rigida_p)
m_ahp_03   = np.array(m_ahp_03)
m_ahp_04   = np.array(m_ahp_04)
m_ahp_05   = np.array(m_ahp_05)
m_ahp_06   = np.array(m_ahp_06)
m_ahp_07   = np.array(m_ahp_07)

plt.figure(2)
plt.plot(SNR_test, m_clf0 * 100, marker='o', label=r"$Clf_{0}$", color='black')
plt.plot(SNR_test, m_clf1 * 100, marker='o', label=r'$Clf_{1}$', color='red')
plt.plot(SNR_test, m_suave * 100, marker='o', label='Voto Suave', color='blue')
plt.plot(SNR_test, m_rigida_n * 100, marker='o', label='Voto Rígido - N', color='green')
plt.plot(SNR_test, m_rigida_p * 100, marker='o', label='Voto Rígido - P', color='orange')
plt.plot(SNR_test, m_ahp_03 * 100, marker='o', label=r'Voto AHP $\alpha=0,3$', color='magenta')
plt.plot(SNR_test, m_ahp_04 * 100, marker='o', label=r'Voto AHP $\alpha=0,4$', color='sienna')
plt.plot(SNR_test, m_ahp_05 * 100, marker='o', label=r'Voto AHP $\alpha=0,5$', color='indigo')
plt.plot(SNR_test, m_ahp_06 * 100, marker='o', label=r'Voto AHP $\alpha=0,6$', color='pink')
plt.plot(SNR_test, m_ahp_07 * 100, marker='o', label=r'Voto AHP $\alpha=0,7$', color='brown')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xticks(SNR_test)
plt.xlabel(r'SNR (dB)')
plt.ylabel('Acurácia - $A_{cc}$ (100%)')
plt.grid()
plt.tight_layout()
plt.show()
plt.savefig('experimento4_resultados_SNR_conjunto_teste.png', format='png', dpi=300)
