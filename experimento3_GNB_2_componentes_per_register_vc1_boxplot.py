# -*- coding: utf-8 -*-
"""
Bloxplot dos resultados de validação cruzada utilizando 22 dobras
"""

import numpy as np

################# PLOT #################
import matplotlib 
import matplotlib.pyplot as plt
matplotlib.rc('xtick', labelsize=9) 
matplotlib.rc('ytick', labelsize=9)
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

R = []

R.append([0.9875, 0.9217, 0.9472, 0.9065, 0.9946, 0.9374, 0.9918])	
R.append([0.9872, 0.9206, 0.9457, 0.904, 0.9946, 0.9378, 0.9915]) 		
R.append([0.987, 0.9297, 0.9447, 0.9034, 0.9958, 0.9575, 0.9899]) 		
R.append([0.9758, 0.872, 0.9146, 0.8541, 0.9888, 0.8908, 0.9844]) 		
R.append([0.9759, 0.8739, 0.9148, 0.8544, 0.9891, 0.8943, 0.9843]) 		
R.append([0.9755, 0.869, 0.911, 0.848, 0.989, 0.891, 0.984]) 		
R.append([0.9772, 0.8784, 0.9174, 0.8585, 0.9898, 0.8992, 0.985]) 		
R.append([0.9777, 0.8814, 0.9208, 0.864, 0.9898, 0.8995, 0.9857])		
R.append([0.978, 0.8838, 0.9238, 0.8689, 0.9896, 0.8992, 0.9861]) 		
R.append([0.9779, 0.8636, 0.9181, 0.8579, 0.9886, 0.8694, 0.9874])		
R.append([0.9744, 0.8559, 0.8923, 0.8179, 0.9904, 0.8976, 0.9815]) 		
R.append([0.9786, 0.8782, 0.9286, 0.8759, 0.9885, 0.8804, 0.988]) 		
R.append([0.9779, 0.8775, 0.9235, 0.8678, 0.9889, 0.8873, 0.9867]) 		
R.append([0.9791, 0.8857, 0.9237, 0.8685, 0.9904, 0.9036, 0.9865]) 		
R.append([0.9575, 0.7648, 0.8507, 0.7543, 0.978, 0.7756, 0.9753]) 		
R.append([0.9579, 0.7673, 0.8583, 0.7655, 0.9771, 0.769, 0.9766]) 		
R.append([0.9696, 0.8241, 0.8841, 0.8046, 0.9856, 0.8446, 0.9811]) 		
R.append([0.9632, 0.8264, 0.8572, 0.7675, 0.9884, 0.895, 0.9706]) 		
R.append([0.9671, 0.8295, 0.866, 0.7789, 0.9887, 0.8873, 0.975]) 		
R.append([0.9654, 0.8146, 0.8586, 0.7672, 0.9872, 0.8681, 0.9747])		
R.append([0.9668, 0.8164, 0.8674, 0.7798, 0.9864, 0.8565, 0.9772]) 		
R.append([0.9644, 0.7876, 0.852, 0.7555, 0.9844, 0.8225, 0.9768]) 

R = np.array(R)

fig = plt.figure()
bp = plt.boxplot(R)
for flier in bp['fliers']:
    flier.set(marker='x', color='red', alpha=0.5)
plt.xticks(np.arange(1, 8), ['$A_{cc}$','$F(\\varsigma)^{+}$','$F(\\varsigma)^{-}$','$S_e$','$S_p$','$P^{+}$','$P^{-}$'])
plt.xlabel('Medidas de performance')
plt.ylabel(r'Boxplot para 22 dobras')
plt.grid()
plt.tight_layout()
plt.show()

fig.savefig('experimento1_per_register_vc_boxplot.png', bbox_inches='tight')
