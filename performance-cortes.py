#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 10:43:20 2019

@author: bruno
"""
#import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.rc('xtick', labelsize=9) 
matplotlib.rc('ytick', labelsize=9)

cuts = [0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.2, 0.3, 0.4, 0.5, 0.6]
Se = [0.9071, 0.9025, 0.8969, 0.8922, 0.8866, 0.8813, 0.8748, 0.8658, 0.8590, 0.8428, 0.7608, 0.6943, 0.6160, 0.5219]
Sp = [0.9615, 0.9690, 0.9743, 0.9788, 0.9826, 0.9854, 0.9875, 0.9891, 0.9908, 0.9928, 0.9971, 0.9979, 0.9984, 0.9987]

plt.figure()
plt.subplot(211)
plt.title('(a)')
plt.hist(total_proba[:,1])
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.grid(alpha=0.2)

plt.subplot(212)
plt.title('(b)')
plt.plot(cuts, Se, c='r', label='$S_e$', marker='o')
plt.plot(cuts, Sp, c='k', label='$S_p$', marker='o')
plt.grid(alpha=0.2)
plt.xlabel('$\\alpha$')
plt.ylabel('Performance')
plt.xticks(cuts, cuts)
plt.xticks(rotation=90)
plt.legend()

plt.tight_layout()
plt.show()