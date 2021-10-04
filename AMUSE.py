#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
Main Author: Bruno Rodrigues de Oliveira - bruno@cerradosites.com
Citation: 
  @article{BRO2015,
  title={Blind source separation by multiresolution analysis using AMUSE algorithm},
  author={de Oliveira, Bruno Rodrigues and Duarte, Marco Aparecido Queiroz and Vieira Filho, Jozu{\'e}},
  journal={Multi-Science Journal},
  volume={1},
  number={3},
  pages={40--45},
  year={2015}
}
'''

import numpy as np

class calc:

    def __init__(self, x, n_sources, tau = 1):
        self.x = x
        self.n_sources = n_sources
        self.tau = tau
        
        R, N = self.x.shape
        Rxx = np.cov(self.x)
        U, S, U = np.linalg.svd(Rxx)
        
        if R > self.n_sources:
            noise_var = np.sum(self.x[self.n_sources+1:R+1])/(R - (self.n_sources + 1) + 1)
        else:
            noise_var = 0
        
        h = U[:,0:self.n_sources]
        T = np.zeros((R, self.n_sources))
        
        #1e-10 é adicionado apenas para evitar erros de divisão por zero
        for m in range(0, self.n_sources):
            T[:, m] = np.dot((S[m] - noise_var + 1e-10)**(-0.5) ,  h[:,m])
        
        T = T.T
        y = np.dot(T, self.x)
        R1, N1 = y.shape
        Ryy = np.dot(y ,  np.hstack((np.zeros((R1, self.tau)), y[:,0:N1 - self.tau])).T) / N1
        Ryy = (Ryy + Ryy.T)/2.0
        D, B = np.linalg.eig(Ryy)
        self.B = B.T
        self.y = y
        self.A = np.dot(self.B, T)
        self.sources = np.dot(self.B, y)

'''
Função utilizada para atrasar os sinais, a fim de compor uma matriz que poderá
ser analisada utilizando o AMUSE
'''
def delay(signal, n_delay = 1):
    #n_delay: número de versoes defasadas
    N = len(signal)
    components = []
    
    i = 0
    j = n_delay
    for n in range(0, n_delay + 1):
        components.append(signal[i:N-j])
        i += 1
        j -= 1
        
    return np.array(components)
    
'''
Example
'''
if __name__ == '__main__':
    
    from scipy import signal
    import pylab as pl
    
    t = np.linspace(0, 1, 1500)
    
    #sources
    s1 = signal.sawtooth(2 * np.pi * 5 * t)
    s2 = np.sin(2 * np.pi * 2 * t)
    
    #mixing Matrix
    A = np.array([[0.38614813, 0.88866387], [0.28162975, 0.14468931]])
    
    #observed signal
    X = np.dot(A, np.c_[s1, s2].T)
    
    #estimated sources
    amuse = calc(X, 2, 1)
    s_hat = amuse.sources
    
    #unmixing matrix
    W = amuse.A
    
    print(W)
    
    pl.subplot(321)
    pl.plot(s1)
    pl.title('Source 1')
    pl.subplot(322)
    pl.plot(s2)
    pl.title('Source 2')
    pl.subplot(323)
    pl.plot(X[0])
    pl.title('Mixture 1')
    pl.subplot(324)
    pl.plot(X[1])
    pl.title('Mixture 2')
    pl.subplot(325)
    pl.plot(s_hat[0])
    pl.title('Estimated Source 1')
    pl.subplot(326)
    pl.plot(s_hat[1])
    pl.title('Estimated Source 2')
    
    pl.show()
    

	
