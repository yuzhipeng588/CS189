# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 18:00:35 2016

@author: Nero
"""
from mnist import MNIST
import sklearn.metrics as metrics
import numpy as np
import scipy
import matplotlib.pyplot as plt
x1=np.array(sorted(30*np.random.random((100,))-15))
x2=np.array(sorted(30*np.random.random((100,))-15))
N1=scipy.stats.norm.pdf(x1,4,2)
N2=0.5*N1+scipy.stats.norm.pdf(x2,3,3)
Mean=np.zeros((2,1))
Mean[0]=np.mean(N1)
Mean[1]=np.mean(N2)
Mean_x1=np.mean(x1)
Mean_x2=np.mean(x2)
Covar=np.cov(N1,N2)
Eigenvalue,Eigenvector=np.linalg.eig(Covar)
plt.figure()
plt.plot(x1,N1,x2,N2)
ax1 = plt.axes()
ax1.arrow(Mean_x1, scipy.stats.norm.pdf(Mean_x1,4,2), Mean_x1+Eigenvector[0,0], scipy.stats.norm.pdf(Mean_x1,4,2)+Eigenvalue[0]*Eigenvector[1,0], head_width=0.05, head_length=0.05, fc='r', ec='r')
ax2 = plt.axes()
ax2.arrow(Mean_x2, scipy.stats.norm.pdf(Mean_x1,4,2)+scipy.stats.norm.pdf(Mean_x2,3,3), Mean_x2+0.5*Eigenvector[0,1], 0.5*scipy.stats.norm.pdf(Mean_x1,4,2)+scipy.stats.norm.pdf(Mean_x2,3,3)+Eigenvalue[1]*Eigenvector[1,1], head_width=0.05, head_length=0.1, fc='b', ec='b')
plt.show()
'''Eigenvector[:,[0,1]]=Eigenvector[:,[1,0]]'''
UT=np.transpose(Eigenvector)
x=np.mat([x1-4,x2-3])
x=[x1-4,x2-3]
xnew=np.dot(UT,x)
plt.figure()

plt.plot(xnew[0,:],N1,xnew[1,:],N2)
plt.xlim([-15,15])
