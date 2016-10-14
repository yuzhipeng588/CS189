# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 12:00:55 2016

@author: Nero
"""

from mnist import MNIST
import sklearn.metrics as metrics
import numpy as np
import scipy
import matplotlib.pyplot as plt

'''3a'''
fig1 = plt.figure()
x1,y1= np.mgrid[0:2:.01, 0:2:.01]
pos1= np.dstack((x1, y1))
rv1=scipy.stats.multivariate_normal(u1, m1)
ax1 = fig1.add_subplot(111)
ax1.contourf(x1, y1, rv1.pdf(pos1))

'''3b'''
fig2 = plt.figure()
x2,y2= np.mgrid[-3:3:.01, -3:3:.01]
pos2= np.dstack((x2, y2))
rv2 =scipy.stats.multivariate_normal(u2, m2)
ax2 = fig2.add_subplot(111)
ax2.contourf(x2,y2,rv2.pdf(pos2))

'''3c'''
u3=[0,2]
u4=[2,0]
m3=m4=np.mat([[2,1],[1,1]])
fig3 = plt.figure()
x3,y3= np.mgrid[-3:3:.01, -3:3:.01]
pos3= np.dstack((x3, y3))
rv3 =scipy.stats.multivariate_normal(u3, m3)
rv4=scipy.stats.multivariate_normal(u4, m4)
ax3 = fig3.add_subplot(111)
ax3.contourf(x3,y3,rv3.pdf(pos3)-rv4.pdf(pos3))

'''3d'''
u3=[0,2]
u4=[2,0]
m3=np.mat([[2,1],[1,1]])
m4=np.mat([[2,1],[1,3]])
fig4 = plt.figure()
x3,y3= np.mgrid[-3:3:.01, -3:3:.01]
pos3= np.dstack((x3, y3))
rv3 =scipy.stats.multivariate_normal(u3, m3)
rv4=scipy.stats.multivariate_normal(u4, m4)
ax3 = fig4.add_subplot(111)
ax3.contourf(x3,y3,rv3.pdf(pos3)-rv4.pdf(pos3))

'''3e'''
u3=[1,1]
u4=[-1,-1]
m3=np.mat([[2,0],[0,1]])
m4=np.mat([[2,1],[1,2]])
fig5 = plt.figure()
x3,y3= np.mgrid[-3:3:.01, -3:3:.01]
pos3= np.dstack((x3, y3))
rv3 =scipy.stats.multivariate_normal(u3, m3)
rv4=scipy.stats.multivariate_normal(u4, m4)
ax3 = fig5.add_subplot(111)
ax3.contourf(x3,y3,rv3.pdf(pos3)-rv4.pdf(pos3))



