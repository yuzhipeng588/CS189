#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 16:59:28 2016

@author: mac
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
mu = 0
sigma = 1
w1 = np.random.normal(mu, sigma, 100)
w2 = np.random.normal(mu, sigma, 100)
v1 = np.random.normal(mu, sigma, 100)
v2 = np.random.normal(mu, sigma, 100)
theta1 = np.random.uniform(0,2*np.pi,100)
theta2 = np.random.uniform(0,2*np.pi,100)
x1,x2 = np.append(8*np.cos(theta1)+w1,v1),np.append(8*np.cos(theta2)+w2,v2)
y = np.append(np.ones(len(w1)),-1*np.ones(len(w2)))
x = np.column_stack((x1,x2))
plt.figure(1)
plt.plot(x1[0:99], x2[0:99],'rs', label='y = 1')
plt.plot(x1[100:199], x2[100:199], 'bs',label='y = -1')
plt.title('points of two classes')
plt.ylabel('x2')
plt.xlabel('x1')
plt.show()

def predict(K,a):
    pred=np.dot(K,a)
    pred_label=np.ones(pred.shape[0])
    for i in range(pred.shape[0]):
        if pred[i]<=0.1:
            pred_label[i]=-1
    return pred_label
#polynomial kernel
def poly_kernel(x,lamda=1e-6):
    phi = np.dot(x,np.transpose(x))
    K = phi*phi
    I = np.eye(K.shape[0])
    a = np.dot(np.linalg.inv(K+I),np.reshape(y,(200,1)))
    return a,K
#gaussian kernel
def gaussian_kernel(x,lamda=1e-6,gamma=10):
    x1=x[:,0]
    x2=x[:,1]
    xc1=np.tile(x1,(200,1))
    xr1=np.tile(np.reshape(x1,(200,1)),(1,200))
    xc2=np.tile(x2,(200,1))
    xr2=np.tile(np.reshape(x2,(200,1)),(1,200))
    dist=(xc1-xr1)**2+(xc2-xr2)**2
    K=np.exp(-gamma*dist)
    I = np.eye(K.shape[0])
    a = np.dot(np.linalg.inv(K+I),np.reshape(y,(200,1)))
    return a,K
#a,K=poly_kernel(x)
a,K=gaussian_kernel(x)
pred=predict(K,a)
print("Train accuracy: {0}".format(metrics.accuracy_score(y, pred)))

'''
xlist = np.linspace(-10.0, 10.0, 10)
ylist = np.linspace(-10.0, 10.0, 10)
X, Y = np.meshgrid(xlist, ylist)
xx=np.c_[X.ravel(), Y.ravel()]
plt.figure(2)
aa,KK=gaussian_kernel(xx)
Z=np.transpose(np.dot(KK,aa))
Z=Z.reshape(len(xlist))
cp = plt.contour(X, Y, Z)
plt.clabel(cp, 
          fontsize=10)
plt.title('Contour Plot')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
'''

















