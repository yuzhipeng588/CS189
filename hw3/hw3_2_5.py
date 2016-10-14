# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 00:12:33 2016

@author: Nero
"""
import matplotlib.pyplot as plt
import numpy as np
import operator
#convert original prediction to {0,1}
def toY(Ypred):
    for i in range(Ypred.shape[0]):
        if Ypred[i]>0.5:
            Ypred[i]=1
        else:
            Ypred[i]=0
    return Ypred.tolist()
            
X=[4,5,5.6,6.8,7,7.2,8,0.8,1,1.2,2.5,2.6,3,4.3,3]
Y=[1,1,1,1,1,1,1,0,0,0,0,0,0,0,1]
data=dict(zip(X, Y))
data=sorted(data.items(), key=operator.itemgetter(1))
XY=[(ele1, ele2)for ele1,ele2 in data]
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(*zip(*XY),color='r',label="origin")   
data=dict(data) 
X=list(data.keys())
Y=np.mat(list(data.values()))

#standardize X to mean 0 variance 1
Xs=(X-np.mean(X))/np.var(X)**0.5
Xsnew=[Xs,np.zeros(14)+1]
Xsnew=np.mat(Xsnew)

lumda=0.07
beta0=np.mat([[1],[0]])
u=np.zeros([14,1])
beta=beta0

#logistic regression
for k in range(3):
    for i in range(14):
        u[i]=1/(1+np.e**(-1*np.dot(np.transpose(beta),Xsnew[:,i])[0,0]))
    domain=np.zeros([14,14])
    for i in range(14):
        domain[i,i]=(1-u[i])*u[i]
    updateterm1=np.linalg.inv(2*lumda+np.dot(np.dot(Xsnew,domain),np.transpose(Xsnew)))
    updateterm2=(2*lumda*beta-np.dot(Xsnew,np.transpose(Y)-u))
    update=np.dot(updateterm1,updateterm2)
    beta=beta-update
Ypred=np.dot(np.transpose(Xsnew),beta)
Ypred=toY(Ypred)

#make plot beautiful
data=dict(zip(X, Ypred))
data=sorted(data.items(), key=operator.itemgetter(1))
XY=[(ele1, ele2)for ele1,ele2 in data]
plt.plot(*zip(*XY),linestyle=':', color='b',label="logistic")

#linear regresssion
for k in range(3):
    for i in range(14):
        u[i]=1/(1+np.e**(-1*np.dot(np.transpose(beta),Xsnew[:,i])[0,0]))
    domain=np.zeros([14,14])
    for i in range(14):
        domain[i,i]=(1-u[i])*u[i]
    updateterm1=np.linalg.inv(2*lumda+np.dot(Xsnew,np.transpose(Xsnew)))
    updateterm2=(2*lumda*beta+np.dot(np.dot(Xsnew,np.transpose(Xsnew)),beta)-np.dot(Xsnew,np.transpose(Y)))
    update=np.dot(updateterm1,updateterm2)
    beta=beta-update
Ypred=np.dot(np.transpose(Xsnew),beta)
Ypred=toY(Ypred)

#make plot beautiful
data=dict(zip(X, Ypred))
data=sorted(data.items(), key=operator.itemgetter(1))
XY=[(ele1, ele2)for ele1,ele2 in data]
plt.plot(*zip(*XY),linestyle='-.', color='g',label="linear")
plt.ylim([-0.1,1.1])
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)    