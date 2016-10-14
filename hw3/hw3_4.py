# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 21:25:55 2016

@author: Nero
"""
import scipy.io as io
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
test = io.loadmat('data/spam.mat')
Xtrain=test['Xtrain']
Xtest=test['Xtest']
ytrain=test['ytrain']

#
def phi(X,wr,b):
    X=np.sqrt(2.0/wr.shape[0])*np.cos(np.dot(X,np.transpose(wr))+np.transpose(b))
    return X

#standardize pre-processing 
def standardize(Xdata):
    for i in range(Xdata.shape[1]):
        Xdata[:,i]=(Xdata[:,i]-np.mean(Xdata[:,i]))/np.var(Xdata[:,i])**0.5
    return Xdata

#logistic pre-processing 
def transformlog(Xdata):
    for i in range(Xdata.shape[1]):
        Xdata[:,i]=np.log(Xdata[:,i]+0.1)
    return Xdata

#binarize pre-processing   
def Binarize(Xdata):
    Xdata=transformlog(Xdata)
    for (i,j) in range(Xdata.shape[0],Xdata.shape[1]):
        if Xdata[i][j]>0:
            Xdata[i][j]=1
        else:
            Xdata[i][j]=0
    return Xdata

#raw predictions to real predictions    
def toY(Ypred):
    for i in range(Ypred.shape[0]):
        if Ypred[i]>0.5:
            Ypred[i]=1
        else:
            Ypred[i]=0
    return Ypred.tolist()

#sigmoid e~
def sigmoid(Xtrain,w):
    beta=np.dot(Xtrain,w)
    return 1/(1+np.e**(-1*beta))

#logistic regression with l2 regularization gradient decent method   
def log_reg_l2_gd(Xtrain,ytrain):
    Iter=10000
    lumda=1
    #step=0.001 for Standardize
    #step=0.00005 for transformlog
    step=0.00001
    Loss=[]
    w=np.zeros([Xtrain.shape[1],1])
    for i in range(Iter):
        s=np.zeros([Xtrain.shape[0],Xtrain.shape[1]])
        s=sigmoid(Xtrain,w)
        gradient=2*lumda*w-np.dot(np.transpose(Xtrain),ytrain-s)
        w=w-gradient*step
        loss=-1*np.dot(np.transpose(ytrain),np.log(s))-np.dot((1-np.transpose(ytrain)),np.log(1-s))
        Loss=np.append(Loss,loss)
    plt.plot(np.arange(Iter),Loss)
    return w
    
#logistic regression with l2 regularization stochastic gradient decent method  
def log_reg_l2_sgd(Xtrain,ytrain):
    Iter=20000
    lumda=1
    #step=0.001 for Standardize
    #step=0.00005 for transformlog
    #step=0.0001
    Loss=[]
    w=np.zeros([Xtrain.shape[1],1])
    for i in range(Iter): 
        step=0.002/(i+1)**0.3
        k=np.random.randint(0,3450)
        s=np.zeros([Xtrain.shape[0],Xtrain.shape[1]])
        s=sigmoid(Xtrain[k,:],w)
        m=np.zeros([Xtrain.shape[1],1])
        m[:,0]=Xtrain[k,:].T*(ytrain[k,:]-s)
        gradient=2*lumda*w-m
        w=w-gradient*step
        s=sigmoid(Xtrain,w)
        loss=-1*np.dot(np.transpose(ytrain),np.log(s))-np.dot((1-np.transpose(ytrain)),np.log(1-s))
        Loss=np.append(Loss,loss)
    plt.plot(np.arange(Iter),Loss)
    return w

#
'''
Id=np.eye(Xtrain.shape[1])
wr=np.random.multivariate_normal(np.zeros(Xtrain.shape[1]),0.1**2*Id,1000)
b=2*np.pi*np.random.random(1000)
Xtrain=phi(Xtrain,wr,b)
'''
   
#Xtrain=standardize(Xtrain)
Xtrain=transformlog(Xtrain)
#Xtrain=Binarize(Xtrain)
#gradient decent method
Ypred=sigmoid(Xtrain,log_reg_l2_gd(Xtrain,ytrain))

#stochastic method
#Ypred=sigmoid(Xtrain,log_reg_l2_sgd(Xtrain,ytrain))
Ypred=toY(Ypred)
print("Train accuracy: {0}".format(metrics.accuracy_score(ytrain, Ypred)))

Xtest=transformlog(Xtest)
#Xtest=phi(Xtest,wr,b)
#Xtest=standardize(Xtest)
Ypred=sigmoid(Xtest,log_reg_l2_gd(Xtrain,ytrain))
Ypred=toY(Ypred)