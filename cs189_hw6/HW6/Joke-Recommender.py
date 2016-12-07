# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 16:23:07 2016

@author: Zhipeng
"""
''' Problem 4 '''

import scipy
import numpy as np
from numpy import *
from numpy import genfromtxt

# Question 1: We learn a lower dimensional vector representation for users and jokes, using the training data

Data = scipy.io.loadmat('data_hw6_cs189_fa16/joke_data/joke_train.mat')

R = Data ['train']

R[isnan(R)] =0 # We replace all missing values by zero

# We are going to proceed to the singular value decomposition of matrix R

U, s, V = scipy.sparse.linalg.svds(R, k=5, which='LM') # First five singular vectors and singular values of matrix R

# Question 2

# We here create a function to calculate the mean squared error as a function of d, and the Estimated Matrix

def MSE(d):
    U, s, V = scipy.sparse.linalg.svds(R, k=d, which='LM')
    USV = np.dot(U,np.dot(np.diag(s),V))
    Sum = 0
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if R[i,j] != 0:
                Sum += (USV[i,j] - R[i,j])**2 # Squared distance between the matrix and our estimate
    return Sum, USV
    
print("MSE_d=2: {0}".format(MSE(2)[0]))
print("MSE_d=5: {0}".format(MSE(5)[0]))
print("MSE_d=10: {0}".format(MSE(10)[0]))
print("MSE_d=20: {0}".format(MSE(20)[0]))

# We approximate the matrix R by the matrix USV

Validation_set = genfromtxt('data_hw6_cs189_fa16/joke_data/validation.txt', delimiter=',') 

# We now compare our approximate matrix to the values present in the validation set

def accuracy(d):
    
    Sum, Estimate = MSE(d)

    Num_Correct_Predictions = 0

    Num_Incorrect_Predictions = 0
    
    Total = 0 # We create ths variable to count only the times where users gave a rating (we will not take the zeros into account)

    for i in range(Validation_set.shape[0]):
        if Validation_set[i,2] * Estimate[Validation_set[i,0]-1,Validation_set[i,1]-1] > 0:
            Num_Correct_Predictions += 1
            Total +=1
        if Validation_set[i,2] * Estimate[Validation_set[i,0]-1,Validation_set[i,1]-1] < 0:
            Num_Incorrect_Predictions += 1
            Total +=1
            
    return Num_Correct_Predictions/float(Total), Num_Incorrect_Predictions/float(Total)
    
print("accuracy_d=2: {0}".format(accuracy(2)[0]))
print("accuracy_d=5: {0}".format(accuracy(5)[0]))
print("accuracy_d=10: {0}".format(accuracy(10)[0]))
print("accuracy_d=20: {0}".format(accuracy(20)[0]))

'''' Part 3: Alternative Minimization Scheme '''


R = Data ['train']

d = 10

Lambda = 1000

# Step 1: We randomy initialize the ui and vj

# The ui are column vector of size (24983,1). We initialize a random matrix U instead
def update_U(U,V,i,n,m): # update the vector Ui when V is fixed
# First step is to construct the intermediairy vector and matrix we need, before using Ridge Regression's formula:
    R_vec = np.zeros((m,1))
    V_local = np.zeros(shape(V))
    for j in range(m):
        if isnan(R[i,j]) == False:
            R_vec[j] = R[i,j]
            V_local[j,:] = V[j,:]
    return np.linalg.solve(Lambda*np.eye(d)+np.dot(V_local.T,V_local),np.dot(V_local.T,R_vec))
    
def update_V(U,V,j,n,m):
    R_vec = np.zeros((n,1))
    U_local = V_local = np.zeros(shape(U))
    for i in range(n):
        if isnan(R[i,j]) == False:
            R_vec[i] = R[i,j]
            U_local[i,:] = U[i,:]
    return np.linalg.solve(Lambda*np.eye(d)+np.dot(U_local.T,U_local),np.dot(U_local.T,R_vec))   

def update_UV(U,V,n,m):
    for i in range(n):
        U[i,:] = update_U(U,V,i,n,m).reshape(d)
    for j in range(m):
        V[j,:] = update_V(U,V,j,n,m).reshape(d)
    return U,V
    
U = np.random.rand(R.shape[0],d)
# The vi are row vectors, each of size (1, 100). Similarly, we initialize a random matrix U instead
V = np.random.rand(R.shape[1],d)
n = R.shape[0]
m = R.shape[1]

while np.allclose(update_UV(U,V,n,m)[0],U,rtol=1, atol=1) == False or np.allclose(update_UV(U,V,n,m)[1],V,rtol=1, atol=1) == False:
    U,V = update_UV(U,V,n,m)

#Accuracy on the validation set 

def accuracy_2():
    
    Estimate = np.dot(U,V.T)

    Num_Correct_Predictions = 0

    Num_Incorrect_Predictions = 0
    
    Total = 0 # We create ths variable to count only the times where users gave a rating (we will not take the zeros into account)

    for i in range(Validation_set.shape[0]):
        if Validation_set[i,2] * Estimate[Validation_set[i,0]-1,Validation_set[i,1]-1] > 0:
            Num_Correct_Predictions += 1
            Total +=1
        if Validation_set[i,2] * Estimate[Validation_set[i,0]-1,Validation_set[i,1]-1] < 0:
            Num_Incorrect_Predictions += 1
            Total +=1
    return Num_Correct_Predictions/float(Total), Num_Incorrect_Predictions/float(Total)
    
print("accuracy_Step3: {0}".format(accuracy_2()))

# Part 4 - Recommanding jokes

# For this question, we will use the parameters that gave us the best accuracy, namel: Lambda = 1000, d = 10 (these parameters are already set with these values above)

Estimate = np.dot(U,V.T) # Matrix Estimate

Kaggle_set = genfromtxt('data_hw6_cs189_fa16/joke_data/query.txt', delimiter=',') 

# We here create an array containing our results

Kaggle_Array = np.zeros((Kaggle_set.shape[0],2))

for i in range(Kaggle_Array.shape[0]):
    Kaggle_Array[i,0] = Kaggle_set[i,0]
    if Estimate[Kaggle_set[i,1]-1,Kaggle_set[i,2]-1] >0 :  
        Kaggle_Array[i,1] = 1
 
with open("kaggle submission.csv","w") as f:
    writer = csv.writer(f)
    writer.writerows(Kaggle_Array)