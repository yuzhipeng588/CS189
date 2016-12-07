# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 18:42:00 2016

@author: Zhipeng
"""
""" Problem 1 """
k = 10
import scipy
import sklearn.metrics

import random
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

Data = scipy.io.loadmat('data_hw6_cs189_fa16/mnist_data/images.mat')
X = Data['images']

# Our dataset X consists in 60,000 unlabeled images; Each contains 28x28 pixels
    
# For all x in X, assign x to the cluster of which the center is the closest to X
    
# For this we create a list of centers, containing for each x the coordinates of its cluster
    
# The metric we use to calculate the distance between two points is the Frobenius norm of the matrix of differences
    
def centers_to_clusters(X, Centers): # This function takes the centers as inputs, and outputs the clusters structure (each time, a point is classified into the cluster of the closest center)
    Clusters_Index = np.zeros(X.shape[2]) # This a list of the index of the cluster for each 
    
    for i in range(X.shape[2]):
        index_center = 0
        min_distance = LA.norm(X[:,:,i] - Centers[:,:,0],'fro') # We initialize this point's cluster index at 0
        distance = 0
        for j in range(k):
            distance = LA.norm(X[:,:,i] - Centers[:,:,j],'fro')
            if distance < min_distance:
                index_center = j
                min_distance = distance
        Clusters_Index[i] = index_center
    return Clusters_Index
# We have thus here created above the array giving the matrix of the cluster center for every dataset point
# Below, we write down the step where we change the centers to the averages of the points in the cluster:

def clusters_to_centers_2(X,Clusters_Index): # Takes as inputs the clusters structure and returns the new centers, calculated as the clusters averages
    
    Clusters_Average = np.zeros((X.shape[0],X.shape[1],k)) # This is the new list of centers, calculated by averaging
    Lengths = np.zeros(k)
    Sums_Elements = np.zeros((X.shape[0],X.shape[1],k))

    for j in range(X.shape[2]):
        C = Clusters_Index[j]
        Lengths[C] += 1.0
        Sums_Elements[:,:,C] += X[:,:,j]
    
    for i in range(k):
        Clusters_Average[:,:,i] = Sums_Elements[:,:,i]/Lengths[i]
    
    return Clusters_Average # List of coordinates of the new centers (calculated as the clusters average)
    
def New_Cluster (X,Centers): # This method summarizes one step of the recursion: Given one clusters structure, we first go from clusters to centers, then go from centers to the new clusters
    return clusters_to_centers_2(X,centers_to_clusters(X, Centers)) 
    
if __name__ == "__main__":
    # Step 1: Choosing randomly k points in X
    List_ind = random.sample(range(X.shape[2]), k) # List of indexes of centers
    
    # We now create the list of centers
    Centers = np.zeros((X.shape[0],X.shape[1],k))
    for i in range(k):
        Centers[:,:,i] = X[:,:,List_ind[i]] # We thus here have our list of centers
        
    while np.allclose(New_Cluster(X,Centers),Centers,rtol=10**(-2), atol=10**(-4)) == False:
        #np.array_equal(New_Cluster(X,Centers),Centers) == False:
        Centers = New_Cluster(X,Centers)  
    for i in range(k):
        plt.matshow(Centers[:,:,i]) # To visualize the centers