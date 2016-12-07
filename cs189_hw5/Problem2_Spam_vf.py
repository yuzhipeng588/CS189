# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 20:07:32 2016

@author: Zhipeng
"""

""" Problem 2.1 : Spam Data """
import math
max_height = 10
import numpy as np
import scipy
import sklearn.metrics as metrics

Data = scipy.io.loadmat('hw5_data/spam_data/spam_data.mat')

Xtrain =  Data['training_data']
Ytrain = Data['training_labels'].T

Xvalidation = Xtrain[5000:,:]
Yvalidation = Ytrain[5000:]

Xtrain = Xtrain[:5000,:]
Ytrain = Ytrain[:5000]

Xtest = Data['test_data']

# We first implement a function that takes as input a dataset, a feature and a threshold value and outputs two datasets:

def splitset(rows,feature,threshold,labels):

    labels1 = np.array([])
    labels2 = np.array([])
    set1 = rows[rows[:,feature] > threshold]
    set2 = rows[rows[:,feature] <= threshold]

    for i in range(rows.shape[0]):
        if rows[i,feature] > threshold:
            labels1 = np.append(labels1,labels[i],axis=0)
        else:
            labels2 = np.append(labels2,labels[i],axis=0)

    return (set1,set2,labels1,labels2)

# We implement the DecisionTree class below

class DecisionTree:
    
    def __init__(self, data = None):
        self.data=data
    
    def entropy(self, data, labels):
        
        labels_1 = labels[labels == 1]
        labels_0 = labels[labels == 0]
        
        p1 = len(labels_1)/ (labels_1.shape[0]+ labels_0.shape[0])
        p0 = len(labels_0)/ (labels_1.shape[0]+ labels_0.shape[0])
        
        if p1 == 0:
            return 1
        elif p0 == 0:
            return 1
        else:
            return - p1 * math.log(p1) - p0 * math.log(p0)
        
    def impurity(self,s1,labels1,s2,labels2):
        # This function outputs the "badness" of the specified split on the input data
        # We here choose, as a measure of the "badness", the weighted average entropy of the two child sets 
        p1 = len(s1)/(len(s1)+len(s2))
        p2 = len(s2)/(len(s1)+len(s2))
        
        if p1 == 0:
            return p2 * DecisionTree.entropy(self,s2,labels2)
        elif p2 == 0:
            return p1 * DecisionTree.entropy(self,s1,labels1)
        else:
            return p1 * DecisionTree.entropy(self,s1,labels1) + p2 * DecisionTree.entropy(self,s2,labels2)
            
    def segmenter(self,data,labels):
        # finds the best split rule for a Node using the impurity measure and input data
        # outputs the feature and threshold value
        # also outputs the two sets resulting of this treshold and value
        
        current_impurity = 0.0
        lowest_impurity = 1000000000000000000000000000000 # We need to start with a "dummy" (very high) entropy value
        best_criteria = None
        best_sets = None
        feature_count = data.shape[1]
        
        # Our outer loop goes through the different features:
        
        for feature in range(feature_count):
            # Generate the list of all possible different values in the considered column
            global column_values
            column_values=[]
            
            for i in range(data.shape[0]):
                column_values.append(data[i,feature])
                
            for j in range(len(column_values)):
                (set1,set2,labels1,labels2) = splitset(data,feature,column_values[j],labels)
                current_impurity = DecisionTree.impurity(self,set1,labels1,set2,labels2)
                
                if current_impurity < lowest_impurity and set1.shape[0]>0 and set2.shape[0]>0: 
                    lowest_impurity = current_impurity
                    best_criteria = (feature,column_values[j])
                    best_sets = (set1,set2)
        return best_criteria, best_sets
    
    def train(self,data,labels,height):
        # Grows a decision tree by constructing nodes
        # BWe implement it recursively, and start with the two special case
        
        # Special case 1: Dataset is empty --> We return an empty node
        if data.shape[0]==1:
            return Node(label = labels[0])
            
        if data.shape[0] == 0:
            return Node(label = 1)
        
        if height == max_height:
            N1 = 0
            N2 = 0
            
            for i in range(data.shape[0]):
                if labels[i] == 0:
                    N1 += 1
                else:
                    N2 += 1
            if N1 > N2:
                return Node(label = 0) # We return a leaf tree with the label equal to 0 in this case
            else:
                return Node(label = 1) #In this other case, we return a leaf tree with label equal to 1
            
        height += 1
        best_criteria, best_sets = self.segmenter(data,labels)

        leftBranch = Tree.train(best_sets[1],labels,height) # Applies the function to the "left branch" (i.e. when we are below the threshold)
        rightBranch = Tree.train(best_sets[0],labels,height) # Applies the function to the "right branch" (i.e. when we are above the threshold)
        
        return Node(col=best_criteria[0],value=best_criteria[1],left=leftBranch,right=rightBranch) # We return in this case an intermediairy node, with a decision rule and left and right nodes (left is below the threshold, right is above) 
                
    def classify(self,data,Node):
        if Node.label != None:
            return Node.label
        elif data[Node.col] > Node.value:
                return Tree.classify(data,Node.right)
        else:
                return Tree.classify(data,Node.left)
        
    def predict(self,data,Node):
        results_array = np.array([])
        for i in range(data.shape[0]):
            results_array= np.append(results_array,Tree.classify(data[i,:],Node))
        return results_array
        
# We implement the class Node below
    
class Node:
    
    def __init__(self,col=-1,value=None,left=None,right=None,label=None):
        self.col=col
        self.value=value
        self.label=label
        self.right=right
        self.left=left

Tree = DecisionTree(Xtrain)
mytree=Tree.train(Xtrain,Ytrain,1)
Results_Train = Tree.predict(Xtrain,mytree)
Results_Validation = Tree.predict(Xvalidation,mytree)
print("Train accuracy: {0}".format(metrics.accuracy_score(Ytrain, Results_Train)))
print("Validation accuracy: {0}".format(metrics.accuracy_score(Yvalidation, Results_Validation)))
