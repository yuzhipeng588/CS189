#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 02:54:55 2016

@author: mac
"""

import csv
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from collections import Counter
import numpy as np

def little_corpus(data,max_features=40):
    corpus=[]
    rev_corpus=[]
    for i in range(data.shape[1]):
        all_items=rawdata.ix[:,i].tolist()
        unique_items_ordered = [x[0] for x in Counter(all_items).most_common()]
        item_ids = {}
        rev_item_ids = {}
        for i, x in enumerate(unique_items_ordered[:min([max_features,len(unique_items_ordered)])]):
            item_ids[x] = i + 1  # so we can pad with 0s
            rev_item_ids[i + 1] = x
        corpus.append(item_ids)
        rev_corpus.append(rev_item_ids)
    return corpus,rev_corpus

def data_onehot(rawdata,corpus):
    X = []
    for i in range(rawdata.shape[0]):
        temp_list=[]
        for j,x in enumerate(rawdata.ix[i,:].tolist()):
            if not x in corpus[j]:
                temp_list.append(0)
            else:
                temp_list.append(corpus[j][x])
        X.append(temp_list)
    return np.array(X)
#logistic pre-processing 
def transformlog(Xdata):
    for i in range(Xdata.shape[1]):
        Xdata[:,i]=np.log(Xdata[:,i]+0.1)
    return Xdata    

def load_data():
    train=pd.read_csv('hw5_data/census_data/train_data.csv',delimiter=',')
    test=pd.read_csv('hw5_data/census_data/test_data.csv',delimiter=',')
    return train,test

def vectorized(rawdata,rawdata_t):
    rawdata=rawdata.drop('label', 1)
    x_dic=rawdata.to_dict(orient='index')
    x_list=[x_dic[i]for i in range(len(x_dic))]
    vec = DictVectorizer()
    x_train=vec.fit_transform(x_list).toarray()
    xtest_dic=rawdata_t.to_dict(orient='index')
    xtest_list=[xtest_dic[i]for i in range(len(xtest_dic))]
    # direct transform using fitted dictvectorizer()
    x_test=vec.transform(xtest_list).toarray()
    return x_train,x_test


import scipy.io
mat = scipy.io.loadmat('hw5_data/spam_data/spam_data.mat')
x_train=mat['training_data']
x_test=mat['test_data']
y_train=mat['training_labels']
y_train=np.reshape(y_train,(y_train.shape[1],))
#x_train=transformlog(x_train)
#x_test=transformlog(x_test)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_features=6,max_depth=30)
#clf = DecisionTreeClassifier()
clf.fit(x_train,y_train)
clf.score(x_train,y_train)

y_pred=clf.predict(x_test)