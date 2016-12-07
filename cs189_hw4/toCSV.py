#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 02:13:55 2016

@author: mac
"""

import csv
import numpy as np
predL = np.zeros(predY.shape[0])
for index3 in range(predY.shape[0]): 
    predL[index3]=np.argmax(predY[index3,:])
predL=np.int_(predL)
f = open("kaggle", 'w',newline='')
writer = csv.writer(f)
writer.writerow( ('Id', 'Category' ))   
i=1
for item in predL:
    writer.writerow((i,item))
    i=i+1
f.close()