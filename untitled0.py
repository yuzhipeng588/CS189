# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 02:19:47 2016

@author: Nero
"""

import csv
f = open("kaggle", 'w',newline='')
writer = csv.writer(f)
pred_labels_test = pred_labels_test.astype(int)
writer.writerow( ('Id', 'Category' ))   
i=1
for item in pred_labels_test:
    writer.writerow((i,item))
    i=i+1
f.close()