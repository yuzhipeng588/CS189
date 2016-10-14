# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 19:05:16 2016

@author: Nero
"""

import csv
Ypred=np.transpose(np.int_(Ypred))
f = open("kaggle", 'w',newline='')
writer = csv.writer(f)
writer.writerow( ('Id', 'Category' ))   
i=1
for item in Ypred[0]:
    writer.writerow((i,item))
    i=i+1
f.close()