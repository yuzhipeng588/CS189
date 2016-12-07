#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 01:05:50 2016

@author: mac
"""

import csv
import numpy as np
f = open("kaggle_spam_6.30", 'w',newline='')
writer = csv.writer(f)
writer.writerow( ('Id', 'Category' ))   
i=1
for item in y_pred:
    writer.writerow((i,item))
    i=i+1
f.close()