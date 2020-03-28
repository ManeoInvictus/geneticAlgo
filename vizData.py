#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 18:03:47 2020

@author: ajay
"""

'''
Load data from csv into pandas, viz using seaborn
'''

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

#Load csv data
df = pd.read_csv('./trainData.csv')

#Split labels and data
labels = np.array(df.pop('y'))
x = np.array(df.pop('x'))

#Split data into corresponding labels
x_1 = x[labels==1]
x_2 = x[labels==2]

#Plot distributions of data
plt.figure()
sns.kdeplot(x_1)
sns.kdeplot(x_2)

plt.figure()
sns.kdeplot(x_1,x_2)