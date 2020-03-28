#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 18:03:07 2020

@author: ajay
"""

'''
Generate data from two Normal Distributions, visualize them and save them as csv
'''

import numpy as np
import pandas as pd
from scipy.stats.kde import gaussian_kde
import matplotlib.pyplot as plt
from sklearn.datasets import make_gaussian_quantiles

N = 500*2 #number of samples

#generating data
X1, y1 = make_gaussian_quantiles(cov=3,n_samples=N,n_features=2,n_classes=2,shuffle=False)
X2, y2 = make_gaussian_quantiles(mean=[5,5],cov=1,n_samples=N,n_features=2,n_classes=2,shuffle=False)
X = np.concatenate((X1,X2))
y = np.concatenate((y1,-y2+1))

df = pd.DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'red', 1:'blue'}
fig, ax = plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
plt.show()

df.to_csv('./trainData.csv',index=False)
