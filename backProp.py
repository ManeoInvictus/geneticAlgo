#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 18:05:32 2020

@author: ajay
"""

'''
1. Load data into pandas
2. Split test train
3. Build model, save it and weights
4. Train the network, save the weights
'''

import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import OneHotEncoder

#Set random seed
np.random.seed(0)

df = pd.read_csv('./trainData.csv')

#Split labels and data
labels = np.array(df.pop('label'))
x = df[['x','y']].to_numpy()

#Split data into corresponding labels
x_0 = x[labels==0,:]
x_1 = x[labels==1,:]

onehot_encoder = OneHotEncoder(sparse=False)
labels = onehot_encoder.fit_transform(labels.reshape(len(labels),1))

train_idxs = np.random.choice(x_0.shape[0],size=int(0.7*x_0.shape[0]),replace=False)
val_idxs = [i for i in range(x_0.shape[0]) if not i in train_idxs]

train_x = np.vstack((x_0[train_idxs,:],x_1[train_idxs,:]))
train_y = np.vstack((labels[train_idxs,:],labels[train_idxs,:]))

val_x = np.vstack((x_0[val_idxs,:],x_1[val_idxs,:]))
val_y = np.vstack((labels[val_idxs,:],labels[val_idxs,:]))

shuffle_idxs = np.random.randint(train_x.shape[0],size=train_x.shape[0])
train_x = train_x[shuffle_idxs,:]
train_y = train_y[shuffle_idxs,:]

model = keras.Sequential()
model.add(layers.Dense(4,activation='sigmoid',use_bias=False,input_shape=(2,)))
model.add(layers.Dense(4,activation='sigmoid',use_bias=False))
model.add(layers.Dense(2,activation='sigmoid',use_bias=False))

model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

model.save('model.h5')
model.save_weights('model_weights_untrain.h5')

model.fit(train_x,train_y,epochs=100,batch_size=None,shuffle=True,validation_data=(val_x,val_y))

model.save_weights('model_weights_train.h5')