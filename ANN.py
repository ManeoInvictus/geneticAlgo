#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 11:44:50 2020

@author: ajay
"""

import numpy as np

def evalModel(weights,model,x,labels):
    
    model.set_weights(weights)
    results = model.evaluate(x,labels,verbose=0)
    return results[-1], model.predict(x)

def fitness(weights_mat,model,x,labels):
    
    accuracy = np.empty(shape=(weights_mat.shape[0]))
    for sol_idx in range(weights_mat.shape[0]):
        curr_sol_mat = weights_mat[sol_idx, :]
        model.set_weights(curr_sol_mat)
        results = model.evaluate(x,labels,verbose=0)
        accuracy[sol_idx] = results[-1]*100

    return accuracy