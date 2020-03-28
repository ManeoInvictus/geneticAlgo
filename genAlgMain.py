#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 11:32:00 2020

@author: ajay
"""

import pandas as pd
import numpy as np
from tensorflow import keras
# from tensorflow.keras import layers
from sklearn.preprocessing import OneHotEncoder

#Set random seed
np.random.seed(0)

#load data
df = pd.read_csv('./trainData.csv')

#Split labels and data
labels = np.array(df.pop('label'))
x = df[['x','y']].to_numpy()

# #Split data into corresponding labels
# x_0 = x[labels==0,:]
# x_1 = x[labels==1,:]

onehot_encoder = OneHotEncoder(sparse=False)
labels = onehot_encoder.fit_transform(labels.reshape(len(labels),1))

#load model
model = keras.models.load_model('model.h5')
loaded_weights = model.load_weights('model_weights_untrain.h5')
model.set_weights(loaded_weights)

def mat_to_vector(mat_pop_weights):
     pop_weights_vector = []
     for sol_idx in range(mat_pop_weights.shape[0]):
         curr_vector = []
         for layer_idx in range(mat_pop_weights.shape[1]):
             vector_weights = np.reshape(mat_pop_weights[sol_idx, layer_idx], newshape=(mat_pop_weights[sol_idx, layer_idx].size))
             curr_vector.extend(vector_weights)
         pop_weights_vector.append(curr_vector)
     return np.array(pop_weights_vector)

def vector_to_mat(vector_pop_weights, mat_pop_weights):
    mat_weights = []
    for sol_idx in range(mat_pop_weights.shape[0]):
        start = 0
        end = 0
        for layer_idx in range(mat_pop_weights.shape[1]):
            end = end + mat_pop_weights[sol_idx, layer_idx].size
            curr_vector = vector_pop_weights[sol_idx, start:end]
            mat_layer_weights = np.reshape(curr_vector, newshape=(mat_pop_weights[sol_idx, layer_idx].shape))
            mat_weights.append(mat_layer_weights)
            start = end
    return np.reshape(mat_weights, newshape=mat_pop_weights.shape)

def fitness(weights_mat):
    
    accuracy = np.empty(shape=(weights_mat.shape[0]))
    for sol_idx in range(weights_mat.shape[0]):
        curr_sol_mat = weights_mat[sol_idx, :]
        accuracy[sol_idx], _ = evalModel(curr_sol_mat)

    return accuracy

def evalModel(weights):
    
    model.set_weights(weights)
    results = model.evaluate(x,labels)
    return results[-1], model.predict(x)

#Genetic algorithm parameters:
#    Mating Pool Size (Number of Parents)
#    Population Size
#    Number of Generations
#    Mutation Percent

sol_per_pop = 8
num_parents_mating = 4
nGenerations = 1000
mutation_percent = 10

best_outputs = []
accuracies = np.empty(shape=(nGenerations))

pop_weights_mat = np.array(loaded_weights)
pop_weights_vector = mat_to_vector(pop_weights_mat)

for generation in range(nGenerations):
    print("Generation : ", generation)

    # converting the solutions from being vectors to matrices.
    pop_weights_mat = vector_to_mat(pop_weights_vector, 
                                       pop_weights_mat)

    # Measuring the fitness of each chromosome in the population.
    fitness = fitness(pop_weights_mat)
    accuracies[generation] = fitness[0]
    print("Fitness")
    print(fitness)

    # Selecting the best parents in the population for mating.
    parents = select_mating_pool(pop_weights_vector, 
                                    fitness.copy(), 
                                    num_parents_mating)
    # print("Parents")
    # print(parents)

    # Generating next generation using crossover.
    offspring_crossover = ga.crossover(parents,

                                       offspring_size=(pop_weights_vector.shape[0]-parents.shape[0], pop_weights_vector.shape[1]))

    print("Crossover")
    print(offspring_crossover)

    # Adding some variations to the offsrping using mutation.
    offspring_mutation = ga.mutation(offspring_crossover, 

                                     mutation_percent=mutation_percent)
    print("Mutation")
    print(offspring_mutation)

    # Creating the new population based on the parents and offspring.
    pop_weights_vector[0:parents.shape[0], :] = parents
    pop_weights_vector[parents.shape[0]:, :] = offspring_mutation

pop_weights_mat = ga.vector_to_mat(pop_weights_vector, pop_weights_mat)
best_weights = pop_weights_mat [0, :]
acc, predictions = ANN.predict_outputs(best_weights, data_inputs, data_outputs, activation="sigmoid")
print("Accuracy of the best solution is : ", acc)

matplotlib.pyplot.plot(accuracies, linewidth=5, color="black")
matplotlib.pyplot.xlabel("Iteration", fontsize=20)
matplotlib.pyplot.ylabel("Fitness", fontsize=20)
matplotlib.pyplot.xticks(numpy.arange(0, num_generations+1, 100), fontsize=15)
matplotlib.pyplot.yticks(numpy.arange(0, 101, 5), fontsize=15)

f = open("weights_"+str(num_generations)+"_iterations_"+str(mutation_percent)+"%_mutation.pkl", "wb")
pickle.dump(pop_weights_mat, f)
f.close()

