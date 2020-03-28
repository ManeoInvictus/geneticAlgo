#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 11:47:21 2020

@author: ajay
"""

import pandas as pd
import numpy as np
from tensorflow import keras
# from tensorflow.keras import layers
from sklearn.preprocessing import OneHotEncoder
import ga
import ANN
# import pickle
import matplotlib.pyplot as plt

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
model.load_weights('model_weights_untrain.h5')
loaded_weights = np.array(model.get_weights())

#Genetic algorithm parameters:
#    Mating Pool Size (Number of Parents)
#    Population Size
#    Number of Generations
#    Mutation Percent

sol_per_pop = 16
num_parents_mating = 8
num_generations = 1000
mutation_percent = 50

#Creating the initial population.
initial_pop_weights = []
kwg = keras.initializers.glorot_uniform()
for curr_sol in np.arange(0, sol_per_pop):
    input_HL1_weights = np.asarray(kwg(loaded_weights[0].shape))

    HL1_HL2_weights = np.asarray(kwg(loaded_weights[1].shape))

    HL2_output_weights = np.asarray(kwg(loaded_weights[2].shape))

    initial_pop_weights.append(np.array([input_HL1_weights, 
                                                HL1_HL2_weights, 
                                                HL2_output_weights]))

pop_weights_mat = np.array(initial_pop_weights)
pop_weights_vector = ga.mat_to_vector(pop_weights_mat)

best_outputs = []
accuracies = np.empty(shape=(num_generations))

for generation in range(num_generations):
    print("Generation : ", generation)

    # converting the solutions from being vectors to matrices.
    pop_weights_mat = ga.vector_to_mat(pop_weights_vector, 
                                       pop_weights_mat)

    # Measuring the fitness of each chromosome in the population.
    fitness = ANN.fitness(pop_weights_mat,model,x,labels)

    accuracies[generation] = fitness[0]
    print("Fitness")
    print(fitness)

    # Selecting the best parents in the population for mating.
    parents = ga.select_mating_pool(pop_weights_vector, 

                                    fitness.copy(), 

                                    num_parents_mating)
    # print("Parents")
    # print(parents)

    # Generating next generation using crossover.
    offspring_crossover = ga.crossover(parents,

                                       offspring_size=(pop_weights_vector.shape[0]-parents.shape[0], pop_weights_vector.shape[1]))

    # print("Crossover")
    # print(offspring_crossover)

    # Adding some variations to the offsrping using mutation.
    offspring_mutation = ga.mutation(offspring_crossover, 

                                     mutation_percent=mutation_percent)
    # print("Mutation")
    # print(offspring_mutation)

    # Creating the new population based on the parents and offspring.
    pop_weights_vector[0:parents.shape[0], :] = parents
    pop_weights_vector[parents.shape[0]:, :] = offspring_mutation

pop_weights_mat = ga.vector_to_mat(pop_weights_vector, pop_weights_mat)
best_weights = pop_weights_mat [0, :]
acc, predictions = ANN.evalModel(best_weights, model, x, labels)
print("Accuracy of the best solution is : ", acc)

plt.plot(accuracies, linewidth=5, color="black")
plt.xlabel("Iteration", fontsize=20)
plt.ylabel("Fitness", fontsize=20)
plt.xticks(np.arange(0, num_generations+1, 100), fontsize=15)
plt.yticks(np.arange(0, 101, 5), fontsize=15)

