#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""# Machine learning librairie"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt

"""# Import genetic algorithm librairies"""
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import random


from preprocessing import preprocessing

 # Ramdom Forest With  Genetic algorithm fine-tuning hyper-parameter
 
# This folllowing hyper-paramter will be fine tune using genetic algorithm :

# *   n_estimator
# *   max_depth
# *   max_features
# *   min_samples_leaf


### Genetic algorithm
 
"""load data"""

file = "transactions.csv"


X_train, X_val, X_test, y_train, y_val, y_test = preprocessing(file)

### Defines hyper-parameters boundary so genetic algorithm search in this space
n_estimators_low     = 50
n_estimators_up      = 1000
max_depth_low        = 2
max_depth_up         = 20
max_feature_low      = 2
max_feature_up       = 9
min_samples_leaf_low = 2
min_samples_leaf_up  = 100
low = [n_estimators_low, max_depth_low, max_feature_low, min_samples_leaf_low]
up  = [n_estimators_up, max_depth_up, max_feature_up, min_samples_leaf_up] 

"""Type creator"""
 
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

"""Toolbox"""

toolbox = base.Toolbox()
toolbox.register('estimators_int', random.randint, n_estimators_low, 
                 n_estimators_up)
toolbox.register('max_depth_int', random.randint, max_depth_low, max_depth_up)
toolbox.register('max_feature_int', random.randint, max_feature_low, 
                 max_feature_up)
toolbox.register('min_sample_int', random.randint, 
                 min_samples_leaf_low, min_samples_leaf_up)
toolbox.register("individual", tools.initCycle, creator.Individual, 
                 (toolbox.estimators_int, toolbox.max_depth_int, 
                  toolbox.max_feature_int, toolbox.min_sample_int), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

"""F1 score as fitness function which evaluate individual"""

def evaluation(individual) :
    params = { 'n_estimators'          : individual[0],
               'max_depth'             : individual[1],
               'max_features'          : individual[2],
               'min_samples_leaf'      : individual[3]
            }
    rdf_model = RandomForestClassifier(bootstrap = True, random_state=42, 
                                        n_jobs=-1)
    rdf_model.set_params(**params)
    rdf_model.fit(X_train, y_train)
    v_score = f1_score(y_true = y_val, y_pred = rdf_model.predict(X_val))
    return (v_score,)
 
"""Genetic operators"""
 
toolbox.register("evaluate", evaluation)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutUniformInt, low = low, up = up, indpb=0.2)
toolbox.register("select", tools.selRoulette)

"""creating initial population and statistcs"""

pop = toolbox.population(30)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean, axis=0)
stats.register("std", np.std, axis=0)
stats.register("min", np.min, axis=0)
stats.register("max", np.max, axis=0)

"""Evolving the population"""

NGEN    = 20
MU      = 5
LAMBDA  = 10
CXPB    = 0.7
MUTPB   = 0.2
logbook = tools.Logbook()
logbook.header = ['gen', 'nevals'] + (stats.fields)
pop, logbook, hof =  algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, 
                        MUTPB, NGEN, stats=stats, halloffame=hof, verbose=True)

"""Plot statistic after training"""

print("Best individual is: %s\nwith fitness: %s" % (hof[0], hof[0].fitness))
gen, avg, min_, max_ = logbook.select("gen", "avg", "min", "max")
plt.plot(gen, avg, label="average")
plt.plot(gen, min_, label="minimum")
plt.plot(gen, max_, label="maximum")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.legend(loc="lower right")
plt.show()