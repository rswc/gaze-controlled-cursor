#pylint: disable=invalid-name
# Pre-determined starting population
# Fill to capacity by randomizing/mutation
# Fitness function
# -- Load calibration data
# Process
# -- kill of set number of worst models
# -- crossover and mutation, keep constant population size
# -- repeat
# -- terminate after set number of generations
# Also look into
# -- expanding populations
# -- using more than two parents for crossover
# -- keeping track of best overall model

import random
import numpy as np
from sortedcontainers import SortedList

class Model: #TODO: slots
    def __init__(self, norm=False, out_ac=None, epochs=0, layers=[], error=None):
        self.norm = norm
        self.out_ac = out_ac
        self.epochs = epochs
        self.layers = layers
        self.error = error

    def __lt__(self, other):
        return self.error < other.error

    def __le__(self, other):
        return self.error <= other.error

    def __gt__(self, other):
        return self.error > other.error

    def __ge__(self, other):
        return self.error >= other.error

class Util:
    @staticmethod
    def choice_crossover(a, b):
        if a is b:
            return a
        else:
            return random.choice((a, b))

    @staticmethod
    def normal_crossover(a, b):
        x = np.random.normal(scale=0.5)
        if x > 1:
            x = 1
        if x < -1:
            x = -1
        
        if x is 0:
            return int((a + b) / 2)
        elif x > 0:
            return int((x * a + (1-x) * b) / 2)
        else:
            return int(((x-1) * a + x * b) / 2)
        

def simple_crossover(a, b):
    model = Model()

    model.norm = Util.choice_crossover(a.norm, b.norm)
    model.out_ac = Util.choice_crossover(a.out_ac, b.out_ac)
    model.epochs = Util.normal_crossover(a.epochs, b.epochs)

    

    return model

def simple_mutation(a, b):
    pass

class GeneticCalculator:
    def __init__(self, population, fitness_func, crossover_func=simple_crossover, selection_amount=1,
                 mutation_func=simple_mutation):
        self.__fitness = fitness_func
        self.__crossover = crossover_func
        self.__mutate = mutation_func
        self.__generation = 0
        self.__selection_amount = selection_amount

        population = [(el, fitness_func(el)) for el in population]

        self.__population = SortedList(population)

    def start(self, max_generations):
        for _ in range(max_generations):
            # Selection
            del self.__population[-self.__selection_amount:]

            births = []

            # Crossover
            for i in range(self.__selection_amount, step=2):
                births.append(self.__crossover(self.__population[i], self.__population[i+1]))

            # Mutation
            for b in births:
                b = self.__mutate(b)

            # Fitness
