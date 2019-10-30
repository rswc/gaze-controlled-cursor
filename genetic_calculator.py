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
    def __init__(self, norm=False, out_ac=None, epochs=0, layers=None, error=None):
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

    @staticmethod
    def layer_crossover(long, short, size_diff):
        s_len = len(short)
        x_point = random.randrange(1, s_len)
        diff_x_point = random.randint(0, size_diff)

        if random.randint(0, 1) is 1:
            return long.layers[0:x_point] + short.layers[x_point:] + long[s_len:s_len + diff_x_point]
        else:
            return short.layers[0:x_point] + long.layers[x_point:s_len] + long[s_len:s_len + diff_x_point]

def simple_selection(pop, selection_amount, min_pop_size=10):
    total_deleted = 0

    for i in range(len(pop)):
        if random.random() < 0.5:
            del pop[-i]
            total_deleted += 1
    if total_deleted < selection_amount and len(pop) > min_pop_size:
        del pop[-selection_amount+total_deleted:]

    return total_deleted

def simple_crossover(a, b):
    model = Model()

    model.norm = Util.choice_crossover(a.norm, b.norm)
    model.out_ac = Util.choice_crossover(a.out_ac, b.out_ac)
    model.epochs = Util.normal_crossover(a.epochs, b.epochs)

    l = len(a) - len(b)
    if l >= 0:       # a >= b
        model.layers = Util.layer_crossover(a, b, l)
    elif l < 0:      # a < b
        model.layers = Util.layer_crossover(b, a, -l)

    return model

def simple_mutation(model, pm=0.05):
    if random.random() < pm:
        model.norm = not model.norm
    if random.random() < pm:
        model.out_ac = GeneticCalculator.ACTIVATION_FUNCTIONS[random.randrange(0, len(GeneticCalculator.ACTIVATION_FUNCTIONS))]
    if random.random() < pm:
        model.epochs += random.choice([-10, -5, -2, 2, 5, 10])

    for i in range(len(model.layers)):
        if random.random() < (0.1 * pm):
            model.layers[i][0] += random.choice([-10, -5, -2, 0, 0, 2, 5, 10])
            model.layers[i][0] = GeneticCalculator.ACTIVATION_FUNCTIONS[random.randrange(0, len(GeneticCalculator.ACTIVATION_FUNCTIONS))]

class GeneticCalculator:
    ACTIVATION_FUNCTIONS = ['relu', 'linear']

    def __init__(self, population, fitness_func, crossover_func=simple_crossover, selection_amount=1,
                 mutation_func=simple_mutation, muation_probablility=0.05, selection_func=simple_selection):
        self.__fitness = fitness_func
        self.__select = selection_func
        self.__crossover = crossover_func
        self.__mutate = mutation_func
        self.__generation = 0
        self.__selection_amount = selection_amount

        for i, model in enumerate(population):
            if type(model) is list: 
                population[i] = GeneticCalculator.to_model(model)

        self.__population = SortedList(population)
        self.__pm = muation_probablility

    def start(self, max_generations):
        for _ in range(max_generations):
            # Selection
            total_deleted = self.__select(self.__population, self.__selection_amount)

            births = []

            # Crossover
            for i in range(total_deleted):
                births.append(self.__crossover(self.__population[i], self.__population[i+1]))

            # Mutation & Fitness
            for model in births:
                self.__mutate(model)
                self.__fitness(model)

            self.__population.update(births)

            print("Generation: {0:3d} | Min loss: {1:14.8f}".format(self.__generation, self.__population[0]))

            self.__generation += 1

        print("#  | error               | norm  | out_ac       | epochs | hidden layers")
        print("--------------------------------------------------")
        for i, result in enumerate(self.__population):
            print("{0:2d} | {1:14.8f} | {2} | {3:12s} | {4:6d} | {5}".format(i, result.error, result.norm,
             result.out_ac, result.epochs, result.layers))

    @staticmethod
    def to_model(it):
        return Model(it[0], it[2], it[3], it[1])
