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

from sortedcontainers import SortedList
import random

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

def simple_crossover(a, b):
    model = Model()

    if a.norm is b.norm:
        model.norm = a.norm
    else:
        model.norm = random.choice((a.norm, b.norm))

    if a.out_ac is b.out_ac:
        model.out_ac = a.out_ac
    else:
        model.out_ac = random.choice((a.out_ac, b.out_ac))

    

    return model

def simple_mutation(a, b):
    pass

class GeneticCalculator:
    def __init__(self, population, fitness_func, crossover_func=simple_crossover, selection_amount=1,
                 mutation_func=simple_mutation):
        self.__fitness = fitness_func
        self.__crossover = crossover_func
        self.__mutation = mutation_func
        self.__generation = 0
        self.__selection_amount = selection_amount

        population = [(el, fitness_func(el)) for el in population]

        self.__population = SortedList(population)

    def start(self, max_generations):
        for _ in range(max_generations):
            # Selection
            del self.__population[-self.__selection_amount:]

            # Crossover
            for i in range(self.__selection_amount, step=2):
                self.__crossover(self.__population[i], self.__population[i+1])

            # Mutation

            # Fitness
