#pylint: disable=invalid-name
# Pre-determined starting population
# Fill to capacity by randomizing/mutation

# Also look into
# -- using more than two parents for crossover
# -- multiple children from single crossover
# -- keeping track of best overall model

import random
import numpy as np
from tensorflow import keras
from sortedcontainers import SortedList

class Model:
    __slots__ = ['norm', 'out_ac', 'epochs', 'layers', 'error']

    def __init__(self, norm=False, out_ac=None, epochs=0, layers=None, error=0):
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

    def __str__(self):
        layers = ". "
        for layer in self.layers:
            layers += "({0}, {1}) ".format(layer[0], layer[1][0])

        return "{0:11.8f} | {1} | {2:6s} | {3:3d} | {4}".format(self.error, self.norm, self.out_ac, self.epochs, layers)

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
            return int(x * a + (1-x) * b)
        else:
            return int((1+x) * a - x * b)

    @staticmethod
    def layer_crossover(long, short, size_diff):
        s_len = len(short)
        x_point = random.randrange(1, s_len)
        diff_x_point = random.randint(0, size_diff)

        if random.randint(0, 1) is 1:
            return long[0:x_point] + short[x_point:] + long[s_len:s_len + diff_x_point]
        else:
            return short[0:x_point] + long[x_point:s_len] + long[s_len:s_len + diff_x_point]

def simple_selection(pop, selection_amount, min_pop_size=10, protected_amount=6):
    total_deleted = 0

    for i in range(len(pop) - protected_amount):
        if random.random() < 0.5:
            del pop[-i-1]
            total_deleted += 1
        if total_deleted is selection_amount:
            return total_deleted

    if total_deleted < selection_amount and len(pop) > min_pop_size:
        del pop[-selection_amount+total_deleted:]

    return total_deleted

def simple_crossover(a, b):
    model = Model()

    model.norm = Util.choice_crossover(a.norm, b.norm)
    model.out_ac = Util.choice_crossover(a.out_ac, b.out_ac)
    model.epochs = Util.normal_crossover(a.epochs, b.epochs)

    l = len(a.layers) - len(b.layers)
    if l >= 0:       # a >= b
        model.layers = Util.layer_crossover(a.layers, b.layers, l)
    elif l < 0:      # a < b
        model.layers = Util.layer_crossover(b.layers, a.layers, -l)

    return model

def simple_mutation(model, pm):
    if random.random() < pm:
        model.out_ac = GeneticCalculator.ACTIVATION_FUNCTIONS[random.randrange(0, len(GeneticCalculator.ACTIVATION_FUNCTIONS))]
    if random.random() < pm:
        model.epochs += random.choice([-7, -5, -2, 2, 5, 7])

    for i in range(len(model.layers)):
        if random.random() < (0.7 * pm):
            model.layers[i][0] += random.choice([-10, -5, -2, 2, 5, 10])
            model.layers[i][1] = GeneticCalculator.ACTIVATION_FUNCTIONS[random.randrange(0, len(GeneticCalculator.ACTIVATION_FUNCTIONS))]

    return model

class GeneticCalculator:
    """
    Runs my implementation of a genetic algorithm. And it could run yours too, because
    all of the functions are swappable!

    Args:
        population: Initial population
        fitness_func: Function for calculating the fitness of a given model. Should
                      accept a Model object and return the loss value of a test
        selection_amount: (default=1) Max amount of models to die each generation
        selection_probability: (default=0.4) Dictates which models get selected for crossover
                               or removal from the population. As it approaches 1, the selection
                               converges around the first and last models in the population.
                               If it is equal to 1, the calculator will use selection_amount of best,
                               and worst models, respectively
        mutation_probability: (default=0.05) Mutation probability value passed to mutation_func
        selection_func: (default=simple_selection) Function to be used for killing models.
                        Should accept a SortedList with the total population and selection_amount.
                        Should return the total number of models deleted. Should use del to remove models
        crossover_func: (default=simple_crossover) Function to be used in the crossover step.
                        Should accept two Model objects. Should return a new Model object.
                        Only (2 parents => 1 child) operations are supprted as of today.
        mutation_func: (default=simple_mutation) Function to be used for mutation. Applied to every
                       new model before it enters the population. Should accept a Model object
                       and mutation_probability. Should return the modifed model
        verbose: (default=1) Verbosity mode of the calcuator:
                                0 - silent
                                1 - will print th entire population after finishing a set
                                2 - will also print a summary of the best models each generation
                                3 - will also print the models added each generation
    """
    ACTIVATION_FUNCTIONS = ['relu', 'linear']

    def __init__(self, population, fitness_func, crossover_func=simple_crossover, selection_amount=1,
                 mutation_func=simple_mutation, mutation_probablility=0.1, selection_func=simple_selection,
                 verbose=1, selection_probability=0.4):
        self.__fitness = fitness_func
        self.__select = selection_func
        self.__crossover = crossover_func
        self.__mutate = mutation_func
        self.__generation = 0
        self.__selection_amount = selection_amount
        self.__pm = mutation_probablility
        self.__ps = selection_probability
        self.__verbosity = verbose

        with open('out.txt', 'w') as OUTPUT_FILE:
            for i, model in enumerate(population, 0):
                if self.__verbosity > 1:
                    print("Processing initial population model ({0}/{1})...".format(i, len(population)))
                    OUTPUT_FILE.write("Processing initial population model ({0}/{1})...\n".format(i, len(population)))

                if type(model) is list: 
                    model = GeneticCalculator.to_model(model)
                model.error = self.__fitness(model)
                population[i] = model

        self.__population = SortedList(population)

    def start(self, max_generations):
        """
        Starts a new set of calculations

        Args:
            max_generations: Number of generations after which the set is considered finished.
                             Calculations can be continued by calling start() again.
        """
        with open('out.txt', 'a') as OUTPUT_FILE:
            for _ in range(max_generations):
                if self.__generation % 10 is 0 and self.__generation > 0:
                    keras.backend.clear_session()

                if self.__verbosity > 2:
                    print("Generation starting...")
                    OUTPUT_FILE.write("Generation starting...\n")

                # Selection
                total_deleted = self.__select(self.__population, self.__selection_amount)

                parents = []

                # Selection of parents
                for _ in range(total_deleted * 2):
                    for model in self.__population:
                        if not model in parents and random.random() < self.__ps:
                            parents.append(model)

                # Crossover, mutation & fitness
                for _ in range(total_deleted):
                    model = self.__crossover(parents.pop(), parents.pop())
                    model = self.__mutate(model, self.__pm)
                    model.error = self.__fitness(model)
                    self.__population.add(model)
                    if self.__verbosity > 2:
                        print(" [+]", str(model))
                        OUTPUT_FILE.write(" [+]" + str(model) + "\n")

                if self.__verbosity > 1:
                    print("Generation: {0:3d}, min_loss: {1:14.8f}".format(self.__generation, self.__population[0].error))
                    OUTPUT_FILE.write("Generation: {0:3d}, min_loss: {1:14.8f}".format(self.__generation, self.__population[0].error) + "\n")
                    self.print_population(8, file=OUTPUT_FILE)

                self.__generation += 1

            if self.__verbosity > 0:
                print("\nFINAL POPULATION:")
                OUTPUT_FILE.write("\nFINAL POPULATION:\n")
                self.print_population(file=OUTPUT_FILE)

    def print_population(self, n=None, file=None):
        print("#  | error      | norm  | out_ac | epochs | hidden layers")
        print("---------------------------------------------------------")
        for i, result in enumerate(self.__population[:n]):
            print(i, "|", str(result))
        print("\n")

        if file:
            file.write("#  | error      | norm  | out_ac | epochs | hidden layers\n")
            file.write("---------------------------------------------------------\n")
            for i, result in enumerate(self.__population[:n]):
                file.write("" + str(i) + " | " + str(result) + "\n")

    @staticmethod
    def to_model(it):
        """
        Converts the iterable format used in model_calculator.py to a Model object
        """
        return Model(it[0], it[2], it[3], it[1])

    @staticmethod
    def random(num_models, min_layers, max_layers, layer_size_choice, layer_activation,
               norm_choice, out_ac_choice, epochs_choice):
        """
        Generates a list of num_models random model configurations using the options
        provided in the rest of the parameters
        """
        models = []
        for _ in range(num_models):
            layers = []
            for _ in range(random.randint(min_layers, max_layers)):
                layers.append([random.choice(layer_size_choice), random.choice(layer_activation)])
            models.append(Model(random.choice(norm_choice), random.choice(out_ac_choice),
                                random.choice(epochs_choice), layers))

        return models
