from genetic_calculator import GeneticCalculator, Model
from tensorflow import keras
from tensorboard.plugins.hparams import api as hp
import numpy as np

def normalize(array):
    return (array - array.min(0)) / array.ptp(0)

raw_data = np.load("captured_calibrations/combined_results.npy", allow_pickle=True)

# del datapoints with empty vectors
mask = np.ones(len(raw_data), dtype=bool)
data = np.zeros((len(raw_data), 13), dtype='O')

for i, dp in enumerate(raw_data):
    if (dp[3].size is 0 or dp[5].size is 0):
        mask[i] = False
    
    flattened = np.concatenate([[*dp[0], *dp[1], *dp[2]], dp[3], [dp[4]], dp[5]], axis=0).astype('float32')
    if flattened.shape[0] is data.shape[1]:
        data[i] = flattened
del raw_data

data = data[mask, ...]

# split into training and testing sets
mask = np.random.choice([True, False], len(data), p=[0.75, 0.25])

training_data = data[mask, ...][:, 2:]
training_labels = data[mask, ...][:, :2]

mask = ~mask

testing_data = data[mask, ...][:, 2:]
testing_labels = data[mask, ...][:, :2]

norm_training_data = normalize(training_data)
norm_testing_data = normalize(testing_data)

del data

def model_fitness(tpl):
    global training_data, training_labels, norm_training_data, norm_training_labels, testing_data, testing_labels

    model = keras.Sequential()
    model.add(keras.Input(shape=(11,), name='data'))

    for layer in tpl.layers:
        model.add(keras.layers.Dense(layer[0], activation=layer[1]))

    model.add(keras.layers.Dense(2, activation=tpl.out_ac, name='output'))

    optimizer = keras.optimizers.Adam(learning_rate=tpl.learning_rate)
    
    model.compile(optimizer=optimizer, loss='mean_absolute_error')

    hparams = {
        "Num_layers": len(tpl.layers),
        "Epochs": tpl.epochs,
        "Batch_size": tpl.batch_size,
        "Learning_rate": tpl.learning_rate
    }

    model.fit(norm_training_data, training_labels, epochs=tpl.epochs, batch_size=tpl.batch_size, verbose=0)

    return model.evaluate(norm_testing_data, testing_labels, verbose=0)

# Initial population
pop = GeneticCalculator.random(15, min_layers=4, max_layers=9, layer_size_choice=[16, 32, 48, 64, 128, 256, 512],
                             layer_activation=['relu', 'linear'], out_ac_choice=['relu', 'linear'],
                             epochs_choice=[40, 50, 60], batch_size_choice=[24, 32, 40], learning_rate_choice=[0.001])

# Gen calc instance
gen_calc = GeneticCalculator(pop, model_fitness, mutation_probability=0.25, selection_amount=6, selection_probability=0.6, verbose=3)
gen_calc.start(2)

while True:
    c = input("Set finished.\nC - Continue  E - Edit  Q - Quit")
    if c is 'C':
        gen_calc.start(int(input("Number of generations > ")))
    elif c is 'E':
        gen_calc.reconfigure(int(input("Selection amount > ")), float(input("Mutation probability > ")), float(input("Selection probability > ")))
    else:
        break
