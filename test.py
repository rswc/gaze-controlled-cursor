from tensorflow import keras, device
import numpy as np
import random
#from IPython.display import SVG
#from keras.utils import model_to_dot

from keras.utils import plot_model

random.seed(1984)


def printtest():
    print("k")
    for i in range(15):
        index = random.randint(0, 400)
        
        normed = _normalize(testing_data[index:index+1], minim, ptp)
        # normed = normalize__(testing_data[index:index+1])
        td = normed

        print(type(td), td.shape, td)

        print('\nModel returns:', model.predict(td))
        print('Expected:', testing_labels[index])


def normalize(array):
    return (array - array.min(0)) / array.ptp(0)
    
def _normalize(array, mini, ptpp):
    return (array - mini / ptpp)


raw_data = np.load("new_combined_results.npy", allow_pickle=True)

# del datapoints with empty vectors
mask = np.ones(len(raw_data), dtype=bool)
data = np.zeros((len(raw_data), 13), dtype='O')

for i, dp in enumerate(raw_data):
    if (dp[3].size is 0 or dp[5].size is 0):
        mask[i] = False
    
    #flattened = np.concatenate([[*dp[0], dp[1][0] / 1920, dp[1][1] / 1080, dp[2][0] / 1920, dp[2][1] / 1080,], dp[3], [dp[4] / 2073600], dp[5]], axis=0).astype('float32')
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
norm_training_labels = normalize(training_labels)

norm_testing_data = normalize(testing_data)
norm_testing_labels = normalize(testing_labels)

print(testing_data)


minim = testing_data.min(0)
ptp = testing_data.ptp(0)

print('min: ', minim, ' ptp: ', ptp)

#print(norm_training_data)

del data

first = [[512,253,254, 200,128,64,60], [512,253,254, 200,128,64,60],[512,253,254, 200,128,64,60]]
for a in range(3):
    print(a, "MODEL")
    model = keras.Sequential()
    model.add(keras.Input(shape=(11,), name='data'))
    for k in range(7):
        model.add(keras.layers.Dense(first[a][k], activation='relu'))





    weights = np.array([1,1,1,1,1,1,1,1,1,1,1])


    model.add(keras.layers.Dense(2, activation='linear', name='output'))

    model.compile(optimizer='adam', loss='mean_absolute_error', 
                metrics=['mean_absolute_error'])

    model.fit(norm_training_data, training_labels, epochs=65, class_weight=weights)

    #plot_model(model, to_file='model.png')

    test_loss, test_mse = model.evaluate(norm_testing_data, testing_labels, verbose=0)

    #printtest()


    


    m = testing_data.min(0)
    p = testing_data.ptp(0)

    succ = 0
    for i in range(15):
        z = random.randrange(len(testing_data)-1)
        t = [[(testing_data[z] - m )/ p]]

        pred = model.predict(t)
        exp = testing_labels[z]
        print('---------------------------')
        print('Test: ', i)
        if(abs(pred[0][0] - exp[0]) < 0.15 and abs(pred[0][1]-exp[1]) < 0.15 ):
            #print("Succeeded")
            succ+=1
        print('pred| x=', float(int(pred[0][0]*100/100)),"| y=", float(int(pred[0][1]*100)/100))
        print('exp | x=', float(int(exp[0]*100)/100),"| y=", float(int(exp[1]*100)/100))


    print("Predictions close to expected (<0.15 inacuraccy) ", succ, "/15")
    print("Test MAE:", test_mse)

    