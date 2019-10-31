from tensorflow import keras, device
import numpy as np
import random

random.seed(1984)


def printtest():
    for i in range(15):
        index = random.randint(0, 1100)
        
        normed = _normalize(testing_data[index:index+1], minim, ptp)
        # normed = normalize__(testing_data[index:index+1])
        td = normed

        print(type(td), td.shape, td)

        print('\nModel returns:', model.predict(td))
        print('Expected:', testing_labels[index])


def _normalize(array, mini, ptpp):
    return (array - mini / ptpp)

def normalize(array):
    return (array - array.min(0)) / array.ptp(0)


raw_data = np.load("capresultsRotatin.npy", allow_pickle=True)

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

model = keras.Sequential()
model.add(keras.Input(shape=(11,), name='data'))
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dense(512, activation='relu'))






weights = np.array([1,1,1,1,2,2,2,1,2,2,2])


model.add(keras.layers.Dense(2, activation='linear', name='output'))

model.compile(optimizer='adam', loss='mean_absolute_error', 
            metrics=['mean_absolute_error'])

model.fit(norm_training_data, training_labels, epochs=55, class_weight=weights)



test_loss, test_mse = model.evaluate(norm_testing_data, testing_labels, verbose=0)

#printtest()

print("\nTest MSE:", test_mse)



#a = np.array([minim, ptp, testing_data[600]])


m = testing_data.min(0)
p = testing_data.ptp(0)

#print('m', m,'p',p)
#nor = normalize(a)




t = [[(testing_data[400] - m )/ p]]

print('normalized: ', t)
print('pred: ', model.predict(t))
print('exp: ', testing_labels[400])