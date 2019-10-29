from tensorflow import keras, device
import numpy as np
import random
from datetime import datetime
random.seed(datetime.now())


def printtest():
    for i in range(3):
        index = random.randint(0, 500)
        td = testing_data[index:index+1]

        #print(type(td), td.shape, td)

        #print('\nModel returns:', model.predict(td))
        #print('Expected:', testing_labels[index])

        points.append([model.predict(td),  testing_labels[index]])

def normalize(array):
    return (array - array.min(0)) / array.ptp(0)


capture_results = []

ran = 100

for it in range(ran):
    print('TEST ', it, ' z ',ran)
    raw_data = np.load("capresults.npy", allow_pickle=True)

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




    #print(norm_training_data)

    del data

    lay = random.randint(1,20)
    model = keras.Sequential()

    neur = [16,16,32,32,64, 64,128,128,256,256,512,1024]
   
    model.add(keras.Input(shape=(11,), name='data'))

    layer = []
    for k in range(lay):
        ne = random.randint(0,len(neur)-1)
        model.add(keras.layers.Dense(neur[ne], activation='relu'))
        layer.append(neur[ne])

    r = random.randint(0,4)
    eyes = [1,1,1,1,2]
    rr = random.randint(0,4)
    gaze = [1,1,1,1,2]
    rrr = random.randint(0,4)
    face_s = [1,1,1,1,2]
    rrrr = random.randint(0,4)
    head_p = [1,1,1,1,2]

    ws = [eyes[r], gaze[rr], face_s[rrr], head_p[rrrr]]
    weights = np.array([eyes[r],eyes[r],eyes[r],eyes[r],gaze[rr], gaze[rr],gaze[rr],face_s[rrr],head_p[rrrr], head_p[rrrr], head_p[rrrr]])


    model.add(keras.layers.Dense(2, activation='linear', name='output'))

    model.compile(optimizer='adam', loss='mean_absolute_error', 
                metrics=['mean_absolute_error'])

    ep = random.randint(40,70)
    model.fit(norm_training_data, norm_training_labels, epochs=ep, class_weight=weights)



    test_loss, test_mse = model.evaluate(testing_data, testing_labels, verbose=0)

    #points = []

    
    #for i in range(3):
        #index = random.randint(0, 500)
       # td = testing_data[index:index+1]

        #print(type(td), td.shape, td)

        #print('\nModel returns:', model.predict(td))
        #print('Expected:', testing_labels[index])

        #points.append([model.predict(td),  testing_labels[index]])
    
    #print(points)

    capture_results.append([test_mse, ep, ws, lay ,layer])
    print("\nTest",it," MSE:", test_mse)
print(type(capture_results), capture_results)


np.save("captured_randoms_first_try", capture_results)