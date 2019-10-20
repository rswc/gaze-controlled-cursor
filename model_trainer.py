from tensorflow import keras
import numpy as np

def normalize(array):
    return (array - array.min(0)) / array.ptp(0)

def printtest():
    for i in range(17, 29):
        td = testing_data[i:i+1]

        print('\nWith data:', td)
        print('Model returns:', model.predict(td))
        print('Expected:', testing_labels[17])

# Format:
# [point, left_eye_midpoint, right_eye_midpoint, gaze_vector, face_size, head_pose]

raw_data = np.load("capresults.npy", allow_pickle=True)

# del datapoints with empty vectors
mask = np.ones(len(raw_data), dtype=bool)
data = np.zeros((len(raw_data), 13), dtype='O')

for i, dp in enumerate(raw_data):
    if (dp[3].size is 0 or dp[5].size is 0):
        mask[i] = False
    
    flattened = np.concatenate([[*dp[0], dp[1][0] / 1920, dp[1][1] / 1080, dp[2][0] / 1920, dp[2][1] / 1080,], dp[3], [dp[4] / 2073600], dp[5]], axis=0).astype('float64')
    if flattened.shape[0] is data.shape[1]:
        data[i] = flattened
    
# Format:
# [point.x, point.y, left_eye_midpoint, right_eye_midpoint, face_size, gaze_vector.x, gaze_vector.y, gaze_vector.z, head_pose.x, head_pose.y, head_pose.z,]

data = data[mask, ...]

# split into training and testing sets
mask = np.random.choice([True, False], len(data), p=[0.75, 0.25])

training_data = data[mask, ...][:, 2:]
training_labels = data[mask, ...][:, :2]

mask = ~mask

testing_data = data[mask, ...][:, 2:]
testing_labels = data[mask, ...][:, :2]


#TODO: normalize?
training_data = normalize(training_data)

# design the model
model = keras.Sequential([
    keras.Input(shape=(11,), name='data'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(2, activation='linear')
])

model.compile(optimizer='adam', loss='mean_squared_error', 
              metrics=['mean_squared_error'])

# train the model
model.fit(training_data, training_labels, epochs=32)

# test the model
test_loss, test_acc = model.evaluate(testing_data, testing_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# convert to openvino-mo-understandable format
# printtest()
