import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

t = np.load("configurations_all.npy", allow_pickle = True)
print(type(t))
print("dtype:", t.dtype)
print("ndim: ", t.ndim)
print("shape:", t.shape)
print("size: ", t.size)
#print(t)


configurations = np.load("configurations_all.npy", allow_pickle = True)

#print("#  | loss         | norm  | out_ac       | epochs | hidden layers |")
#print("--------------------------------------------------")
#for i, result in enumerate(configurations):
 #   print("{0:2d} | {1:6.10f} | {2:5} | {3:12s} | {4:6d} | {5}".format(i, result[4], result[0], result[2], result[3], result[1]))

loss = np.linspace(0,5,10, endpoint=True)
norm = np.linspace(1,5,10,endpoint=True)
#print(loss)
plt.figure()
plt.plot(norm, loss)



#plt.hist(t['norm'],bins=5)
#plt.title('Absences')
