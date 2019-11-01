import numpy as np

a = np.load("captured_calibrations/capresults.npy", allow_pickle = True)
b = np.load("captured_calibrations/capresultsDif.npy", allow_pickle = True)
c = np.load("captured_calibrations/capresultsRotatin.npy", allow_pickle = True)

combined = np.concatenate((a,b,c))

np.save("combined_results.npy", combined)