import numpy as np

a = np.load("captured_calibrations/capresults.npy", allow_pickle = True)
b = np.load("captured_calibrations/capresultsDif.npy", allow_pickle = True)
c = np.load("captured_calibrations/capresultsRotatin.npy", allow_pickle = True)
d = np.load("captured_calibrations/capresults_new_points.npy", allow_pickle = True)
e = np.load("captured_calibrations/capresults_new_points_big.npy", allow_pickle = True)
f = np.load("captured_calibrations/capresults_only_segments_1.npy", allow_pickle = True)
g = np.load("captured_calibrations/capresults_only_segments.npy", allow_pickle = True)
combined = np.concatenate((d,e))

np.save("combined_results.npy", combined)