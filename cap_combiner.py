import numpy as np

#a = np.load("captured_calibrations/capresults.npy", allow_pickle = True)
#b = np.load("captured_calibrations/capresultsDif.npy", allow_pickle = True)
#c = np.load("captured_calibrations/capresultsRotatin.npy", allow_pickle = True)
#d = np.load("captured_calibrations/capresults_new_points.npy", allow_pickle = True)
#e = np.load("captured_calibrations/capresults_new_points_big.npy", allow_pickle = True)
#f = np.load("captured_calibrations/capresults_only_segments_1.npy", allow_pickle = True)
#g = np.load("captured_calibrations/capresults_only_segments.npy", allow_pickle = True)
h = np.load("captured_calibrations/new_calibration_0.npy", allow_pickle = True)
i = np.load("captured_calibrations/new_calibration_1.npy", allow_pickle = True)
j = np.load("captured_calibrations/new_calibration_2.npy", allow_pickle = True)
k = np.load("captured_calibrations/new_calibration_3.npy", allow_pickle = True)
l = np.load("captured_calibrations/new_calibration_4.npy", allow_pickle = True)
#m = np.load("captured_calibrations/new_calibration_5.npy", allow_pickle = True)#back forth
n = np.load("captured_calibrations/new_calibration_6.npy", allow_pickle = True)
combined = np.concatenate((h,i,j,k,l,n))

np.save("new_combined_results.npy", combined)