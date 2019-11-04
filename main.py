#pylint: disable=missing-docstring, invalid-name
import cv2
import numpy as np
import face_processing as fp


video = cv2.VideoCapture(0)

cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

capture = False
training_pts = [(0.04, 0.56), (0.37, 0.42), (0.7, 0.28),
                (0.1, 0.1), (0.1, 0.9), (0.9, 0.1), (0.9, 0.9),
                (0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0)]
capture_results = []
active_point = (0.5, 0.5)

fp.init(True)

while video.isOpened():
    ret, frame = video.read()
    
    if not ret:
        break

    img = np.zeros((1080, 1920, 3))

    faces = fp.process(frame)

    for face in faces:
        face.draw_bbox(frame)
        face.draw_pts(frame)
        face.draw_gaze(frame)

    if capture and len(faces) is 1:
        cv2.circle(img, (50, 50), 8, (0.24, 0.48, 0.9), -1)

            # save face & point data to capture_results
            #capture_results.append([active_point, face.l_mid, face.r_mid, gaze_avg.get()[0], face.size, face.h_pose])

    (pt_x, pt_y) = np.array([active_point[0] * 1920, active_point[1] * 1080]).astype('int')
    cv2.circle(img, (pt_x, pt_y), 5, (0.9, 0.9, 0.9), -1)

    #cv2.imshow('window', img)

    frame = cv2.resize(frame, (1344, 756))
    cv2.imshow('frame', frame)

    

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('x'):
        if not training_pts:
            break
        if capture:
            active_point = training_pts.pop()
        capture = not capture

# np.save("capresults_avg", capture_results)

video.release()
cv2.waitKey(0)
