#pylint: disable=missing-docstring, invalid-name
import cv2
import numpy as np
import face_processing
from face_processing import Face

def DrawBoundingBoxes(predictions, image, conf=0.5):
    canvas = image.copy()                             # copy instead of modifying the original image
    predictions_1 = predictions[0][0]                 # subset dataframe
    confidence = predictions_1[:, 2]                   # getting conf value [image_id, label, conf, x_min, y_min, x_max, y_max]
    topresults = predictions_1[(confidence > conf)]     # choosing only predictions with conf value bigger than treshold
    (h, w) = canvas.shape[:2]                        # setting the variable h and w according to image height
    faces = []
    
    for detection in topresults:
        box = detection[3:7] * np.array([w, h, w, h]) # determine box location
        (xmin, ymin, xmax, ymax) = box.astype("int") # assign box location value to xmin, ymin, xmax, ymax

        xmin = max(xmin - 15, 0)
        xmax = min(xmax + 15, w)
        ymin = max(ymin - 15, 0)
        ymax = min(ymax + 15, h)

        faces.append([image[ymin:ymax, xmin:xmax, :], xmin, ymin])

        cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)  # make a rectangle
        cv2.putText(canvas, str(round(detection[2] * 100, 2))+"%", (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0,0), 2)
    cv2.putText(canvas, str(len(topresults))+" face(s) detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0,0), 2)
    return canvas, faces

def DrawPoints(predictions, image, x, y, w, h):
    canvas = image.copy()
    points = predictions[0][0]
    for i in range(35):
        c_x = int(points[2 * i] * w) + x
        c_y = int(points[2 * i + 1] * h) + y
        cv2.circle(canvas, (c_x, c_y), 2, (240, 240, 24), -1)
    return canvas

video = cv2.VideoCapture(0)


while video.isOpened():
    ret, frame = video.read()
    if not ret: 
        break

    faces = face_processing.process(frame)

    for face in faces:
        face.draw_bbox(frame)
        face.draw_pts(frame)
        face.show_eyes()

        # send 'em
        # you got the gaze vectors

    
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.waitKey(0)
