#pylint: disable=missing-docstring, invalid-name
import cv2
import numpy as np
from opv import OpvModel

def DrawBoundingBoxes(predictions, image, conf=0.5):
    canvas = image.copy()                             # copy instead of modifying the original image
    predictions_1 = predictions[0][0]                 # subset dataframe
    confidence = predictions_1[:,2]                   # getting conf value [image_id, label, conf, x_min, y_min, x_max, y_max]
    topresults = predictions_1[(confidence>conf)]     # choosing only predictions with conf value bigger than treshold
    (h, w) = canvas.shape[:2]                        # setting the variable h and w according to image height
    faces = []
    
    #
    for detection in topresults:
        box = detection[3:7] * np.array([w, h, w, h]) # determine box location
        (xmin, ymin, xmax, ymax) = box.astype("int") # assign box location value to xmin, ymin, xmax, ymax

        if xmin - 15 >= 0: #TODO: this is ugly as shit
            xmin = xmin - 15
        else:
            xmin = 0
        if xmax + 15 <= w:
            xmax = xmax + 15
        else:
            xmax = w
        if ymin - 15 >= 0:
            ymin = ymin - 15
        else:
            ymin = 0
        if ymax + 15 <= h:
            ymax = ymax + 15
        else:
            ymax = h

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

model_fd = OpvModel("face-detection-adas-0001", "GPU")
model_lm = OpvModel("facial-landmarks-35-adas-0002", "GPU", ncs=2)
model_hp = OpvModel("head-pose-estimation-adas-0001", "GPU", ncs=3)

while video.isOpened():
    ret, frame = video.read()
    if not ret: 
        break

    predictions_fd = model_fd.Predict({'data': frame})
    frame, faces = DrawBoundingBoxes(predictions_fd, frame) #TODO: proper separation

    predictions_lm = []
    predictions_hp = []

    for face in faces:
            predictions_lm.append(model_lm.Predict({'data': face[0]}))
            predictions_hp.append(model_hp.Predict({'data': face[0]}))

            (x, y) = face[1:3]
            (h, w) = face[0].shape[:2]
            frame = DrawPoints(predictions_lm, frame, x, y, w, h)

    
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.waitKey(0)
