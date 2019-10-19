#pylint: disable=missing-docstring, invalid-name
import cv2
import numpy as np
import face_processing
from face_processing import Face


video = cv2.VideoCapture(0)


while video.isOpened():
    ret, frame = video.read()
    if not ret: 
        break

    faces = face_processing.process(frame)

    for face in faces:
        face.draw_bbox(frame)
        face.draw_pts(frame)
        face.draw_gaze(frame)

    
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.waitKey(0)
