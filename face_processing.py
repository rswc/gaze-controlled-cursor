import cv2
import numpy as np
from opv import OpvModel

model_fd = OpvModel("face-detection-adas-0001", "GPU")
model_lm = OpvModel("facial-landmarks-35-adas-0002", "GPU", ncs=2)
model_hp = OpvModel("head-pose-estimation-adas-0001", "GPU", ncs=3)
model_ge = OpvModel("gaze-estimation-adas-0002", "GPU", ncs=4)

def process(image, conf=0.5):
    f_predictions = model_fd.Predict({'data': image})
    faces = []
    
    predictions_1 = f_predictions[0][0]                 # subset dataframe
    confidence = predictions_1[:, 2]                   # getting conf value [image_id, label, conf, x_min, y_min, x_max, y_max]
    topresults = predictions_1[(confidence > conf)]     # choosing only predictions with conf value bigger than treshold
    (h, w) = image.shape[:2]

    for result in topresults:
        box = result[3:7] * np.array([w, h, w, h]) # determine box location
        (xmin, ymin, xmax, ymax) = box.astype("int") # assign box location value to xmin, ymin, xmax, ymax

        xmin = max(xmin - 15, 0)
        xmax = min(xmax + 15, w)
        ymin = max(ymin - 15, 0)
        ymax = min(ymax + 15, h)

        face_crop = image[ymin:ymax, xmin:xmax, :]

        faces.append(Face((xmin, ymin), (xmax, ymax), result[2], face_crop))

    return faces

class Face:
    #TODO: docstrings

    def __init__(self, p1, p2, conf, image):
        self.p1 = p1
        self.p2 = p2
        self.conf = conf
        self.image = image
        
        # Landmarks
        lm = model_lm.Predict({'data': self.image})[0]
        (h, w) = self.image.shape[:2]

        self.eye_pts = []
        for i in range(0, 8, 2):
            self.eye_pts.append(np.array([lm[i] * w, lm[i + 1] * h]).astype("int"))

        # Eyes #TODO: make this a lot better
        (self.l_p2, self.l_p1, self.l_mid, self.l_eye) = self.get_eye(self.eye_pts[1], self.eye_pts[0])
        (self.r_p2, self.r_p1, self.r_mid, self.r_eye) = self.get_eye(self.eye_pts[2], self.eye_pts[3])

        if self.l_eye.size is 0 or self.l_eye[0].size is 0 or self.r_eye.size is 0 or self.r_eye[0].size is 0:
            return

        # Head pose
        self.h_pose = np.array([angle[0][0] for angle in model_hp.Predict({'data': self.image})])

        # Gaze estimation
        self.gaze = model_ge.Predict({'left_eye_image': self.l_eye, 'right_eye_image': self.r_eye, 'head_pose_angles': self.h_pose})[0]

    def get_eye(self, p1, p2):
        eye_norm = cv2.norm(p1 - p2)
        midpoint = (p1 + p2) / 2

        (xmin, ymin) = (midpoint - eye_norm * 1.6).astype("int")
        (xmax, ymax) = (midpoint + eye_norm * 1.6).astype("int")

        assert xmax > midpoint[0] and xmin < midpoint[0]
        assert ymax > midpoint[1] and ymin < midpoint[1]

        (w, h) = self.p1
        xmin = max(xmin, 0)
        xmax = min(xmax, w)
        ymin = max(ymin, 0)
        ymax = min(ymax, h)

        return ((xmax + w, ymax + h), (xmin + w, ymin + h), (int(midpoint[0] + w), int(midpoint[1] + h)), self.image[ymin:ymax, xmin:xmax, :])

    def draw_bbox(self, image):
        cv2.rectangle(image, self.p2, self.p1, (230, 230, 230), 2)  # make a rectangle
        cv2.putText(image, str(round(self.conf * 100, 2))+"%", self.p2, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0,0), 2)

    def draw_pts(self, image):
        (w, h) = self.p1
        for i in range(4):
            c_x = self.eye_pts[i][0] + w
            c_y = self.eye_pts[i][1] + h
            cv2.circle(image, (c_x, c_y), 2, (240, 240, 24), -1)
        cv2.circle(image, self.l_mid, 2, (24, 48, 240), -1)
        cv2.circle(image, self.r_mid, 2, (24, 48, 240), -1)
        cv2.rectangle(image, self.l_p2, self.l_p1, (230, 230, 230), 1)
        cv2.rectangle(image, self.r_p2, self.r_p1, (230, 230, 230), 1)

    def draw_gaze(self, image):
        gaze_arrow = (np.array([self.gaze[0], -self.gaze[1]]) * 0.4 * self.image.shape[0]).astype("int")
        cv2.arrowedLine(image, self.l_mid, (self.l_mid[0] + gaze_arrow[0], self.l_mid[1] + gaze_arrow[1]), (240, 24, 24))
        cv2.arrowedLine(image, self.r_mid, (self.r_mid[0] + gaze_arrow[0], self.r_mid[1] + gaze_arrow[1]), (240, 24, 24))

    def show_eyes(self):
        if self.l_eye.size > 0: 
            cv2.imshow('left eye', self.l_eye)
            cv2.resizeWindow('left eye', 256, 256)
        if self.r_eye.size > 0:
            cv2.imshow('right eye', self.r_eye)
            cv2.resizeWindow('right eye', 256, 256)

