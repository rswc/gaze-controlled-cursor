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

        faces.append(Face((xmin, ymin), (xmax, ymax), result[2], image))

    return faces

class Face:
    """
    General class for calculating all necessary face attributes and displaying the results.

    Args:
        p1: Bounding box minimum point (in image fraction, 0-1)
        p1: Bounding box maximum point (in image fraction, 0-1)
        conf: Face detection confidence
        image: Image from which the face will be cut out

    Attributes:
        p1: Bounding box minimum point (in image fraction, 0-1)
        p1: Bounding box maximum point (in image fraction, 0-1)
        conf: Face detection confidence
        image: Image of the face
        gaze: The estimated gaze vector
        h_pose: The estimated head pose
        size: Area of the image (in px^2)
        eye_pts: The four eye corner points, local coordinates (in numpy arrays of image fractions, 0-1)
        l_mid, r_mid: The two eye midpoints, global coordinates (in tuples of image fractions, 0-1)
        l_p1, l_p2, r_p1, r_p2: The four eye corner points, global coordinates (in tuples of image fractions, 0-1)
        l_eye, r_eye: Images of the eyes, cut out from the local face image
    """

    def __init__(self, p1, p2, conf, image):
        self.p1 = p1
        self.p2 = p2
        self.conf = conf
        self.image = image[p1[1]:p2[1], p1[0]:p2[0], :]
        self.gaze = np.array([])
        self.h_pose = np.array([])

        (h, w) = self.image.shape[:2]
        self.size = h * w
        
        # Landmarks
        lm = model_lm.Predict({'data': self.image})[0]

        # Only points 0 to 3 interest us
        self.eye_pts = []
        for i in range(0, 8, 2):
            self.eye_pts.append(np.array([lm[i] * w, lm[i + 1] * h]).astype("int"))

        # Eyes #TODO: make this a lot better
        (self.l_p2, self.l_p1, self.l_mid, self.l_eye) = self.get_eye(self.eye_pts[1], self.eye_pts[0])
        (self.r_p2, self.r_p1, self.r_mid, self.r_eye) = self.get_eye(self.eye_pts[2], self.eye_pts[3])

        if self.l_eye.size is 0 or self.l_eye[0].size is 0 or self.r_eye.size is 0 or self.r_eye[0].size is 0:
            return

        # Head pose estimation
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
        cv2.rectangle(image, self.p2, self.p1, (230, 230, 230), 2)
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
        if not self.gaze.size:
            return
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

class PropertyAverager:
    """
    Instance-based helper class for averaging face data. It will first attempt
    to collect the specified minimum number of samples, and can then return the average
    and standard deviation every frame, unless it becomes invalidated (for ex. by std
    exceeding the limit). In that case, the process will start over.

    Args:
        length: The desired amount of samples
        std_limit: (Optional) If the standard deviation exceeds this limit,
                   the instance is invalidated and reset
    """
    def __init__(self, length, std_limit=0.5, size=3):
        self.length = length
        self.std_limit = std_limit
        self.std = 0
        self.valid = False
        self.__values = []
        self.__avg = np.zeros(size)
        self.__sum = np.zeros(size)
        self.__next = 0                        # The id of the value in __values next to be replaced
        self.__size = size                     # Size of the averaged vector

    def add(self, vector):
        if vector.size != self.__size:
            self.invalidate()
            return

        # Fill __values to capacity, then replace the oldest one each frame
        if len(self.__values) < self.length:
            self.__values.append(vector)
            self.__sum = self.__sum + vector
        else:
            self.__sum = self.__sum - self.__values[self.__next] + vector
            self.__values[self.__next] = vector

        # Cycle through values to replace
        self.__next = self.__next + 1
        if self.__next >= self.length:
            self.__next = 0

        if self.valid:
            # Make sure std is not too high
            self.std = np.sum(np.std(self.__values, axis=0))
            if self.std > self.std_limit:
                self.invalidate()
                return

            self.__avg = self.__sum / self.length

        elif len(self.__values) is self.length:
            # The average is considered valid when the desired number of values has been reached
            self.valid = True

    def invalidate(self):
        self.valid = False
        self.__values = []
        self.__sum = np.array([0., 0., 0.])
        self.__next = 0

    def get(self):
        if self.valid:
            return (self.__avg, self.std)
        else:
            return False

    def draw(self, image, face):
        if not self.valid:
            return
        gaze_arrow = (np.array([self.__avg[0], -self.__avg[1]]) * 0.4 * face.image.shape[0]).astype("int")
        cv2.arrowedLine(image, face.l_mid, (face.l_mid[0] + gaze_arrow[0], face.l_mid[1] + gaze_arrow[1]), (24, 24, 240))
        cv2.arrowedLine(image, face.r_mid, (face.r_mid[0] + gaze_arrow[0], face.r_mid[1] + gaze_arrow[1]), (24, 24, 240))
        