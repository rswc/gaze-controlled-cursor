import cv2
import numpy as np
from opv import OpvModel

CURSOR_MODE = False
__NORM_MIN = 0
__NORM_PTP = 1

model_fd = None
model_lm = None
model_hp = None
model_ge = None
model_cu = None

def normalize(array):
    return (array - __NORM_MIN) / __NORM_PTP


# The face processing module can be used both during calibration and cursor inference.
# During calibration we don't want to waste time initializing an untrained or inexistent
# cursor model, so the cursor_mode arg can be used to signify whether or not we want
# it to be run
def init(cursor_mode=False, norm_min=0, norm_ptp=1):
    global model_fd, model_lm, model_hp, model_ge, model_cu, CURSOR_MODE, __NORM_MIN, __NORM_PTP

    __NORM_MIN = norm_min
    __NORM_PTP = norm_ptp

    model_fd = OpvModel("face-detection-adas-0001", "GPU")
    model_lm = OpvModel("facial-landmarks-35-adas-0002", "GPU", ncs=2)
    model_hp = OpvModel("head-pose-estimation-adas-0001", "GPU", ncs=3)
    model_ge = OpvModel("gaze-estimation-adas-0002", "GPU", ncs=4)
    if cursor_mode:
        CURSOR_MODE = True
        model_cu = OpvModel("cursor-estimation-0001", "GPU", ncs=5)

def process(image, conf=0.5):
    if model_fd is None:
        raise ValueError("Face processing module not initialized! Call init() first")

    f_predictions = model_fd.Predict({'data': image})
    faces = []

    # This part is ripped straight from the face detection demo from the bootcamp,
    # except it creates Face objects instead of drawing a box
    predictions_1 = f_predictions[0][0]
    confidence = predictions_1[:, 2]
    topresults = predictions_1[(confidence > conf)]
    (h, w) = image.shape[:2]

    for result in topresults:
        box = result[3:7] * np.array([w, h, w, h])
        (xmin, ymin, xmax, ymax) = box.astype("int")

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

    # On second thought, it might've been smarter to make Face a data-only object,
    # and leave all of the calculations in process(), but it's a bit late for that now
    def __init__(self, p1, p2, conf, image):
        self.p1 = p1
        self.p2 = p2
        self.conf = conf
        self.image = image[p1[1]:p2[1], p1[0]:p2[0], :]
        self.gaze = np.array([])
        self.h_pose = np.array([])
        self.cursor = None

        (h, w) = self.image.shape[:2]
        self.size = h * w
        
        # Landmarks
        lm = model_lm.Predict({'data': self.image})[0]

        # Only points 0-3 (eye corners) interest us
        self.eye_pts = []
        for i in range(0, 8, 2):
            self.eye_pts.append(np.array([lm[i] * w, lm[i + 1] * h]).astype("int"))

        # Eyes
        (self.l_p2, self.l_p1, self.l_mid, self.l_eye) = self.get_eye(self.eye_pts[1], self.eye_pts[0])
        (self.r_p2, self.r_p1, self.r_mid, self.r_eye) = self.get_eye(self.eye_pts[2], self.eye_pts[3])

        # I guess eye detection can sometimes fail? At this comment's time of writing this line is 17 days old
        if self.l_eye.size is 0 or self.l_eye[0].size is 0 or self.r_eye.size is 0 or self.r_eye[0].size is 0:
            return

        # Head pose estimation
        self.h_pose = np.array([angle[0][0] for angle in model_hp.Predict({'data': self.image})])

        # Gaze estimation
        self.gaze = model_ge.Predict({'left_eye_image': self.l_eye, 'right_eye_image': self.r_eye, 'head_pose_angles': self.h_pose})[0]

        if CURSOR_MODE:
            ret = model_cu.Predict({'data': self.__cursor_data})
            self.cursor = ret

    def get_eye(self, p1, p2):
        # Get the norm & midpoint of the eye
        eye_norm = cv2.norm(p1 - p2)
        midpoint = (p1 + p2) / 2

        # Make a box around the midpoint. Size depends on norm
        (xmin, ymin) = (midpoint - eye_norm * 1.6).astype("int")
        (xmax, ymax) = (midpoint + eye_norm * 1.6).astype("int")

        # Don't ask
        assert xmax > midpoint[0] and xmin < midpoint[0]
        assert ymax > midpoint[1] and ymin < midpoint[1]

        # The eye images are cut out from the face cutout (self.image),
        # which, in itself, is cut out from the frame supplied by the camera
        # This part here is to ensure we do not cut out something that does not exist
        (w, h) = self.p1
        xmin = max(xmin, 0)
        xmax = min(xmax, w)
        ymin = max(ymin, 0)
        ymax = min(ymax, h)

        # The method returns a tuple: first, the two points of the eye's bounding box,
        # then, the midpoint, and finally th image cutout of the eye.
        # Points are in global, frame-space coordinates, I think?
        return ((xmax + w, ymax + h),
                (xmin + w, ymin + h),
                (int(midpoint[0] + w), int(midpoint[1] + h)),
                self.image[ymin:ymax, xmin:xmax, :])

    @property
    def __cursor_data(self):
        """
        Returns data required by the cursor model in flat list format
        """
        return normalize(np.array([self.l_mid[0], self.l_mid[1], self.r_mid[0], self.r_mid[1],
                self.size, self.gaze[0], self.gaze[1], self.gaze[2],
                self.h_pose[0], self.h_pose[1], self.h_pose[2]]))

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
        """
        Debug function. Shows the eye cutouts in two separate windows.
        """
        if self.l_eye.size > 0: 
            cv2.imshow('left eye', self.l_eye)
            cv2.resizeWindow('left eye', 256, 256)
        if self.r_eye.size > 0:
            cv2.imshow('right eye', self.r_eye)
            cv2.resizeWindow('right eye', 256, 256)

##
# THE FOLLOWING CODE IS NOT UTILIZED IN THE CURRENT VERSION OF THE PROJECT
##

# This was my attempt at reducing the 'jitter' of gaze vectors from the sample model,
# but it didn't function properly. I got distracted with the model calculator and genetic algorithm,
# so I did not manage to fix it before the deadline. It's a bit like that power plant Enron was building
# in India. Please enjoy.
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

        # Make sure std is not too high
        self.std = np.sum(np.std(self.__values[:], axis=0))
        if self.std > self.std_limit:
            self.invalidate()
            return

        if self.valid:
            self.__avg = self.__sum / self.length

        elif len(self.__values) is self.length:
            # The average is considered valid when the desired number of values has been reached
            self.valid = True

    def invalidate(self):
        self.valid = False
        self.__values = []
        self.__sum = np.array(self.__size)
        self.__next = 0

    def get(self):
        if self.valid:
            return (self.__avg, self.std)
        else:
            return False

    def draw_vector(self, image, face):
        if not self.valid or not self.__size is 3:
            return
        gaze_arrow = (np.array([self.__avg[0], -self.__avg[1]]) * 0.4 * face.image.shape[0]).astype("int")
        cv2.arrowedLine(image, face.l_mid, (face.l_mid[0] + gaze_arrow[0], face.l_mid[1] + gaze_arrow[1]), (24, 24, 240))
        cv2.arrowedLine(image, face.r_mid, (face.r_mid[0] + gaze_arrow[0], face.r_mid[1] + gaze_arrow[1]), (24, 24, 240))

    def draw_point(self, image):
        if not self.valid or not self.__size is 2:
            return
        cv2.circle(image, (int(self.__avg[0]), int(self.__avg[1])), 2, (24, 240, 48), -1)
        