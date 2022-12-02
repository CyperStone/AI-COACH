import vg
import cv2
import numpy as np
import mediapipe as mp


class EventQueue:

    def __init__(self):
        self.queue = list()

    def append(self, text, color, counts):
        dict_ = {'text': text, 'color': color}

        for _ in range(counts):
            self.queue.append(dict_)

    def get_update(self):
        if self.queue:
            return self.queue.pop(0)

        else:
            return None


class SoundsQueue:

    def __init__(self):
        self.queue = list()

    def append(self, name):
        self.queue.append(name)

    def get(self):
        if self.queue:
            return self.queue[0]

        else:
            return None

    def update(self):
        self.queue = self.queue[1:]

    def refresh(self, exceptions):
        self.queue = [name for name in self.queue if name in exceptions]

    def is_free(self):
        return not bool(self.queue)


class StateQueue:

    def __init__(self, max_size, init_value):
        self.max_size = max_size
        self.queue = [init_value for _ in range(self.max_size)]

    def update(self, value):
        self.queue = [value] + self.queue[:-1]

    def predict(self):
        return max(set(self.queue), key=self.queue.count)

    def update_predict(self, value):
        self.update(value)

        return self.predict()


class ErrorQueue:

    def __init__(self, max_size, threshold):
        self.max_size = max_size
        self.threshold = threshold
        self.queue = [False for _ in range(self.max_size)]

    def update(self, value):
        self.queue = [value] + self.queue[:-1]

    def predict(self):
        return sum(self.queue) >= self.threshold

    def update_predict(self, value):
        self.update(value)

        return self.predict()

    def refresh(self):
        self.queue = [False for _ in range(self.max_size)]


def list_cameras():
    is_working = True
    camera_port = 0
    working_cameras = []

    while is_working:
        camera = cv2.VideoCapture(camera_port)

        if not camera.isOpened():
            is_working = False
        else:
            is_reading, img = camera.read()

            if is_reading:
                working_cameras.append(camera_port)

        camera_port += 1

    return working_cameras


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):

    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image, None, None

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized, *dim


def _vector_from_landmarks(landmarks, l1, l2):
    pose_enum = mp.solutions.pose.PoseLandmark

    return np.array([landmarks[pose_enum[l1]].x - landmarks[pose_enum[l2]].x,
                     landmarks[pose_enum[l1]].y - landmarks[pose_enum[l2]].y,
                     landmarks[pose_enum[l1]].z - landmarks[pose_enum[l2]].z])


def _distance_between_landmarks(landmarks, l1, l2, look=None):
    pose_enum = mp.solutions.pose.PoseLandmark

    if look is None:
        vector = np.array([landmarks[pose_enum[l1]].x - landmarks[pose_enum[l2]].x,
                           landmarks[pose_enum[l1]].y - landmarks[pose_enum[l2]].y,
                           landmarks[pose_enum[l1]].z - landmarks[pose_enum[l2]].z])
    elif look == 'x':
        vector = np.array([landmarks[pose_enum[l1]].y - landmarks[pose_enum[l2]].y,
                           landmarks[pose_enum[l1]].z - landmarks[pose_enum[l2]].z])
    elif look == 'y':
        vector = np.array([landmarks[pose_enum[l1]].x - landmarks[pose_enum[l2]].x,
                           landmarks[pose_enum[l1]].z - landmarks[pose_enum[l2]].z])
    elif look == 'z':
        vector = np.array([landmarks[pose_enum[l1]].x - landmarks[pose_enum[l2]].x,
                           landmarks[pose_enum[l1]].y - landmarks[pose_enum[l2]].y])
    else:
        raise ValueError('"look" attribute should take one of the following values: None, "x", "y", "z"')

    return np.linalg.norm(vector)


def extract_features_from_landmarks(estimation_result):
    views = {'x': np.array([1, 0, 0]),
             'y': np.array([0, 1, 0]),
             'z': np.array([0, 0, 1])}
    pose_enum = mp.solutions.pose.PoseLandmark

    landmarks = estimation_result.pose_world_landmarks.landmark
    features = dict()

    features['left_knee_x_view_angle'] = vg.angle(_vector_from_landmarks(landmarks, 'LEFT_KNEE', 'LEFT_HIP'),
                                                  _vector_from_landmarks(landmarks, 'LEFT_KNEE', 'LEFT_ANKLE'),
                                                  look=views['x'])
    features['right_knee_x_view_angle'] = vg.angle(_vector_from_landmarks(landmarks, 'RIGHT_KNEE', 'RIGHT_HIP'),
                                                   _vector_from_landmarks(landmarks, 'RIGHT_KNEE', 'RIGHT_ANKLE'),
                                                   look=views['x'])
    features['mean_knees_x_view_angle'] = np.mean([features['left_knee_x_view_angle'],
                                                   features['right_knee_x_view_angle']])
    features['left_elbow_z_view_angle'] = vg.angle(_vector_from_landmarks(landmarks, 'LEFT_ELBOW', 'LEFT_SHOULDER'),
                                                   _vector_from_landmarks(landmarks, 'LEFT_ELBOW', 'LEFT_WRIST'),
                                                   look=views['z'])
    features['right_elbow_z_view_angle'] = vg.angle(_vector_from_landmarks(landmarks, 'RIGHT_ELBOW', 'RIGHT_SHOULDER'),
                                                    _vector_from_landmarks(landmarks, 'RIGHT_ELBOW', 'RIGHT_WRIST'),
                                                    look=views['z'])
    features['left_armpit_x_view_angle'] = vg.angle(_vector_from_landmarks(landmarks, 'LEFT_SHOULDER', 'LEFT_HIP'),
                                                    _vector_from_landmarks(landmarks, 'LEFT_SHOULDER', 'LEFT_ELBOW'),
                                                    look=views['x'])
    features['right_armpit_x_view_angle'] = vg.angle(_vector_from_landmarks(landmarks, 'RIGHT_SHOULDER', 'RIGHT_HIP'),
                                                     _vector_from_landmarks(landmarks, 'RIGHT_SHOULDER', 'RIGHT_ELBOW'),
                                                     look=views['x'])
    features['left_armpit_z_view_angle'] = vg.angle(_vector_from_landmarks(landmarks, 'LEFT_SHOULDER', 'LEFT_HIP'),
                                                    _vector_from_landmarks(landmarks, 'LEFT_SHOULDER', 'LEFT_ELBOW'),
                                                    look=views['z'])
    features['right_armpit_z_view_angle'] = vg.angle(_vector_from_landmarks(landmarks, 'RIGHT_SHOULDER', 'RIGHT_HIP'),
                                                     _vector_from_landmarks(landmarks, 'RIGHT_SHOULDER', 'RIGHT_ELBOW'),
                                                     look=views['z'])
    features['left_torso_x_view_angle'] = vg.angle(_vector_from_landmarks(landmarks, 'LEFT_HIP', 'LEFT_SHOULDER'),
                                                   _vector_from_landmarks(landmarks, 'LEFT_HIP', 'LEFT_KNEE'),
                                                   look=views['x'])
    features['right_torso_x_view_angle'] = vg.angle(_vector_from_landmarks(landmarks, 'RIGHT_HIP', 'RIGHT_SHOULDER'),
                                                    _vector_from_landmarks(landmarks, 'RIGHT_HIP', 'RIGHT_KNEE'),
                                                    look=views['x'])
    features['mean_torso_x_view_angle'] = np.mean([features['left_torso_x_view_angle'],
                                                   features['right_torso_x_view_angle']])
    features['hips_y_view_angle'] = vg.angle(_vector_from_landmarks(landmarks, 'RIGHT_HIP', 'RIGHT_KNEE'),
                                             _vector_from_landmarks(landmarks, 'LEFT_HIP', 'LEFT_KNEE'),
                                             look=views['y'])
    features['hips_z_view_angle'] = vg.angle(_vector_from_landmarks(landmarks, 'RIGHT_HIP', 'RIGHT_KNEE'),
                                             _vector_from_landmarks(landmarks, 'LEFT_HIP', 'LEFT_KNEE'),
                                             look=views['z'])
    features['left_shin_z_view_angle'] = vg.angle(_vector_from_landmarks(landmarks, 'LEFT_ANKLE', 'LEFT_KNEE'),
                                                  np.array([0, 1, 0]),
                                                  look=views['z'])
    features['right_shin_z_view_angle'] = vg.angle(_vector_from_landmarks(landmarks, 'RIGHT_ANKLE', 'RIGHT_KNEE'),
                                                   np.array([0, 1, 0]),
                                                   look=views['z'])
    features['left_shin_x_view_angle'] = vg.angle(_vector_from_landmarks(landmarks, 'LEFT_ANKLE', 'LEFT_KNEE'),
                                                  np.array([0, 1, 0]),
                                                  look=views['x'])
    features['right_shin_x_view_angle'] = vg.angle(_vector_from_landmarks(landmarks, 'RIGHT_ANKLE', 'RIGHT_KNEE'),
                                                   np.array([0, 1, 0]),
                                                   look=views['x'])
    features['feet_y_view_angle'] = vg.angle(_vector_from_landmarks(landmarks, 'RIGHT_HEEL', 'RIGHT_FOOT_INDEX'),
                                             _vector_from_landmarks(landmarks, 'LEFT_HEEL', 'LEFT_FOOT_INDEX'),
                                             look=views['y'])
    features['hips_shoulders_z_view_angle'] = vg.angle(_vector_from_landmarks(landmarks, 'LEFT_HIP', 'RIGHT_HIP'),
                                                       _vector_from_landmarks(landmarks, 'LEFT_SHOULDER',
                                                                              'RIGHT_SHOULDER'),
                                                       look=views['z'])
    features['hips_wrists_z_view_angle'] = vg.angle(_vector_from_landmarks(landmarks, 'LEFT_HIP', 'RIGHT_HIP'),
                                                    _vector_from_landmarks(landmarks, 'LEFT_WRIST', 'RIGHT_WRIST'),
                                                    look=views['z'])
    features['hips_fingers_z_view_angle'] = vg.angle(_vector_from_landmarks(landmarks, 'LEFT_HIP', 'RIGHT_HIP'),
                                                     _vector_from_landmarks(landmarks, 'LEFT_INDEX', 'RIGHT_INDEX'),
                                                     look=views['z'])
    features['left_head_shoulders_angle'] = vg.angle(
        _vector_from_landmarks(landmarks, 'LEFT_SHOULDER', 'RIGHT_SHOULDER'),
        _vector_from_landmarks(landmarks, 'LEFT_EAR', 'LEFT_EYE_OUTER'),
        look=views['y'])
    features['right_head_shoulders_angle'] = vg.angle(
        _vector_from_landmarks(landmarks, 'RIGHT_SHOULDER', 'LEFT_SHOULDER'),
        _vector_from_landmarks(landmarks, 'RIGHT_EAR', 'RIGHT_EYE_OUTER'),
        look=views['y'])
    features['head_left_right_angle'] = np.mean(
        [vg.angle(_vector_from_landmarks(landmarks, 'LEFT_SHOULDER', 'RIGHT_SHOULDER'),
                  _vector_from_landmarks(landmarks, 'LEFT_EYE', 'RIGHT_EYE'),
                  look=views['y']),
         vg.angle(_vector_from_landmarks(landmarks, 'LEFT_SHOULDER', 'RIGHT_SHOULDER'),
                  _vector_from_landmarks(landmarks, 'MOUTH_LEFT', 'MOUTH_RIGHT'),
                  look=views['y'])])
    features['head_up_down_angle'] = np.mean([vg.angle(_vector_from_landmarks(landmarks, 'LEFT_SHOULDER', 'LEFT_HIP'),
                                                       _vector_from_landmarks(landmarks, 'LEFT_EAR', 'LEFT_EYE_OUTER'),
                                                       look=views['x']),
                                              vg.angle(_vector_from_landmarks(landmarks, 'RIGHT_SHOULDER', 'RIGHT_HIP'),
                                                       _vector_from_landmarks(landmarks, 'RIGHT_EAR',
                                                                              'RIGHT_EYE_OUTER'),
                                                       look=views['x'])])
    features['left_foot_x_view_angle'] = vg.angle(_vector_from_landmarks(landmarks, 'LEFT_FOOT_INDEX', 'LEFT_HEEL'),
                                                  np.array([0, 0, -1]),
                                                  look=views['x'])
    features['right_foot_x_view_angle'] = vg.angle(_vector_from_landmarks(landmarks, 'RIGHT_FOOT_INDEX', 'RIGHT_HEEL'),
                                                   np.array([0, 0, -1]),
                                                   look=views['x'])

    features['left_torso_dist'] = _distance_between_landmarks(landmarks, 'LEFT_HIP', 'LEFT_SHOULDER')
    features['right_torso_dist'] = _distance_between_landmarks(landmarks, 'RIGHT_HIP', 'RIGHT_SHOULDER')
    features['left_thigh_dist'] = _distance_between_landmarks(landmarks, 'LEFT_HIP', 'LEFT_KNEE')
    features['right_thigh_dist'] = _distance_between_landmarks(landmarks, 'RIGHT_HIP', 'RIGHT_KNEE')\

    features['knees_x_dist'] = np.abs(landmarks[pose_enum['LEFT_KNEE']].x - landmarks[pose_enum['RIGHT_KNEE']].x)
    features['knees_z_dist'] = np.abs(landmarks[pose_enum['LEFT_KNEE']].z - landmarks[pose_enum['RIGHT_KNEE']].z)
    features['ankles_x_dist'] = np.abs(landmarks[pose_enum['LEFT_ANKLE']].x - landmarks[pose_enum['RIGHT_ANKLE']].x)
    features['ankles_z_dist'] = np.abs(landmarks[pose_enum['LEFT_ANKLE']].z - landmarks[pose_enum['RIGHT_ANKLE']].z)
    features['left_wrist_shoulder_x_dist'] = np.abs(
        landmarks[pose_enum['LEFT_WRIST']].x - landmarks[pose_enum['LEFT_SHOULDER']].x)
    features['right_wrist_shoulder_x_dist'] = np.abs(
        landmarks[pose_enum['RIGHT_WRIST']].x - landmarks[pose_enum['RIGHT_SHOULDER']].x)
    features['left_wrist_shoulder_y_dist'] = np.abs(
        landmarks[pose_enum['LEFT_WRIST']].y - landmarks[pose_enum['LEFT_SHOULDER']].y)
    features['right_wrist_shoulder_y_dist'] = np.abs(
        landmarks[pose_enum['RIGHT_WRIST']].y - landmarks[pose_enum['RIGHT_SHOULDER']].y)
    features['left_wrist_shoulder_z_dist'] = np.abs(
        landmarks[pose_enum['LEFT_WRIST']].z - landmarks[pose_enum['LEFT_SHOULDER']].z)
    features['right_wrist_shoulder_z_dist'] = np.abs(
        landmarks[pose_enum['RIGHT_WRIST']].z - landmarks[pose_enum['RIGHT_SHOULDER']].z)

    features['left_knee_hip_x_dist'] = np.abs(
        landmarks[pose_enum['LEFT_KNEE']].x - landmarks[pose_enum['LEFT_HIP']].x)
    features['right_knee_hip_x_dist'] = np.abs(
        landmarks[pose_enum['RIGHT_KNEE']].x - landmarks[pose_enum['RIGHT_HIP']].x)

    if np.abs(landmarks[pose_enum['LEFT_KNEE']].x) < np.abs(landmarks[pose_enum['LEFT_ANKLE']].x):
        features['left_shin_z_view_signed_angle'] = -features['left_shin_z_view_angle']
    else:
        features['left_shin_z_view_signed_angle'] = features['left_shin_z_view_angle']

    if np.abs(landmarks[pose_enum['RIGHT_KNEE']].x) < np.abs(landmarks[pose_enum['RIGHT_ANKLE']].x):
        features['right_shin_z_view_signed_angle'] = -features['right_shin_z_view_angle']
    else:
        features['right_shin_z_view_signed_angle'] = features['right_shin_z_view_angle']

    features['shins_z_view_signed_angle_diff'] = np.abs(features['left_shin_z_view_signed_angle'] -
                                                        features['right_shin_z_view_signed_angle'])
    features['shins_x_view_angle_diff'] = features['left_shin_x_view_angle'] - features['right_shin_x_view_angle']
    features['knees_hips_ratio_x_dist_diff'] = (features['left_knee_hip_x_dist'] - features['right_knee_hip_x_dist']) / \
                                               features['knees_x_dist']

    left_hip_y_view_angle = vg.angle(_vector_from_landmarks(landmarks, 'LEFT_HIP', 'RIGHT_HIP'),
                                     _vector_from_landmarks(landmarks, 'LEFT_HIP', 'LEFT_KNEE'),
                                     look=views['y'])
    right_hip_y_view_angle = vg.angle(_vector_from_landmarks(landmarks, 'RIGHT_HIP', 'LEFT_HIP'),
                                      _vector_from_landmarks(landmarks, 'RIGHT_HIP', 'RIGHT_KNEE'),
                                      look=views['y'])
    features['hips_y_view_angle_diff'] = left_hip_y_view_angle - right_hip_y_view_angle

    left_hip_z_view_angle = vg.angle(_vector_from_landmarks(landmarks, 'LEFT_HIP', 'RIGHT_HIP'),
                                     _vector_from_landmarks(landmarks, 'LEFT_HIP', 'LEFT_KNEE'),
                                     look=views['z'])
    right_hip_z_view_angle = vg.angle(_vector_from_landmarks(landmarks, 'RIGHT_HIP', 'LEFT_HIP'),
                                      _vector_from_landmarks(landmarks, 'RIGHT_HIP', 'RIGHT_KNEE'),
                                      look=views['z'])
    features['hips_z_view_angle_diff'] = left_hip_z_view_angle - right_hip_z_view_angle

    features['knees_ankles_x_dist_diff'] = features['knees_x_dist'] - features['ankles_x_dist']
    features['armpits_z_view_angle_diff'] = features['left_armpit_z_view_angle'] - features['right_armpit_z_view_angle']
    features['elbows_z_view_angle_diff'] = features['left_elbow_z_view_angle'] - features['right_elbow_z_view_angle']
    features['wrists_shoulders_x_dist_diff'] = features['left_wrist_shoulder_x_dist'] - features[
        'right_wrist_shoulder_x_dist']

    return features
