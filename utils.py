import vg
import cv2
import numpy as np
import mediapipe as mp


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
    views = {'all': None,
             'x': np.array([1, 0, 0]),
             'y': np.array([0, 1, 0]),
             'z': np.array([0, 0, 1])}
    pose_enum = mp.solutions.pose.PoseLandmark

    features = dict()
    features['left_knee_angle'] = 0
    features['right_knee_angle'] = 0
    features['left_elbow_angle'] = 0
    features['right_elbow_angle'] = 0
    features['left_armpit_side_angle'] = 0
    features['right_armpit_side_angle'] = 0
    features['left_armpit_front_angle'] = 0
    features['right_armpit_front_angle'] = 0
    features['left_torso_angle'] = 0
    features['right_torso_angle'] = 0
    features['left_hip_angle'] = 0
    features['right_hip_angle'] = 0
    features['hips_angle'] = 0
    features['hips_z_view_angle'] = 0
    features['feet_angle'] = 0
    features['hips_shoulders_angle'] = 0
    features['hips_wrists_angle'] = 0
    features['hips_fingers_angle'] = 0
    features['hips_eyes_angle'] = 0
    features['head_up_down_angle'] = 0
    features['left_foot_angle'] = 0
    features['right_foot_angle'] = 0

    features['shoulders_dist'] = 0
    features['hips_dist'] = 0
    features['knees_dist'] = 0
    features['ankles_dist'] = 0
    features['wrists_dist'] = 0
    features['left_torso_dist'] = 0
    features['right_torso_dist'] = 0
    features['left_thigh_dist'] = 0
    features['right_thigh_dist'] = 0
    features['left_elbow_hip_dist'] = 0
    features['right_elbow_hip_dist'] = 0
    features['left_wrist_shoulder_dist'] = 0
    features['right_wrist_shoulder_dist'] = 0
    features['left_knee_hip_x_dist'] = 0
    features['right_knee_hip_x_dist'] = 0

    if estimation_result.pose_world_landmarks is None:
        return features

    landmarks = estimation_result.pose_world_landmarks.landmark

    features['left_knee_angle'] = vg.angle(_vector_from_landmarks(landmarks, 'LEFT_KNEE', 'LEFT_HIP'),
                                           _vector_from_landmarks(landmarks, 'LEFT_KNEE', 'LEFT_ANKLE'),
                                           look=views['x'])
    features['right_knee_angle'] = vg.angle(_vector_from_landmarks(landmarks, 'RIGHT_KNEE', 'RIGHT_HIP'),
                                            _vector_from_landmarks(landmarks, 'RIGHT_KNEE', 'RIGHT_ANKLE'),
                                            look=views['x'])
    features['left_elbow_angle'] = vg.angle(_vector_from_landmarks(landmarks, 'LEFT_ELBOW', 'LEFT_SHOULDER'),
                                            _vector_from_landmarks(landmarks, 'LEFT_ELBOW', 'LEFT_WRIST'),
                                            look=views['z'])
    features['right_elbow_angle'] = vg.angle(_vector_from_landmarks(landmarks, 'RIGHT_ELBOW', 'RIGHT_SHOULDER'),
                                             _vector_from_landmarks(landmarks, 'RIGHT_ELBOW', 'RIGHT_WRIST'),
                                             look=views['z'])
    features['left_armpit_side_angle'] = vg.signed_angle(_vector_from_landmarks(landmarks, 'LEFT_SHOULDER', 'LEFT_HIP'),
                                                         _vector_from_landmarks(landmarks, 'LEFT_SHOULDER', 'LEFT_ELBOW'),
                                                         look=views['x'])
    features['right_armpit_side_angle'] = vg.signed_angle(_vector_from_landmarks(landmarks, 'RIGHT_SHOULDER', 'RIGHT_HIP'),
                                                          _vector_from_landmarks(landmarks, 'RIGHT_SHOULDER', 'RIGHT_ELBOW'),
                                                          look=views['x'])
    features['left_armpit_front_angle'] = vg.angle(_vector_from_landmarks(landmarks, 'LEFT_SHOULDER', 'LEFT_HIP'),
                                                   _vector_from_landmarks(landmarks, 'LEFT_SHOULDER', 'LEFT_ELBOW'),
                                                   look=views['z'])
    features['right_armpit_front_angle'] = vg.angle(_vector_from_landmarks(landmarks, 'RIGHT_SHOULDER', 'RIGHT_HIP'),
                                                    _vector_from_landmarks(landmarks, 'RIGHT_SHOULDER', 'RIGHT_ELBOW'),
                                                    look=views['z'])
    features['left_torso_angle'] = vg.signed_angle(_vector_from_landmarks(landmarks, 'LEFT_HIP', 'LEFT_SHOULDER'),
                                                   _vector_from_landmarks(landmarks, 'LEFT_HIP', 'LEFT_KNEE'),
                                                   look=views['x'])
    features['right_torso_angle'] = vg.signed_angle(_vector_from_landmarks(landmarks, 'RIGHT_HIP', 'RIGHT_SHOULDER'),
                                                    _vector_from_landmarks(landmarks, 'RIGHT_HIP', 'RIGHT_KNEE'),
                                                    look=views['x'])
    features['left_hip_angle'] = vg.angle(_vector_from_landmarks(landmarks, 'LEFT_HIP', 'RIGHT_HIP'),
                                          _vector_from_landmarks(landmarks, 'LEFT_HIP', 'LEFT_KNEE'),
                                          look=views['all'])
    features['right_hip_angle'] = vg.angle(_vector_from_landmarks(landmarks, 'RIGHT_HIP', 'LEFT_HIP'),
                                           _vector_from_landmarks(landmarks, 'RIGHT_HIP', 'RIGHT_KNEE'),
                                           look=views['all'])
    features['hips_angle'] = vg.angle(_vector_from_landmarks(landmarks, 'RIGHT_HIP', 'RIGHT_KNEE'),
                                      _vector_from_landmarks(landmarks, 'LEFT_HIP', 'LEFT_KNEE'),
                                      look=views['all'])
    features['hips_z_view_angle'] = vg.signed_angle(_vector_from_landmarks(landmarks, 'RIGHT_HIP', 'RIGHT_KNEE'),
                                                    _vector_from_landmarks(landmarks, 'LEFT_HIP', 'LEFT_KNEE'),
                                                    look=views['z'])
    features['feet_angle'] = vg.angle(_vector_from_landmarks(landmarks, 'RIGHT_HEEL', 'RIGHT_FOOT_INDEX'),
                                      _vector_from_landmarks(landmarks, 'LEFT_HEEL', 'LEFT_FOOT_INDEX'),
                                      look=views['all'])
    features['hips_shoulders_angle'] = vg.angle(_vector_from_landmarks(landmarks, 'LEFT_HIP', 'RIGHT_HIP'),
                                                _vector_from_landmarks(landmarks, 'LEFT_SHOULDER', 'RIGHT_SHOULDER'),
                                                look=views['z'])
    features['hips_wrists_angle'] = vg.angle(_vector_from_landmarks(landmarks, 'LEFT_HIP', 'RIGHT_HIP'),
                                             _vector_from_landmarks(landmarks, 'LEFT_WRIST', 'RIGHT_WRIST'),
                                             look=views['z'])
    features['hips_fingers_angle'] = vg.angle(_vector_from_landmarks(landmarks, 'LEFT_HIP', 'RIGHT_HIP'),
                                              _vector_from_landmarks(landmarks, 'LEFT_INDEX', 'RIGHT_INDEX'),
                                              look=views['z'])
    features['hips_eyes_angle'] = vg.angle(_vector_from_landmarks(landmarks, 'LEFT_HIP', 'RIGHT_HIP'),
                                           _vector_from_landmarks(landmarks, 'LEFT_EYE', 'RIGHT_EYE'),
                                           look=views['y'])
    features['head_up_down_angle'] = vg.angle(_vector_from_landmarks(landmarks, 'LEFT_HIP', 'LEFT_SHOULDER'),
                                              _vector_from_landmarks(landmarks, 'LEFT_EAR', 'LEFT_EYE_OUTER'),
                                              look=views['x'])
    features['left_foot_angle'] = vg.angle(_vector_from_landmarks(landmarks, 'LEFT_FOOT_INDEX', 'LEFT_HEEL'),
                                           np.array([0, 0, -1]),
                                           look=views['x'])
    features['right_foot_angle'] = vg.angle(_vector_from_landmarks(landmarks, 'RIGHT_FOOT_INDEX', 'RIGHT_HEEL'),
                                            np.array([0, 0, -1]),
                                            look=views['x'])

    features['shoulders_dist'] = _distance_between_landmarks(landmarks, 'LEFT_SHOULDER', 'RIGHT_SHOULDER')
    features['hips_dist'] = _distance_between_landmarks(landmarks, 'LEFT_HIP', 'RIGHT_HIP')
    features['knees_dist'] = _distance_between_landmarks(landmarks, 'LEFT_KNEE', 'RIGHT_KNEE')
    features['ankles_dist'] = _distance_between_landmarks(landmarks, 'LEFT_ANKLE', 'RIGHT_ANKLE')
    features['wrists_dist'] = _distance_between_landmarks(landmarks, 'LEFT_WRIST', 'RIGHT_WRIST')
    features['left_torso_dist'] = _distance_between_landmarks(landmarks, 'LEFT_HIP', 'LEFT_SHOULDER')
    features['right_torso_dist'] = _distance_between_landmarks(landmarks, 'RIGHT_HIP', 'RIGHT_SHOULDER')
    features['left_thigh_dist'] = _distance_between_landmarks(landmarks, 'LEFT_HIP', 'LEFT_KNEE')
    features['right_thigh_dist'] = _distance_between_landmarks(landmarks, 'RIGHT_HIP', 'RIGHT_KNEE')
    features['left_elbow_hip_dist'] = _distance_between_landmarks(landmarks, 'LEFT_ELBOW', 'LEFT_HIP')
    features['right_elbow_hip_dist'] = _distance_between_landmarks(landmarks, 'RIGHT_ELBOW', 'RIGHT_HIP')
    features['left_wrist_shoulder_dist'] = _distance_between_landmarks(landmarks, 'LEFT_WRIST', 'LEFT_SHOULDER',
                                                                       look='z')
    features['right_wrist_shoulder_dist'] = _distance_between_landmarks(landmarks, 'RIGHT_WRIST', 'RIGHT_SHOULDER',
                                                                        look='z')
    features['left_knee_hip_x_dist'] = np.abs(landmarks[pose_enum['LEFT_KNEE']].x - landmarks[pose_enum['LEFT_HIP']].x)
    features['right_knee_hip_x_dist'] = np.abs(landmarks[pose_enum['RIGHT_KNEE']].x - landmarks[pose_enum['RIGHT_HIP']].x)

    return features
