import cv2
import mediapipe.python.solutions.pose as mp_pose
import mediapipe.python.solutions.drawing_utils as mp_draw


class PoseEstimator:

    def __init__(self,
                 model_complexity=1,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5,
                 static_image_mode=False):

        self.pose = mp_pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            static_image_mode=static_image_mode
        )
        self.img_width = None
        self.img_height = None
        self.last_img = None
        self.last_results = None

    def estimate_pose(self, img):
        self.last_img = img.copy()
        if self.img_width is None:
            self.img_height, self.img_width = self.last_img.shape[:-1]

        self.last_img.flags.writeable = False
        self.last_results = self.pose.process(cv2.cvtColor(self.last_img, cv2.COLOR_BGR2RGB))
        self.last_img.flags.writeable = True

    def draw_landmarks(self, landmark_color, connection_color, is_mistake=None):
        if is_mistake:
            mp_draw.draw_landmarks(self.last_img, self.last_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                   mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                                   mp_draw.DrawingSpec(color=(0, 0, 255), thickness=4, circle_radius=4))
            cv2.rectangle(self.last_img, (0, 0), (self.img_width, self.img_height), (0, 0, 255), 5)

        else:
            mp_draw.draw_landmarks(self.last_img, self.last_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                   mp_draw.DrawingSpec(color=landmark_color, thickness=2, circle_radius=2),
                                   mp_draw.DrawingSpec(color=connection_color, thickness=4, circle_radius=4))

        return self.last_img
