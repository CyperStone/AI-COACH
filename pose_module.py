import cv2
import mediapipe.python.solutions.pose as mp_pose
import mediapipe.python.solutions.drawing_utils as mp_draw
from config import pose_module_config as pose_cnf


class PoseEstimator:

    def __init__(self,
                 model_complexity=pose_cnf.MODEL_COMPLEXITY,
                 min_detection_confidence=pose_cnf.MIN_DETECTION_CONFIDENCE,
                 min_tracking_confidence=pose_cnf.MIN_TRACKING_CONFIDENCE,
                 static_image_mode=pose_cnf.STATIC_IMAGE_MODE):

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

        return self.last_results

    def draw_landmarks(self, landmark_color, connection_color, event=None):

        if event:
            mp_draw.draw_landmarks(
                self.last_img,
                self.last_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=landmark_color,
                                    thickness=pose_cnf.LANDMARKS_THICKNESS,
                                    circle_radius=pose_cnf.LANDMARKS_RADIUS),
                mp_draw.DrawingSpec(color=event['color'],
                                    thickness=pose_cnf.LANDMARK_CONNECTIONS_THICKNESS,
                                    circle_radius=pose_cnf.LANDMARK_CONNECTIONS_RADIUS)
            )

            cv2.rectangle(
                self.last_img,
                (0, 0),
                (self.img_width, self.img_height),
                event['color'],
                pose_cnf.EVENT_BORDER_THICKNESS
            )
            cv2.putText(
                img=self.last_img,
                text=event['text'],
                org=(pose_cnf.EVENT_TEXT_POS_WIDTH, self.img_height - pose_cnf.EVENT_TEXT_POS_HEIGHT),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=pose_cnf.EVENT_TEXT_FONT_SCALE,
                color=event['color'],
                thickness=pose_cnf.EVENT_TEXT_FONT_THICKNESS,
                bottomLeftOrigin=True
            )

        else:
            mp_draw.draw_landmarks(
                self.last_img,
                self.last_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=landmark_color,
                                    thickness=pose_cnf.LANDMARKS_THICKNESS,
                                    circle_radius=pose_cnf.LANDMARKS_RADIUS),
                mp_draw.DrawingSpec(color=connection_color,
                                    thickness=pose_cnf.LANDMARK_CONNECTIONS_THICKNESS,
                                    circle_radius=pose_cnf.LANDMARK_CONNECTIONS_RADIUS)
            )

        return self.last_img
