import cv2
import time
from kivymd.app import MDApp
from kivymd.uix.screenmanager import MDScreenManager
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from kivy.clock import Clock
from plyer import filechooser
from functools import partial
from itertools import cycle
from pose_module import PoseEstimator


Window.size = (360, 640)


class Interface(MDScreenManager):
    pass


class AICoachAPP(MDApp):
    pose_estimator = None
    exercise = None
    source = None
    recording = None
    last_frame = None
    capture = None
    capture_fps = None
    capture_event = None
    capture_reflection = False
    capture_rotation = None
    rotations = cycle([
        cv2.ROTATE_90_CLOCKWISE,
        cv2.ROTATE_180,
        cv2.ROTATE_90_COUNTERCLOCKWISE,
        None
    ])
    t1 = 0
    t2 = None

    def build(self):
        self.theme_cls.primary_palette = 'Yellow'
        self.theme_cls.primary_hue = 'A700'
        self.theme_cls.material_style = 'M3'
        self.theme_cls.theme_style = 'Dark'
        self.theme_cls.theme_style_switch_animation = True
        self.theme_cls.theme_style_switch_animation_duration = 0.3
        self.landmark_color = tuple([int(c * 255) for c in self.theme_cls.bg_dark[::-1]][1:])
        self.connection_color = tuple([int(c * 255) for c in self.theme_cls.primary_dark[::-1]][1:])

    def on_start(self):
        self.pose_estimator = PoseEstimator()

    def change_theme(self):
        if self.theme_cls.theme_style == 'Dark':
            self.theme_cls.primary_palette = 'DeepPurple'
            self.theme_cls.theme_style = 'Light'
        else:
            self.theme_cls.primary_palette = 'Yellow'
            self.theme_cls.theme_style = 'Dark'

        self.landmark_color = tuple([int(c * 255) for c in self.theme_cls.bg_dark[::-1]][1:])
        self.connection_color = tuple([int(c * 255) for c in self.theme_cls.primary_dark[::-1]][1:])

    def select_exercise(self, btn, screen):
        if btn.name == 'deadlift_btn':
            self.exercise = 'deadlift'
            # TODO: load deadlift model
            pass

        elif btn.name == 'squat_btn':
            self.exercise = 'squat'
            # TODO: load squat model
            pass

        self.root.switch_to(screen, direction='left')

    def return_from_capture(self, screen):
        if self.capture_event is not None and self.capture_event.is_triggered:
            self.capture_event.cancel()
        self.capture = None
        self.capture_reflection = False
        self.capture_rotation = None
        self.rotations = cycle([
            cv2.ROTATE_90_CLOCKWISE,
            cv2.ROTATE_180,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
            None
        ])
        self.pose_estimator = PoseEstimator()
        self.root.switch_to(screen, direction='right')

    def live_workout(self, screen):
        self.source = 'camera'
        self.capture = cv2.VideoCapture(4)
        self.capture_fps = self.capture.get(cv2.CAP_PROP_FPS) if self.capture.get(cv2.CAP_PROP_FPS) > 0 else 30
        self.capture_event = Clock.schedule_interval(partial(self.update_screen,
                                                             display='camera_display',
                                                             reps_display='camera_reps_display',
                                                             fps_display='camera_fps_display'),
                                                     1. / self.capture_fps)
        self.root.switch_to(screen, direction='left')

    def upload_recording(self, screen):
        self.recording = filechooser.open_file(
            title=f'Select {self.exercise} recording',
            filters=['*.mp4', '*.mp3', '*.mov'],
            multiple=False
        )
        if self.recording:
            self.source = 'recording'
            self.recording = self.recording[0]

            self.capture = cv2.VideoCapture(self.recording)
            self.capture_fps = self.capture.get(cv2.CAP_PROP_FPS)

            texture = self.process_frame()
            self.root.ids['recording_display'].texture = texture
            self.root.switch_to(screen, direction='left')

    def start_stop_recording(self):
        if self.capture_event is not None and self.capture_event.is_triggered:
            self.capture_event.cancel()
        else:
            self.capture_event = Clock.schedule_interval(partial(self.update_screen,
                                                                 display='recording_display',
                                                                 reps_display='recording_reps_display',
                                                                 fps_display='recording_fps_display'),
                                                         1. / self.capture_fps)

    def process_frame(self):
        ret, frame = self.capture.read()

        if ret:
            self.last_frame = frame
            if self.capture_rotation is not None:
                frame = cv2.rotate(frame, self.capture_rotation)

            self.pose_estimator.estimate_pose(frame)
            is_mistake = False
            frame = self.pose_estimator.draw_landmarks(self.landmark_color, self.connection_color, is_mistake)

            return AICoachAPP.image_to_texture(frame)

    def update_screen(self, *args, display=None, reps_display=None, fps_display=None):
        texture = self.process_frame()
        self.root.ids[fps_display].text = f'FPS: {int(Clock.get_fps())}'

        if texture:
            self.root.ids[display].texture = texture
        else:
            self.capture_event.cancel()
            self.capture = cv2.VideoCapture(self.recording)

    def rotate_video(self, display):
        self.capture_rotation = next(self.rotations)
        self.pose_estimator = PoseEstimator()

        if self.capture_event is None or not self.capture_event.is_triggered:
            frame = self.last_frame
            if self.capture_rotation is not None:
                frame = cv2.rotate(frame, self.capture_rotation)

            texture = AICoachAPP.image_to_texture(frame)
            if texture:
                display.texture = texture

    @staticmethod
    def image_to_texture(img):
        buffer = img.tobytes()
        texture = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')

        return texture


AICoachAPP().run()
