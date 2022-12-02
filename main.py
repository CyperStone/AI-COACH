import os
import cv2
from kivymd.app import MDApp
from kivymd.uix.screenmanager import MDScreenManager
from kivy.core.audio import SoundLoader
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from kivy.clock import Clock
from plyer import filechooser
from functools import partial
from itertools import cycle
from utils import list_cameras, image_resize, EventQueue, SoundsQueue
from pose_module import PoseEstimator
from exercise_modules import SquatModule
from config import app_config


Window.size = app_config.WINDOW_SIZE


class Interface(MDScreenManager):
    pass


class AICoachAPP(MDApp):
    available_cameras = None
    cameras_cycle = None
    main_camera = None
    current_camera = None
    pose_estimator = None
    exercise_module = None
    exercise = None
    source = None
    recording = None
    last_frame = None
    capture = None
    capture_fps = None
    capture_event = None
    capture_rotation = None
    event_queue = None
    current_sound = None
    sound_turn_on = None
    sounds_queue = None
    sounds = None
    rotations = cycle([
        cv2.ROTATE_90_CLOCKWISE,
        cv2.ROTATE_180,
        cv2.ROTATE_90_COUNTERCLOCKWISE,
        None
    ])

    def build(self):
        self.available_cameras = list_cameras()
        self.main_camera = app_config.DEFAULT_CAMERA_ID if app_config.DEFAULT_CAMERA_ID in self.available_cameras \
            else self.available_cameras[0]

        self.current_camera = self.main_camera
        current_camera_idx = self.available_cameras.index(self.current_camera)
        self.cameras_cycle = cycle(self.available_cameras[current_camera_idx:] +
                                   self.available_cameras[:current_camera_idx])

        self.theme_cls.primary_hue = app_config.PRIMARY_HUE
        self.theme_cls.material_style = app_config.MATERIAL_STYLE
        self.theme_cls.primary_palette = app_config.FIRST_PRIMARY_PALETTE
        self.theme_cls.theme_style = app_config.FIRST_THEME_STYLE
        self.theme_cls.theme_style_switch_animation = True
        self.theme_cls.theme_style_switch_animation_duration = app_config.ANIMATION_DURATION
        self.landmark_color = tuple([int(c * 255) for c in self.theme_cls.bg_dark[::-1]][1:])
        self.connection_color = tuple([int(c * 255) for c in self.theme_cls.primary_dark[::-1]][1:])

        self.root.ids['tutorial_text'].text = app_config.TUTORIAL_TEXT

    def on_start(self):
        self.pose_estimator = PoseEstimator()

    def change_theme(self):
        if self.theme_cls.theme_style == app_config.FIRST_THEME_STYLE:
            self.theme_cls.primary_palette = app_config.SECOND_PRIMARY_PALETTE
            self.theme_cls.theme_style = app_config.SECOND_THEME_STYLE
        else:
            self.theme_cls.primary_palette = app_config.FIRST_PRIMARY_PALETTE
            self.theme_cls.theme_style = app_config.FIRST_THEME_STYLE

        self.landmark_color = tuple([int(c * 255) for c in self.theme_cls.bg_dark[::-1]][1:])
        self.connection_color = tuple([int(c * 255) for c in self.theme_cls.primary_dark[::-1]][1:])

    def select_exercise(self, btn, screen):
        if btn.name == 'deadlift_btn':
            self.exercise = 'deadlift'

        elif btn.name == 'squat_btn':
            self.exercise = 'squat'

        self.load_sounds()
        self.root.switch_to(screen, direction='left')

    def load_sounds(self):
        self.sounds = {
            'START_STOP': SoundLoader.load(os.path.join(app_config.SOUNDS_PATH, app_config.SOUNDS['START_STOP']))
        }

        for name, file in app_config.SOUNDS['REPS'].items():
            self.sounds[name] = SoundLoader.load(os.path.join(app_config.SOUNDS_PATH, file))

        for name, file in app_config.SOUNDS['ERRORS'][self.exercise.upper()].items():
            self.sounds[name] = SoundLoader.load(os.path.join(app_config.SOUNDS_PATH, file))

    def change_sound(self, bottom_bar):
        if self.sound_turn_on:
            bottom_bar.ids.right_actions.children[0].icon = app_config.ICONS['VOLUME_ON']
            bottom_bar.ids.right_actions.children[0].tooltip_text = app_config.TOOLTIPS['VOLUME_ON']

            if self.current_sound is not None:
                self.current_sound.stop()
        else:
            bottom_bar.ids.right_actions.children[0].icon = app_config.ICONS['VOLUME_OFF']
            bottom_bar.ids.right_actions.children[0].tooltip_text = app_config.TOOLTIPS['VOLUME_OFF']

        self.sound_turn_on = not self.sound_turn_on

    def play_sound(self):
        sound_name = self.sounds_queue.get()

        if sound_name:
            self.current_sound = self.sounds[sound_name]

            if self.current_sound:
                self.current_sound.bind(on_stop=self.unlock_sounds)
                self.current_sound.play()

        self.sounds_queue.update()

    def unlock_sounds(self, *args):
        self.current_sound = None

    def change_camera(self):
        if self.capture_event is not None and self.capture_event.is_triggered:
            self.capture_event.cancel()

        Clock.schedule_once(self.live_workout, 0)

    def return_from_capture(self, screen, bottom_bar):
        if self.capture_event is not None and self.capture_event.is_triggered:
            self.capture_event.cancel()

        if self.source == 'camera':
            self.current_camera = self.main_camera
            main_camera_idx = self.available_cameras.index(self.main_camera)
            self.cameras_cycle = cycle(self.available_cameras[main_camera_idx:] +
                                       self.available_cameras[:main_camera_idx])
            bottom_bar.ids.right_actions.children[0].icon = app_config.ICONS['VOLUME_OFF']
            bottom_bar.ids.right_actions.children[0].tooltip_text = app_config.TOOLTIPS['VOLUME_OFF']

        else:
            bottom_bar.icon = app_config.ICONS['VIDEO_PLAY']
            bottom_bar.ids.right_actions.children[0].icon = app_config.ICONS['VOLUME_ON']
            bottom_bar.ids.right_actions.children[0].tooltip_text = app_config.TOOLTIPS['VOLUME_ON']

        if self.current_sound is not None:
            self.current_sound.stop()
            self.current_sound = None

        self.capture = None
        self.capture_rotation = None
        self.rotations = cycle([
            cv2.ROTATE_90_CLOCKWISE,
            cv2.ROTATE_180,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
            None
        ])

        self.exercise_module = None
        self.pose_estimator = PoseEstimator()

        self.root.switch_to(screen, direction='right')

    def live_workout(self, *args, screen=None):
        if screen:
            self.source = 'camera'
            self.sound_turn_on = True
        else:
            self.current_camera = next(self.cameras_cycle)

        self.capture = cv2.VideoCapture(self.current_camera)
        self.capture_fps = self.capture.get(cv2.CAP_PROP_FPS) if self.capture.get(cv2.CAP_PROP_FPS) > 0 else 30

        if self.exercise == 'squat':
            self.exercise_module = SquatModule(app_config.SQUAT_MODELS_PATH, self.capture_fps)

        elif self.exercise == 'deadlift':
            # TODO: load deadlift module
            pass

        self.event_queue = EventQueue()
        self.sounds_queue = SoundsQueue()
        self.capture_event = Clock.schedule_interval(partial(self.update_screen,
                                                             display='camera_display',
                                                             reps_display='camera_reps_display'),
                                                     1. / self.capture_fps)
        if screen:
            self.root.switch_to(screen, direction='left')

    def upload_recording(self, screen):
        cwd = os.getcwd()
        self.recording = filechooser.open_file(
            title=f'Select {self.exercise} recording',
            filters=['*.mp4', '*.mp3', '*.mov'],
            multiple=False
        )
        os.chdir(cwd)

        if self.recording:
            self.source = 'recording'
            self.recording = self.recording[0]

            self.capture = cv2.VideoCapture(self.recording)
            self.capture_fps = self.capture.get(cv2.CAP_PROP_FPS)

            if self.exercise == 'squat':
                self.exercise_module = SquatModule(app_config.SQUAT_MODELS_PATH, self.capture_fps)

            elif self.exercise == 'deadlift':
                # TODO: load deadlift module
                pass

            self.sound_turn_on = False
            self.event_queue = EventQueue()
            self.sounds_queue = SoundsQueue()

            texture = self.process_frame()
            self.root.ids['recording_display'].texture = texture
            self.root.ids['recording_reps_display'].text = 'REPS: 0'
            self.root.switch_to(screen, direction='left')

    def start_stop_recording(self, bottom_bar):
        if self.capture_event is not None and self.capture_event.is_triggered:
            self.capture_event.cancel()
            bottom_bar.icon = app_config.ICONS['VIDEO_PLAY']

        else:
            self.capture_event = Clock.schedule_interval(partial(self.update_screen,
                                                                 display='recording_display',
                                                                 reps_display='recording_reps_display',
                                                                 bottom_bar=bottom_bar),
                                                         1. / self.capture_fps)
            bottom_bar.icon = app_config.ICONS['VIDEO_PAUSE']

    def update_screen(self, *args, display=None, reps_display=None, bottom_bar=None):
        texture = self.process_frame(reps_display)

        if texture:
            self.root.ids[display].texture = texture
        else:
            self.capture_event.cancel()

            if bottom_bar:
                self.capture = cv2.VideoCapture(self.recording)
                self.root.ids[reps_display].text = 'REPS: 0'
                bottom_bar.icon = app_config.ICONS['VIDEO_PLAY']

            self.event_queue = EventQueue()
            self.sounds_queue = SoundsQueue()

            if self.exercise == 'squat':
                self.exercise_module = SquatModule(app_config.SQUAT_MODELS_PATH, self.capture_fps)

            if self.current_sound is not None:
                self.current_sound.stop()
                self.current_sound = None

    def process_frame(self, reps_display=None):
        ret, frame = self.capture.read()

        if ret:
            frame, _, _ = image_resize(frame, height=app_config.FRAME_HEIGHT)
            frame = cv2.flip(frame, 0)

            self.last_frame = frame
            if self.capture_rotation is not None:
                frame = cv2.rotate(frame, self.capture_rotation)

            estimation_results = self.pose_estimator.estimate_pose(frame)
            response = self.exercise_module.process_estimation_results(estimation_results)

            self.process_response(response, reps_display)

            if self.sound_turn_on and self.current_sound is None:
                self.play_sound()

            frame = self.pose_estimator.draw_landmarks(self.landmark_color,
                                                       self.connection_color,
                                                       self.event_queue.get_update())

            return AICoachAPP.image_to_texture(frame)

    def process_response(self, response, reps_display=None):
        if response['NEW_REP_EVENT']:
            self.sounds_queue.refresh(exceptions=('START_STOP',))

        if response['START_EVENT']:
            self.event_queue.append('START', self.connection_color, app_config.EVENT_DISPLAY_DURATION)

            if self.sound_turn_on:
                self.sounds_queue.append('START_STOP')

        if response['STOP_EVENT']:
            self.sounds_queue.refresh(exceptions=tuple())
            self.event_queue.append('STOP', self.connection_color, app_config.EVENT_DISPLAY_DURATION)

            if self.sound_turn_on:
                if self.current_sound is not None:
                    self.current_sound.stop()
                    self.current_sound = None

                self.sounds_queue.append('START_STOP')

        for error, value in response['ERRORS'].items():
            if value:
                self.event_queue.append(error.replace('_', ' '),
                                        app_config.ERROR_EVENT_COLOR,
                                        app_config.EVENT_DISPLAY_DURATION)

                if self.sound_turn_on:
                    self.sounds_queue.append(error)

        if response['REP_COUNT_EVENT'] and reps_display:
            reps = int(self.root.ids[reps_display].text.split(':')[1].strip()) + 1
            self.root.ids[reps_display].text = f'REPS: {reps}'

            if self.sound_turn_on:
                rep_name = f'REP_{reps}'

                if self.sounds_queue.is_free() and rep_name in self.sounds.keys():
                    self.sounds_queue.append(rep_name)

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


if __name__ == '__main__':
    AICoachAPP().run()
