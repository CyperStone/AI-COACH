import os
import joblib
import pandas as pd
from scipy.stats import trim_mean
from utils import StateQueue, ErrorQueue, extract_features_from_landmarks
from config import squat_module_config as squat_cnf


class SquatModule:

    def __init__(self, models_path, fps):
        self.fps = fps
        self.start = False
        self.stop = False

        self.state_model = joblib.load(os.path.join(models_path, squat_cnf.MODEL_FILE_NAMES['STATE_MODEL']))
        self.collapsing_knees_model = joblib.load(os.path.join(models_path,
                                                               squat_cnf.MODEL_FILE_NAMES['COLLAPSING_KNEES_MODEL']))
        self.collapsed_torso_model = joblib.load(os.path.join(models_path,
                                                              squat_cnf.MODEL_FILE_NAMES['COLLAPSED_TORSO_MODEL']))
        self.elbows_too_far_back_model = joblib.load(os.path.join(models_path,
                                                                  squat_cnf.MODEL_FILE_NAMES[
                                                                      'ELBOWS_TOO_FAR_BACK_MODEL']))
        self.hip_shift_model = joblib.load(os.path.join(models_path,
                                                        squat_cnf.MODEL_FILE_NAMES['HIP_SHIFT_MODEL']))
        self.asymmetric_grip_model = joblib.load(os.path.join(models_path,
                                                              squat_cnf.MODEL_FILE_NAMES['ASYMMETRIC_GRIP_MODEL']))
        self.bad_head_position_model = joblib.load(os.path.join(models_path,
                                                                squat_cnf.MODEL_FILE_NAMES['BAD_HEAD_POSITION_MODEL']))
        self.raising_heels_model = joblib.load(os.path.join(models_path,
                                                            squat_cnf.MODEL_FILE_NAMES['RAISING_HEELS_MODEL']))

        self.errors = {
            'TOO_FAST_ECCENTRIC_MOVE': False,
            'TOO_SHALLOW': False,
            'COLLAPSING_KNEES': False,
            'COLLAPSED_TORSO': False,
            'ELBOWS_TOO_FAR_BACK': False,
            'HIP_SHIFT': False,
            'ASYMMETRIC_GRIP': False,
            'BAD_HEAD_POSITION': False,
            'RAISING_HEELS': False
        }

        self.error_queues = {
            'COLLAPSING_KNEES': ErrorQueue(squat_cnf.COLLAPSING_KNEES_QUEUE_SIZE,
                                           squat_cnf.COLLAPSING_KNEES_QUEUE_THRESHOLD),
            'COLLAPSED_TORSO': ErrorQueue(squat_cnf.COLLAPSED_TORSO_QUEUE_SIZE,
                                          squat_cnf.COLLAPSED_TORSO_QUEUE_THRESHOLD),
            'ELBOWS_TOO_FAR_BACK': ErrorQueue(squat_cnf.ELBOWS_TOO_FAR_BACK_QUEUE_SIZE,
                                              squat_cnf.ELBOWS_TOO_FAR_BACK_QUEUE_THRESHOLD),
            'HIP_SHIFT': ErrorQueue(squat_cnf.HIP_SHIFT_QUEUE_SIZE,
                                    squat_cnf.HIP_SHIFT_QUEUE_THRESHOLD),
            'ASYMMETRIC_GRIP': ErrorQueue(squat_cnf.ASYMMETRIC_GRIP_QUEUE_SIZE,
                                          squat_cnf.ASYMMETRIC_GRIP_QUEUE_THRESHOLD),
            'BAD_HEAD_POSITION': ErrorQueue(squat_cnf.BAD_HEAD_POSITION_QUEUE_SIZE,
                                            squat_cnf.BAD_HEAD_POSITION_QUEUE_THRESHOLD),
            'RAISING_HEELS': ErrorQueue(squat_cnf.RAISING_HEELS_QUEUE_SIZE,
                                        squat_cnf.RAISING_HEELS_QUEUE_THRESHOLD)
        }

        self.state_queue = StateQueue(squat_cnf.STATE_QUEUE_SIZE, squat_cnf.STATE_QUEUE_INIT_VALUE)
        self.last_state = None
        self.current_state = squat_cnf.STATE_QUEUE_INIT_VALUE

        self.was_bottom_state = False
        self.frames_counter = None
        self.elbows_too_far_back_buffer_counter = None
        self.hip_shift_buffer_counter = None

        self.init_state_features_array = None
        self.init_state_features = None

    def process_estimation_results(self, estimation_results):
        response = {
            'START_EVENT': False,
            'STOP_EVENT': False,
            'REP_COUNT_EVENT': False,
            'NEW_REP_EVENT': False,
            'ERRORS': {
                'TOO_FAST_ECCENTRIC_MOVE': False,
                'TOO_SHALLOW': False,
                'COLLAPSING_KNEES': False,
                'COLLAPSED_TORSO': False,
                'ELBOWS_TOO_FAR_BACK': False,
                'HIP_SHIFT': False,
                'ASYMMETRIC_GRIP': False,
                'BAD_HEAD_POSITION': False,
                'RAISING_HEELS': False
            }
        }

        X = None

        if self.frames_counter:
            self.frames_counter += 1

        if self.elbows_too_far_back_buffer_counter:
            self.elbows_too_far_back_buffer_counter += 1

            if self.elbows_too_far_back_buffer_counter >= squat_cnf.ELBOWS_TOO_FAR_BACK_BUFFER_SIZE:
                self.errors['ELBOWS_TOO_FAR_BACK'] = True
                response['ERRORS']['ELBOWS_TOO_FAR_BACK'] = True
                self.elbows_too_far_back_buffer_counter = None

        if self.hip_shift_buffer_counter:
            self.hip_shift_buffer_counter += 1

            if self.hip_shift_buffer_counter >= squat_cnf.HIP_SHIFT_BUFFER_SIZE:
                self.errors['HIP_SHIFT'] = True
                response['ERRORS']['HIP_SHIFT'] = True
                self.hip_shift_buffer_counter = None

        if estimation_results.pose_world_landmarks is None:
            state_pred = squat_cnf.STATE_DICT_REVERSED['BEFORE/AFTER_SET']

        else:
            X = pd.DataFrame({k: [v] for k, v in extract_features_from_landmarks(estimation_results).items()})

            X_state = X.loc[:, squat_cnf.STATE_INPUT_COLS]
            pred_proba = self.state_model.predict_proba(X_state)[0]

            if pred_proba[squat_cnf.STATE_DICT_REVERSED['INITIAL_PHASE']] > squat_cnf.INITIAL_PHASE_PROB_THRESHOLD:

                state_pred = squat_cnf.STATE_DICT_REVERSED['INITIAL_PHASE']

            elif pred_proba[squat_cnf.STATE_DICT_REVERSED['BOTTOM_PHASE']] > \
                    squat_cnf.BOTTOM_PHASE_PROB_THRESHOLD:

                state_pred = squat_cnf.STATE_DICT_REVERSED['BOTTOM_PHASE']

            elif pred_proba[squat_cnf.STATE_DICT_REVERSED['BEFORE/AFTER_SET']] > \
                    squat_cnf.BEFORE_AFTER_SET_PROB_THRESHOLD:

                state_pred = squat_cnf.STATE_DICT_REVERSED['BEFORE/AFTER_SET']

            else:
                state_pred = squat_cnf.STATE_DICT_REVERSED['MOVING_UP/DOWN']

        state = self.state_queue.update_predict(state_pred)

        if not self.start and squat_cnf.STATE_DICT[state] in ('MOVING_UP/DOWN', 'BOTTOM_PHASE'):
            self.start = True
            self.stop = False
            response['START_EVENT'] = True

        elif self.start and not self.stop and squat_cnf.STATE_DICT[state] == 'BEFORE/AFTER_SET':
            self.stop = True
            self.start = False
            response['STOP_EVENT'] = True

        if state != self.current_state:
            self.last_state = self.current_state
            self.current_state = state

            if self.start:
                if squat_cnf.STATE_DICT[self.current_state] == 'INITIAL_PHASE' and \
                        squat_cnf.STATE_DICT[self.last_state] == 'MOVING_UP/DOWN':

                    response['REP_COUNT_EVENT'] = True

                    if not self.was_bottom_state:
                        response['ERRORS']['TOO_SHALLOW'] = True

                elif squat_cnf.STATE_DICT[self.current_state] == 'MOVING_UP/DOWN' and \
                        squat_cnf.STATE_DICT[self.last_state] == 'INITIAL_PHASE':

                    response['NEW_REP_EVENT'] = True

                    self.frames_counter = 1
                    self.elbows_too_far_back_buffer_counter = None
                    self.hip_shift_buffer_counter = None

                    self.was_bottom_state = False
                    self.errors = {error: False for error in self.errors.keys()}

                    for queue in self.error_queues.values():
                        queue.refresh()

                elif squat_cnf.STATE_DICT[self.current_state] == 'BOTTOM_PHASE':
                    self.was_bottom_state = True

                    if self.frames_counter and \
                            (self.frames_counter / self.fps) < squat_cnf.ECCENTRIC_PHASE_TIME_THRESHOLD:

                        response['ERRORS']['TOO_FAST_ECCENTRIC_MOVE'] = True

                    self.frames_counter = None

        if X is not None:

            if self.init_state_features is None and \
                    squat_cnf.STATE_DICT[self.current_state] == 'INITIAL_PHASE':

                X_init_state_features = X.loc[:, squat_cnf.INIT_STATE_FEATURES_COLS]

                if self.init_state_features_array is None:
                    self.init_state_features_array = X_init_state_features
                else:
                    self.init_state_features_array = pd.concat([self.init_state_features_array, X_init_state_features])

                if len(self.init_state_features_array) >= squat_cnf.INIT_STATE_FEATURES_MIN_SAMPLES:

                    self.init_state_features = pd.DataFrame(self.init_state_features_array.apply(
                        lambda col: trim_mean(col, squat_cnf.INIT_STATE_FEATURES_FRACTION_CUT), axis=0)).T
                    self.init_state_features.columns = [col + '_init' for col in self.init_state_features.columns]
                    self.init_state_features_array = None

            if squat_cnf.STATE_DICT[self.current_state] in squat_cnf.STATES_TO_PREDICT_ERRORS:

                if self.init_state_features is not None:
                    X_init_state_features = pd.concat([X, self.init_state_features], axis=1)

                    X_collapsed_torso = X_init_state_features.loc[:, squat_cnf.COLLAPSED_TORSO_INPUT_COLS]
                    collapsed_torso_proba = self.collapsed_torso_model.predict_proba(X_collapsed_torso)[0][1]

                    if not self.errors['COLLAPSED_TORSO']:
                        collapsed_torso_pred = collapsed_torso_proba > squat_cnf.COLLAPSED_TORSO_PROBA_THRESHOLD

                        if self.error_queues['COLLAPSED_TORSO'].update_predict(collapsed_torso_pred):
                            self.errors['COLLAPSED_TORSO'] = True
                            response['ERRORS']['COLLAPSED_TORSO'] = True
                            self.elbows_too_far_back_buffer_counter = None

                    X_collapsing_knees = X_init_state_features.loc[:, squat_cnf.COLLAPSING_KNEES_INPUT_COLS]
                    collapsing_knees_proba = self.collapsing_knees_model.predict_proba(X_collapsing_knees)[0][1]

                    if not self.errors['COLLAPSING_KNEES']:

                        if self.errors['COLLAPSED_TORSO']:
                            collapsing_knees_pred = collapsing_knees_proba > \
                                                    squat_cnf.COLLAPSING_KNEES_AFTER_COLLAPSED_TORSO_PROBA_THRESHOLD
                        else:
                            collapsing_knees_pred = collapsing_knees_proba > squat_cnf.COLLAPSING_KNEES_PROBA_THRESHOLD

                        if self.error_queues['COLLAPSING_KNEES'].update_predict(collapsing_knees_pred):
                            self.errors['COLLAPSING_KNEES'] = True
                            response['ERRORS']['COLLAPSING_KNEES'] = True
                            self.hip_shift_buffer_counter = None

                    X_hip_shift = X_init_state_features.loc[:, squat_cnf.HIP_SHIFT_INPUT_COLS]
                    hip_shift_proba = self.hip_shift_model.predict_proba(X_hip_shift)[0][1]

                    if not self.errors['HIP_SHIFT'] and self.was_bottom_state and not self.errors['COLLAPSING_KNEES'] \
                            and self.hip_shift_buffer_counter is None:
                        hip_shift_pred = hip_shift_proba > squat_cnf.HIP_SHIFT_PROBA_THRESHOLD

                        if self.error_queues['HIP_SHIFT'].update_predict(hip_shift_pred):
                            self.hip_shift_buffer_counter = 1

                    X_raising_heels = X_init_state_features.loc[:, squat_cnf.RAISING_HEELS_INPUT_COLS]
                    raising_heels_proba = self.raising_heels_model.predict_proba(X_raising_heels)[0][1]
                    if not self.errors['RAISING_HEELS']:

                        raising_heels_pred = raising_heels_proba > squat_cnf.RAISING_HEELS_PROBA_THRESHOLD

                        if self.error_queues['RAISING_HEELS'].update_predict(raising_heels_pred):
                            self.errors['RAISING_HEELS'] = True
                            response['ERRORS']['RAISING_HEELS'] = True

                X_elbows_too_far_back = X.loc[:, squat_cnf.ELBOWS_TOO_FAR_BACK_INPUT_COLS]
                elbows_too_far_back_proba = self.elbows_too_far_back_model.predict_proba(X_elbows_too_far_back)[0][1]

                if not self.errors['ELBOWS_TOO_FAR_BACK'] and self.elbows_too_far_back_buffer_counter is None and \
                        not self.errors['COLLAPSED_TORSO']:

                    elbows_too_far_back_pred = elbows_too_far_back_proba > squat_cnf.ELBOWS_TOO_FAR_BACK_PROBA_THRESHOLD

                    if self.error_queues['ELBOWS_TOO_FAR_BACK'].update_predict(elbows_too_far_back_pred):
                        self.elbows_too_far_back_buffer_counter = 1

                X_asymmetric_grip = X.loc[:, squat_cnf.ASYMMETRIC_GRIP_INPUT_COLS]
                asymmetric_grip_proba = self.asymmetric_grip_model.predict_proba(X_asymmetric_grip)[0][1]

                if not self.errors['ASYMMETRIC_GRIP']:

                    asymmetric_grip_pred = asymmetric_grip_proba > squat_cnf.ASYMMETRIC_GRIP_PROBA_THRESHOLD

                    if self.error_queues['ASYMMETRIC_GRIP'].update_predict(asymmetric_grip_pred):
                        self.errors['ASYMMETRIC_GRIP'] = True
                        response['ERRORS']['ASYMMETRIC_GRIP'] = True

                X_bad_head_position = X.loc[:, squat_cnf.BAD_HEAD_POSITION_INPUT_COLS]
                bad_head_position_proba = self.bad_head_position_model.predict_proba(X_bad_head_position)[0][1]

                if not self.errors['BAD_HEAD_POSITION']:

                    bad_head_position_pred = bad_head_position_proba > squat_cnf.BAD_HEAD_POSITION_PROBA_THRESHOLD

                    if self.error_queues['BAD_HEAD_POSITION'].update_predict(bad_head_position_pred):
                        self.errors['BAD_HEAD_POSITION'] = True
                        response['ERRORS']['BAD_HEAD_POSITION'] = True

        return response
