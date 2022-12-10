from types import SimpleNamespace


app_config = SimpleNamespace(
    WINDOW_SIZE=(360, 640),
    DEFAULT_CAMERA_ID=4,
    SOUNDS_PATH='./assets/sounds',
    SQUAT_MODELS_PATH='./assets/models/squat',
    PRIMARY_HUE='A700',
    MATERIAL_STYLE='M3',
    FIRST_PRIMARY_PALETTE='Yellow',
    FIRST_THEME_STYLE='Dark',
    SECOND_PRIMARY_PALETTE='DeepPurple',
    SECOND_THEME_STYLE='Light',
    ANIMATION_DURATION=0.3,     # seconds
    EVENT_DISPLAY_DURATION=25,  # frames
    ERROR_EVENT_COLOR=(0, 0, 255),
    FRAME_HEIGHT=600,
    SOUNDS={
        'START_STOP': 'start_stop.mp3',
        'ERRORS': {
            'SQUAT': {
                'TOO_FAST_ECCENTRIC_MOVE': 'squat_error_1.mp3',
                'TOO_SHALLOW': 'squat_error_2.mp3',
                'COLLAPSING_KNEES': 'squat_error_3.mp3',
                'COLLAPSED_TORSO': 'squat_error_4.mp3',
                'ELBOWS_TOO_FAR_BACK': 'squat_error_5.mp3',
                'HIP_SHIFT': 'squat_error_6.mp3',
                'ASYMMETRIC_GRIP': 'squat_error_7.mp3',
                'BAD_HEAD_POSITION': 'squat_error_8.mp3',
                'RAISING_HEELS': 'squat_error_9.mp3'
            }
        },
        'REPS': {
            'REP_1': 'rep_1.mp3',
            'REP_2': 'rep_2.mp3',
            'REP_3': 'rep_3.mp3',
            'REP_4': 'rep_4.mp3',
            'REP_5': 'rep_5.mp3',
            'REP_6': 'rep_6.mp3',
            'REP_7': 'rep_7.mp3',
            'REP_8': 'rep_8.mp3',
            'REP_9': 'rep_9.mp3',
            'REP_10': 'rep_10.mp3',
            'REP_11': 'rep_11.mp3',
            'REP_12': 'rep_12.mp3',
            'REP_13': 'rep_13.mp3',
            'REP_14': 'rep_14.mp3',
            'REP_15': 'rep_15.mp3',
            'REP_16': 'rep_16.mp3',
            'REP_17': 'rep_17.mp3',
            'REP_18': 'rep_18.mp3',
            'REP_19': 'rep_19.mp3',
            'REP_20': 'rep_20.mp3'
        }
    },
    ICONS={
        'VOLUME_ON': 'volume-off',
        'VOLUME_OFF': 'volume-high',
        'VIDEO_PLAY': 'play',
        'VIDEO_PAUSE': 'pause'
    },
    TOOLTIPS={
        'VOLUME_ON': 'Turn on sound',
        'VOLUME_OFF': 'Turn off sound'
    },
    TUTORIAL_TEXT="""
    1. Select an exercise type.
    
    2. Select a source type - live workout
        or recording from training.
    
    3. In both of the above cases:
        - camera should be set at about one
           meter height, a few meters away
           from workout spot, so that the
           whole body is visible
        - nothing should cover the face of
           the person who is performing
           exercise
        - the person who is doing the exercise
           should be facing the camera perfectly
        - video stream should be of good
           quality (lighting, resolution)
    
    4. Training assistant system tracks the
        person who is performing the exercise.
        If the system detects a technical error
        made by the user, appropriate messages
        will appear on the screen. The app will
        also play voice guidance, which advises
        the user on fixing specific error during
        the next exercise repetition.
    """
)

pose_module_config = SimpleNamespace(
    MODEL_COMPLEXITY=1,
    MIN_DETECTION_CONFIDENCE=0.5,
    MIN_TRACKING_CONFIDENCE=0.5,
    STATIC_IMAGE_MODE=False,
    LANDMARKS_THICKNESS=2,
    LANDMARK_CONNECTIONS_THICKNESS=2,
    LANDMARKS_RADIUS=2,
    LANDMARK_CONNECTIONS_RADIUS=2,
    EVENT_BORDER_THICKNESS=7,
    EVENT_TEXT_FONT_THICKNESS=2,
    EVENT_TEXT_FONT_SCALE=0.7,
    EVENT_TEXT_POS_WIDTH=15,
    EVENT_TEXT_POS_HEIGHT=30
)

squat_module_config = SimpleNamespace(
    MODEL_FILE_NAMES={
        'STATE_MODEL': 'state_model.jpkl',
        'COLLAPSING_KNEES_MODEL': 'collapsing_knees_model.jpkl',
        'COLLAPSED_TORSO_MODEL': 'collapsed_torso_model.jpkl',
        'ELBOWS_TOO_FAR_BACK_MODEL': 'elbows_too_far_back_model.jpkl',
        'HIP_SHIFT_MODEL': 'hip_shift_model.jpkl',
        'ASYMMETRIC_GRIP_MODEL': 'asymmetric_grip_model.jpkl',
        'BAD_HEAD_POSITION_MODEL': 'bad_head_position_model.jpkl',
        'RAISING_HEELS_MODEL': 'raising_heels_model.jpkl'
    },
    INITIAL_PHASE_PROB_THRESHOLD=0.05,
    BOTTOM_PHASE_PROB_THRESHOLD=0.3,
    BEFORE_AFTER_SET_PROB_THRESHOLD=0.92,
    STATE_QUEUE_SIZE=5,
    STATE_QUEUE_INIT_VALUE=3,
    STATE_INPUT_COLS=[
        'mean_knees_x_view_angle',
        'mean_torso_x_view_angle',
        'left_elbow_z_view_angle',
        'right_elbow_z_view_angle',
        'left_armpit_z_view_angle',
        'right_armpit_z_view_angle',
        'hips_y_view_angle',
        'hips_z_view_angle',
        'left_shin_z_view_angle',
        'right_shin_z_view_angle',
        'left_shin_x_view_angle',
        'right_shin_x_view_angle',
        'feet_y_view_angle',
        'hips_wrists_z_view_angle',
        'hips_fingers_z_view_angle',
        'left_foot_x_view_angle',
        'right_foot_x_view_angle',
        'knees_x_dist',
        'knees_z_dist',
        'ankles_x_dist',
        'ankles_z_dist',
        'left_wrist_shoulder_x_dist',
        'right_wrist_shoulder_x_dist',
        'left_wrist_shoulder_y_dist',
        'right_wrist_shoulder_y_dist'
    ],
    STATE_DICT={
        0: 'INITIAL_PHASE',
        1: 'MOVING_UP/DOWN',
        2: 'BOTTOM_PHASE',
        3: 'BEFORE/AFTER_SET'
    },
    STATE_DICT_REVERSED={
        'INITIAL_PHASE': 0,
        'MOVING_UP/DOWN': 1,
        'BOTTOM_PHASE': 2,
        'BEFORE/AFTER_SET': 3
    },
    ECCENTRIC_PHASE_TIME_THRESHOLD=0.35,     # seconds
    INIT_STATE_FEATURES_MIN_SAMPLES=15,
    INIT_STATE_FEATURES_FRACTION_CUT=0.1,
    INIT_STATE_FEATURES_COLS=[
        'left_torso_dist',
        'right_torso_dist',
        'left_thigh_dist',
        'right_thigh_dist',
        'knees_ankles_x_dist_diff',
        'knees_hips_ratio_x_dist_diff',
        'left_shin_z_view_signed_angle',
        'right_shin_z_view_signed_angle',
        'shins_z_view_signed_angle_diff',
        'left_foot_x_view_angle',
        'right_foot_x_view_angle'
    ],
    STATES_TO_PREDICT_ERRORS=[
        'MOVING_UP/DOWN',
        'BOTTOM_PHASE'
    ],
    COLLAPSING_KNEES_QUEUE_SIZE=6,
    COLLAPSING_KNEES_QUEUE_THRESHOLD=4,
    COLLAPSING_KNEES_PROBA_THRESHOLD=0.5,
    COLLAPSING_KNEES_AFTER_COLLAPSED_TORSO_PROBA_THRESHOLD=0.8,
    COLLAPSING_KNEES_INPUT_COLS=[
        'mean_knees_x_view_angle',
        'mean_torso_x_view_angle',
        'left_thigh_dist_init',
        'right_thigh_dist_init',
        'left_thigh_dist',
        'right_thigh_dist',
        'hips_y_view_angle',
        'hips_z_view_angle',
        'feet_y_view_angle',
        'left_shin_z_view_signed_angle_init',
        'right_shin_z_view_signed_angle_init',
        'left_shin_z_view_signed_angle',
        'right_shin_z_view_signed_angle',
        'knees_ankles_x_dist_diff_test'
    ],
    COLLAPSED_TORSO_QUEUE_SIZE=9,
    COLLAPSED_TORSO_QUEUE_THRESHOLD=6,
    COLLAPSED_TORSO_PROBA_THRESHOLD=0.5,
    COLLAPSED_TORSO_INPUT_COLS=[
        'left_knee_x_view_angle',
        'right_knee_x_view_angle',
        'left_torso_x_view_angle',
        'right_torso_x_view_angle',
        'left_torso_dist_init',
        'right_torso_dist_init',
        'left_torso_dist',
        'right_torso_dist'
    ],
    ELBOWS_TOO_FAR_BACK_BUFFER_SIZE=20,
    ELBOWS_TOO_FAR_BACK_QUEUE_SIZE=9,
    ELBOWS_TOO_FAR_BACK_QUEUE_THRESHOLD=6,
    ELBOWS_TOO_FAR_BACK_PROBA_THRESHOLD=0.5,
    ELBOWS_TOO_FAR_BACK_INPUT_COLS=[
        'mean_knees_x_view_angle',
        'mean_torso_x_view_angle',
        'left_elbow_z_view_angle',
        'right_elbow_z_view_angle',
        'left_armpit_x_view_angle',
        'right_armpit_x_view_angle',
        'left_armpit_z_view_angle',
        'right_armpit_z_view_angle',
        'left_wrist_shoulder_z_dist',
        'right_wrist_shoulder_z_dist'
    ],
    HIP_SHIFT_BUFFER_SIZE=20,
    HIP_SHIFT_QUEUE_SIZE=6,
    HIP_SHIFT_QUEUE_THRESHOLD=4,
    HIP_SHIFT_PROBA_THRESHOLD=0.5,
    HIP_SHIFT_INPUT_COLS=[
        'mean_knees_x_view_angle',
        'mean_torso_x_view_angle',
        'hips_y_view_angle_diff',
        'hips_z_view_angle_diff',
        'shins_z_view_signed_angle_diff',
        'shins_z_view_signed_angle_diff_init',
        'shins_x_view_angle_diff',
        'knees_hips_ratio_x_dist_diff',
        'knees_hips_ratio_x_dist_diff_init'
    ],
    ASYMMETRIC_GRIP_QUEUE_SIZE=15,
    ASYMMETRIC_GRIP_QUEUE_THRESHOLD=10,
    ASYMMETRIC_GRIP_PROBA_THRESHOLD=0.5,
    ASYMMETRIC_GRIP_INPUT_COLS=[
        'mean_knees_x_view_angle',
        'mean_torso_x_view_angle',
        'armpits_z_view_angle_diff',
        'elbows_z_view_angle_diff',
        'wrists_shoulders_x_dist_diff',
        'hips_shoulders_z_view_angle',
        'hips_wrists_z_view_angle',
        'hips_fingers_z_view_angle'
    ],
    BAD_HEAD_POSITION_QUEUE_SIZE=6,
    BAD_HEAD_POSITION_QUEUE_THRESHOLD=4,
    BAD_HEAD_POSITION_PROBA_THRESHOLD=0.5,
    BAD_HEAD_POSITION_INPUT_COLS=[
        'mean_knees_x_view_angle',
        'mean_torso_x_view_angle',
        'left_head_shoulders_angle',
        'right_head_shoulders_angle',
        'head_left_right_angle',
        'head_up_down_angle'
    ],
    RAISING_HEELS_QUEUE_SIZE=6,
    RAISING_HEELS_QUEUE_THRESHOLD=4,
    RAISING_HEELS_PROBA_THRESHOLD=0.8,
    RAISING_HEELS_INPUT_COLS=[
        'mean_knees_x_view_angle',
        'mean_torso_x_view_angle',
        'left_shin_x_view_angle',
        'right_shin_x_view_angle',
        'left_foot_x_view_angle',
        'right_foot_x_view_angle',
        'left_foot_x_view_angle_init',
        'right_foot_x_view_angle_init',
        'feet_y_view_angle',
        'ankles_x_dist'
    ]
)
