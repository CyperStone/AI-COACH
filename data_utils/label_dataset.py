import sys
sys.path.append('..')

import os
import cv2
import msvcrt
import argparse
import numpy as np
import pandas as pd
import mediapipe as mp
import plotly.graph_objs as go
import plotly.offline
from multiprocessing import Process
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QUrl
from PyQt5 import QtWebEngineWidgets
from pose_module import PoseEstimator
from utils import image_resize, extract_features_from_landmarks


FRAMES_STEP = 5
SET_STATE_MAPPING = {'q': 1, 'w': 2, 'e': 3}
ERROR_UNCERTAINTY_MAPPING = {'1': 1, '2': 2, '3': 3}
SQUAT_ERRORS_MAPPING = {'z': 'RAISED_HEELS',
                        'x': 'KNEES_VALGUS',
                        'c': 'HEAD_POSITION',
                        'v': 'ELBOWS_BACKWARDS',
                        'b': 'ASYMMETRIC_HIPS',
                        'n': 'COLLAPSED_TORSO',
                        'm': 'ASYMMETRIC_BAR'}
DEADLIFT_ERRORS_MAPPING = {}
EXERCISE_MAPPINGS = {'squat': SQUAT_ERRORS_MAPPING,
                     'deadlift': DEADLIFT_ERRORS_MAPPING}

class PoseViewer(QtWebEngineWidgets.QWebEngineView):

    def __init__(self, fig, app_exec=True):
        self.app = QApplication.instance() if QApplication.instance() else QApplication(sys.argv)

        super().__init__()

        self.resize(1000, 600)
        self.move(800, 10)
        self.file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'temp.html'))
        plotly.offline.plot(fig, filename=self.file_path, auto_open=False)
        self.load(QUrl.fromLocalFile(self.file_path))
        self.setWindowTitle('Pose Viewer')
        self.show()

        if app_exec:
            self.app.exec_()

    def closeEvent(self, event):
        os.remove(self.file_path)


def get_pose_figure(estimation_result, angles_str, distances_str):
    X = list()
    Y = list()
    Z = list()

    camera = dict(up=dict(x=0, y=0, z=0),
                  center=dict(x=0, y=0, z=0),
                  eye=dict(x=1.5, y=-0.6, z=-1.5))

    if estimation_result.pose_world_landmarks is None:
        X.append(0)
        Y.append(0)
        Z.append(0)

        landmarks_trace = go.Scatter3d(x=X, y=Y, z=Z, mode='markers')
        fig = go.Figure(data=[landmarks_trace])

    else:
        for landmark in estimation_result.pose_world_landmarks.landmark:
            X.append(landmark.x)
            Y.append(landmark.y)
            Z.append(landmark.z)

        landmarks_trace = go.Scatter3d(x=X, y=Y, z=Z, mode='markers')
        fig = go.Figure(data=[landmarks_trace])

        for connection in mp.solutions.pose.POSE_CONNECTIONS:
            p1 = connection[0]
            p2 = connection[1]
            fig.add_trace(go.Scatter3d(x=[X[p1], X[p2]],
                                       y=[Y[p1], Y[p2]],
                                       z=[Z[p1], Z[p2]],
                                       mode='lines',
                                       line_color='black',
                                       line_width=5))

    fig.update_layout(scene=dict(xaxis_range=[-1, 1],
                                 yaxis_range=[-1, 1],
                                 zaxis_range=[-1, 1],
                                 xaxis_showticklabels=False,
                                 yaxis_showticklabels=False,
                                 zaxis_showticklabels=False,
                                 camera=camera),
                      margin=dict(r=10, l=10, b=20, t=10),
                      showlegend=False)
    fig.add_annotation(text=angles_str,
                       align='left',
                       showarrow=False,
                       xref='paper',
                       x=1,
                       y=0.9,
                       bordercolor='black',
                       borderwidth=1)
    fig.add_annotation(text=distances_str,
                       align='left',
                       showarrow=False,
                       xref='paper',
                       yref='paper',
                       x=0,
                       y=0.9,
                       bordercolor='black',
                       borderwidth=1)
    fig.update_traces(marker_size=5)

    return fig


def _get_features_info_string(features):
    angles_str = 'ANGLES'
    distances_str = 'DISTANCES'

    for feature_name, value in features.items():
        if feature_name.endswith('angle'):
            feature_name = feature_name[:-6]
            angles_str += f'<br>{feature_name}: {value:.2f}\N{DEGREE SIGN}'
        elif feature_name.endswith('dist'):
            feature_name = feature_name[:-5]
            distances_str += f'<br>{feature_name}: {value:.2f}'

    return angles_str, distances_str


def add_pose_estimation_results(df_row, estimation_result):
    pose_enum = mp.solutions.pose.PoseLandmark
    landmarks = estimation_result.pose_world_landmarks.landmark

    for enum in pose_enum:
        enum_name = enum.name
        enum_number = enum.numerator
        landmark = landmarks[enum_number]

        df_row[f'{enum_name}_X'] = landmark.x
        df_row[f'{enum_name}_Y'] = landmark.y
        df_row[f'{enum_name}_Z'] = landmark.z
        df_row[f'{enum_name}_VISIBILITY'] = landmark.visibility


def show_plot_window(figure):
    PoseViewer(figure)


def show_image_window(image):
    window_name = 'Frame Viewer'
    cv2.namedWindow(window_name)

    cv2.imshow(window_name, image)
    cv2.moveWindow(window_name, 100, 10)
    cv2.resizeWindow(window_name, image.shape[1], image.shape[0])
    cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Label mistakes on strength training videos' frames")
    parser.add_argument('exercise_type', type=str, help='Type of recorded exercise to label',
                        choices=('squat', 'deadlift'))
    parser.add_argument('camera_type', type=str, help='Type of camera the videos were recorded',
                        choices=('front', 'side', 'angle'))
    args = parser.parse_args()

    videos_path = f'..\\data\\{args.exercise_type}\\{args.camera_type}_camera'
    df_path = f'..\\data\\tabular_data\\{args.exercise_type}_{args.camera_type}.csv'

    df = pd.read_csv(df_path, sep=',')
    max_video_id = df['VIDEO_ID'].max() + 1 if not np.isnan(df['VIDEO_ID'].max()) else 0
    mapping = EXERCISE_MAPPINGS[args.exercise_type]

    video_names = sorted([file for file in os.listdir(videos_path) if int(file.split('.')[0]) >= max_video_id],
                         key=lambda file: int(file.split('.')[0]))

    for video_name in video_names:
        video_id = int(video_name.split('.')[0])
        video_path = os.path.join(videos_path, video_name)
        person_id = int(input(f'Type the ID of the person exercising on the recording number {video_id}: '))

        video = cv2.VideoCapture(video_path)
        pose_estimator = PoseEstimator()

        frame_num = 0
        end_video_event = False

        while video.isOpened() and not end_video_event:
            ret, frame = video.read()

            if ret:
                frame, width, height = image_resize(frame, height=600)
                pose_estimator.estimate_pose(frame)

                if frame_num % FRAMES_STEP == 0:
                    frame = pose_estimator.draw_landmarks((0, 0, 0), (255, 0, 0))

                    result = pose_estimator.last_results
                    pose_features = extract_features_from_landmarks(result)
                    angles_info, distances_info = _get_features_info_string(pose_features)
                    pose_figure = get_pose_figure(result, angles_info, distances_info)

                    plot_process = Process(target=show_plot_window, args=(pose_figure, ))
                    image_process = Process(target=show_image_window, args=(frame, ))

                    plot_process.start()
                    image_process.start()

                    print('Provide the type of error')
                    error_msg = msvcrt.getwch()

                    while error_msg not in mapping.keys() and error_msg != ' ' and\
                            bytes(error_msg, 'utf-8') not in [b'\x1b', b'\r']:
                        print('Incorrect error type key...')
                        error_msg = msvcrt.getwch()

                    if bytes(error_msg, 'utf-8') == b'\x1b':
                        end_video_event = True
                        print(f'Skipped {args.exercise_type} ({args.camera_type}) recording number {video_id}')

                    elif bytes(error_msg, 'utf-8') == b'\r':
                        print(f'Skipped the frame {frame_num} of {args.exercise_type} ({args.camera_type})'
                              f' recording number {video_id}')

                    else:
                        new_row = pd.DataFrame({col: [0] for col in df.columns})

                        new_row['VIDEO_ID'] = video_id
                        new_row['FRAME'] = frame_num
                        new_row['PERSON_ID'] = person_id

                        add_pose_estimation_results(new_row, result)
                        print('Adding new row to dataset...')

                        while error_msg != ' ':
                            print('Provide the uncertainty level of error')
                            uncertainty_msg = msvcrt.getwch()

                            while uncertainty_msg not in ERROR_UNCERTAINTY_MAPPING.keys():
                                print('Incorrect error uncertainty key...')
                                uncertainty_msg = msvcrt.getwch()

                            new_row[mapping[error_msg]] = ERROR_UNCERTAINTY_MAPPING[uncertainty_msg]
                            print(f'Labeled {mapping[error_msg]} error with '
                                  f'{ERROR_UNCERTAINTY_MAPPING[uncertainty_msg]} uncertainty')

                            print('\n', 'Provide the type of another error')
                            error_msg = msvcrt.getwch()
                            while error_msg not in mapping.keys() and error_msg != ' ':
                                print('Incorrect error type key...')
                                error_msg = msvcrt.getwch()

                        print('\n', 'No more errors found...')
                        print('Provide the state of exercise set')
                        state_msg = msvcrt.getwch()
                        while state_msg not in SET_STATE_MAPPING.keys() and state_msg != ' ':
                            print('Incorrect set state key...')
                            state_msg = msvcrt.getwch()

                        if state_msg != ' ':
                            new_row['SET_STATE'] = SET_STATE_MAPPING[state_msg]

                        df = pd.concat([df, new_row], ignore_index=True)

                    cv2.destroyAllWindows()

                    plot_process.terminate()
                    image_process.terminate()

                    plot_process.join()
                    image_process.join()

                frame_num += 1
            else:
                break

        df.to_csv(df_path, index=False)
        print('Saved labeling results to .csv file')
        print('_' * 50)

        print(f'Finished labeling of {args.exercise_type} ({args.camera_type}) recording number {video_id}')
        print('Push Esc key to exit or any other key to continue')
        msg = msvcrt.getwch()

        if bytes(msg, 'utf-8') == b'\x1b':
            break
