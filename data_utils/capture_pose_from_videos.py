import os
import time
import cv2
import msvcrt
import argparse
import mediapipe as mp
from multiprocessing import Process, Event


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized, *dim


def capture_pose_from_video(video_path,
                            screen_placement,
                            start_event,
                            stop_event,
                            result_height=500,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5,
                            model_complexity=1):

    window_title = video_path.split('\\')[-1].split('.')[0]

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    start_event.wait()
    video = cv2.VideoCapture(video_path)
    t1 = 0

    with mp_pose.Pose(min_detection_confidence=min_detection_confidence,
                      min_tracking_confidence=min_tracking_confidence,
                      model_complexity=model_complexity) as pose:
        while video.isOpened() and not stop_event.is_set():
            ret, frame = video.read()

            if ret:
                t2 = time.time()
                fps = int(1 / (t2 - t1))
                t1 = t2

                frame.flags.writeable = False
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(img)

                img.flags.writeable = True
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=5, circle_radius=5),
                                          mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=5, circle_radius=5))

                img, new_width, new_height = image_resize(img, height=result_height)

                cv2.putText(img, f'FPS: {fps}', (new_width - 70, new_height - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imshow(f'POSE| {window_title}', img)
                cv2.moveWindow(f'POSE| {window_title}', *screen_placement)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            else:
                break

    video.release()
    cv2.destroyWindow(f'POSE| {window_title}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Display captured poses from training videos.')
    parser.add_argument('exercise_type', type=str, help='Type of recorded exercise to display',
                        choices=('squat', 'deadlift'))
    parser.add_argument('-n', '--video_num', type=int, help='Which number of the video')
    args = parser.parse_args()

    data_path = f'..\\data\\{args.exercise_type}'
    max_video_num = max([int(filename.split('_')[0]) for filename in os.listdir(data_path) \
                         if os.path.isfile(os.path.join(data_path, filename))])
    video_num = args.video_num if args.video_num is not None else max_video_num
    placement_dict = {'front': (150, 10), 'side': (600, 10), 'angle': (1050, 10)}

    start_event = Event()
    stop_event = Event()
    front_process = Process(target=capture_pose_from_video,
                            args=(os.path.join(data_path, f'{video_num}_{args.exercise_type}_front.mp4'),
                                  placement_dict['front'], start_event, stop_event))
    side_process = Process(target=capture_pose_from_video,
                           args=(os.path.join(data_path, f'{video_num}_{args.exercise_type}_side.mp4'),
                                 placement_dict['side'], start_event, stop_event))
    angle_process = Process(target=capture_pose_from_video,
                            args=(os.path.join(data_path, f'{video_num}_{args.exercise_type}_angle.mp4'),
                                  placement_dict['angle'], start_event, stop_event))

    front_process.start()
    side_process.start()
    angle_process.start()

    while not stop_event.is_set():
        msg = msvcrt.getwch()
        if msg == 'z':
            start_event.set()
        if msg == 'x':
            stop_event.set()

    front_process.join()
    side_process.join()
    angle_process.join()
