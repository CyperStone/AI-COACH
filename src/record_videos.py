import os
import cv2
import argparse
import msvcrt
import threading


parser = argparse.ArgumentParser(description='Record training videos from multiple cameras and save them to files.')
parser.add_argument('exercise_type', type=str, help='Type of exercise to record', choices=('squat', 'deadlift'))
parser.add_argument('-c1', '--front_cam_id', type=int, required=True, help='ID of the front camera')
parser.add_argument('-c2', '--side_cam_id', type=int, required=True, help='ID of the side camera')
parser.add_argument('-c3', '--angle_cam_id', type=int, required=True, help=u'ID of the 45\N{DEGREE SIGN} camera')
args = parser.parse_args()

data_path = f'..\\data\\{args.exercise_type}'
max_video_num = max([int(filename.split('_')[0]) for filename in os.listdir(data_path)])
placement_dict = {'front': (150, 10), 'side': (600, 10), 'angle': (1050, 10)}

start_event = threading.Event()
end_event = threading.Event()

width = 1920
height = 1080


class CamThread(threading.Thread):
    def __init__(self, camera_name, camera_id):
        super(CamThread, self).__init__()
        self.camera_name = camera_name
        self.camera_id = camera_id

    def run(self):
        capture_and_save(self.camera_name, self.camera_id)


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
    return resized


def capture_and_save(camera_name, camera_id):
    global width
    global height

    print(f'{camera_name} camera is waiting for trigger...')
    start_event.wait()

    print(f'Starting {camera_name} camera...')
    cv2.namedWindow(camera_name)

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    output_path = os.path.join(data_path, f'{max_video_num + 1}_{args.exercise_type}_{camera_name}.mp4')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (height, width), True)

    while cap.isOpened() and not end_event.is_set():
        ret, frame = cap.read()

        if ret:
            rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            out.write(rotated_frame)

            image = image_resize(rotated_frame, height=500)
            cv2.imshow(camera_name, image)
            cv2.moveWindow(camera_name, *placement_dict[camera_name])

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()
    print(f'Saving record from {camera_name} camera...')

    cv2.destroyWindow(camera_name)


def check_event():
    lock = threading.Lock()
    while not end_event.is_set():
        with lock:
            msg = msvcrt.getwch()
            if msg == 'z':
                start_event.set()
            if msg == 'x':
                end_event.set()


if __name__ == '__main__':
    front_cam_id = args.front_cam_id
    side_cam_id = args.side_cam_id
    angle_cam_id = args.angle_cam_id

    event_thread = threading.Thread(target=check_event)
    front_thread = CamThread('front', front_cam_id)
    side_thread = CamThread('side', side_cam_id)
    angle_thread = CamThread('angle', angle_cam_id)

    event_thread.start()
    front_thread.start()
    side_thread.start()
    angle_thread.start()

    event_thread.join()
    front_thread.join()
    side_thread.join()
    angle_thread.join()
