import pose_estimation_ind as pose_est_ind
import argparse
import cv2
import copy



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    # parser.add_argument('--image', type=str, default='0.jpg')
    parser.add_argument('--input', type=str, default='', help='input to play [#, .avi,.h5]. default=demo')
    parser.add_argument('--rotate', type=int, default=0, help='rotate video. default=0')
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--kpt_model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--camera_fov', type=int, default=0,
                        help='camera fov (diagonal). default=calibration camera fov')
    parser.add_argument('--camera_focal_len', type=float, default=0,
                        help='camera focal length (diagonal). default=calibration camera focal length')
    parser.add_argument('--scales', type=str, default='[None]', help='for multiple scales, eg. [1.0, (1.1, 0.05)]')

    # Init input
    args = parser.parse_args()
    [input, camera_fov, camera_focal_len, fov_conv_factor, fps, frame_h, frame_w] = pose_est_ind.Init_Input(args)

    pre_recording = 0
    recording = 0
    frame_cnt = 0
    num_of_files_cnt = 0
    while (1):
        succ, image = input.read()
        if (not succ):
            break
        image_disp = copy.deepcopy(image)
        key = (cv2.waitKey(1) & 0xFF)
        if (key == 114):
            pre_recording = 1
            timeout = 0
            frame_cnt = 0
            video_name = str(num_of_files_cnt) + '.avi'
            out = cv2.VideoWriter(video_name,
                                  cv2.VideoWriter_fourcc(*'DIB '), fps, (int(frame_w), int(frame_h)))
        if (key == 27):  # escape
            break

        if pre_recording:
            timeout += 1
            cv2.putText(image_disp, 'R',
                        (int(frame_w / 2), 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        3)
            cv2.putText(image_disp, str(timeout),
                        (int(frame_w / 2), 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        3)
            if timeout >= 150:
                recording = 1
                pre_recording = 0
        if recording:
            frame_cnt += 1
            out.write(image)
            cv2.putText(image_disp, 'R',
                        (int(frame_w / 2), 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3)
            cv2.putText(image_disp, str(frame_cnt),
                        (int(frame_w / 2), 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3)

        if frame_cnt >= 2000:
            recording = 0
            out.release()
            num_of_files_cnt += 1
            frame_cnt = 0

        cv2.imshow('image', image_disp)

