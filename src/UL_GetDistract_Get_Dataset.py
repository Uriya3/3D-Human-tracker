import numpy as np
import cv2 as cv2
import h5py
import pickle
import pose_estimation_ind as pose_est_ind
import UL_GetDistance_ML as distance_train
import UL_GetDistance3_ML as distance_train3
import os
import msvcrt
import argparse
from tqdm import tqdm
import time

def AqcData(acq_device=0, show=1):
    args.input = acq_device
    video = cv2.VideoCapture(args.input)
    len = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    x_dataset_all = np.zeros((len,), dtype='bool')
    distracted = False
    for image_index in tqdm(range(len)):
        succ, image = video.read()
        if (not succ):
            continue
        if (show):
            image_h, image_w = image.shape[:2]
            cv2.putText(image, 'Distracted = ' + str(distracted),
                        (int(image_h), int(image_w/2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        3)
            cv2.imshow('image', image)
            if ((cv2.waitKey(1) & 0xFF) == 27):
                break
            if ((cv2.waitKey(1) & 0xFF) == 100):
                distracted = not distracted

        x_dataset_all[image_index] = distracted

    if (show):
        cv2.destroyAllWindows()
    pose_est_ind.Finit_Input(video)

    return x_dataset_all


def get_all_dataset_from_videos(videos, h5file_all, show=0):
    if os.path.exists(h5file_all):
        os.remove(h5file_all)
    with h5py.File(h5file_all, 'a') as hf:
        hf.create_dataset("Y_Train_All", (0,), 'float64', maxshape=(None,), chunks=True)
        for video in videos:
            vid_directory = 'videos\\' + video + '\\'
            dists = []
            video_index = 0
            for i_ in os.listdir(vid_directory):
                dists.insert(video_index, os.path.splitext(i_)[0])
                video_index = video_index + 1
            dists = sorted(dists, key=float)
            for dist in dists:
                video_name = vid_directory + str(dist) + '.mp4'
                if (not os.path.isfile(video_name)):
                    continue
                print(video_name)
                Y_dataset_all = AqcData(acq_device=video_name, show=show)
                hf["Y_Train_All"].resize((hf["Y_Train_All"].shape[0] + Y_dataset_all.shape[0]), axis=0)
                hf["Y_Train_All"][-Y_dataset_all.shape[0]:] = Y_dataset_all

def get_all_dataset_from_h5file(h5file, h5file_all):
    if os.path.exists(h5file_all):
        os.remove(h5file_all)
    with h5py.File(h5file, 'r') as hf_in:
        X_dataset_In = hf_in.get('X_Train_All')
        Y_dataset_all = np.empty(X_dataset_In.shape[0])
        with h5py.File(h5file_all, 'a') as hf:
            hf.create_dataset("X_Train_All", (0, 18, 2), 'float64', maxshape=(None, pose_est_ind.C_BACKGROUND, 4),
                              chunks=True)
            hf["X_Train_All"].resize((hf["X_Train_All"].shape[0] + X_dataset_In.shape[0]), axis=0)
            hf["X_Train_All"][-X_dataset_In.shape[0]:] = X_dataset_In
            hf.create_dataset("Y_Train_All", (0,), 'float64', maxshape=(None,), chunks=True)

            for frame_indx, body_parts in enumerate(X_dataset_In):
                ratio = pose_est_ind.get_eye_nose_ratio_xy(body_parts[pose_est_ind.C_NOSE][0], body_parts[pose_est_ind.C_NOSE][1],
                            body_parts[pose_est_ind.C_RIGHT_EYE][0], body_parts[pose_est_ind.C_RIGHT_EYE][1],
                                           body_parts[pose_est_ind.C_LEFT_EYE][0], body_parts[pose_est_ind.C_LEFT_EYE][1])
                distracted = (ratio < 0.8 or ratio > 1.2)
                Y_dataset_all[frame_indx] = distracted

            hf["Y_Train_All"].resize((hf["Y_Train_All"].shape[0] + Y_dataset_all.shape[0]), axis=0)
            hf["Y_Train_All"][-Y_dataset_all.shape[0]:] = Y_dataset_all


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UL_GetDistance_Gen_Dataset')
    parser.add_argument('--video_file', type=str, default='')
    parser.add_argument('--h5_file', type=str, default='')
    parser.add_argument('--show', type=int, default=1)
    args = parser.parse_args()
    model = 'mobilenet_thin'
    resolution = '368x368'
    videos = ['buri']
    timestr = time.strftime("%Y%m%d-%H%M")
    if (not args.video_file):
        if (args.h5_file == ''):
            h5file_all = 'labels_distracted_ALL' + model + timestr + '.h5'
        else:
            h5file_all = args.h5_file
    else:
        h5file_all = str(args.video_file)

    #get_all_dataset_from_videos(videos, h5file_all, args.show)
    get_all_dataset_from_h5file('default_368x368_ALL_web.h5', h5file_all)