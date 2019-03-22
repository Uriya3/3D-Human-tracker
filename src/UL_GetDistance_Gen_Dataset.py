import numpy as np
import cv2 as cv2
from estimator import TfPoseEstimator
import h5py
import pickle
import pose_estimation_ind as pose_est_ind
import UL_GetDistance_ML as distance_train
import UL_GetDistance3_ML as distance_train3
import os
import time
import argparse
import copy
from tqdm import tqdm

gl_e = 0

def AqcData(acq_device=0, resolution='', model='mobilenet_thin', scales='[None]', rotate=0, show=0, person_height = 1):
    '''global gl_e
    video = cv2.VideoCapture(acq_device)
    len = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w = str(int(video.get(cv2.CAP_PROP_FRAME_WIDTH)/16)*16)
    vid_h = str(int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)/16)*16)
    if not resolution:
        resolution = vid_w + 'x' + vid_h
    x_dataset_all = np.zeros([2*len, pose_est_ind.C_BACKGROUND, 2], dtype='float64')
    if (not gl_e):
        w, h = model_wh(resolution)
        gl_e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))
    e = gl_e
    scales = ast.literal_eval(scales)'''
    args.input = acq_device
    args.resolution = resolution
    args.kpt_model = model
    args.scales = scales
    args.camera_fov = -1
    args.camera_focal_len = -1
    [input, _, _, _, _, _, _] = pose_est_ind.Init_Input(args)
    len = int(input.get(cv2.CAP_PROP_FRAME_COUNT))
    x_dataset_all = np.zeros([2 * len, pose_est_ind.C_BACKGROUND, 2], dtype='float64')
    num_hum_dtctd = 0
    for image_index in tqdm(range(len)):
        [humans, image, succ] = pose_est_ind.Infer_Humans(input, args)
        if (not succ):
            continue
        if (show):
            show_image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
            cv2.imshow('image', show_image)
            if ((cv2.waitKey(1) & 0xFF) == 27):
                break
        for human in humans:
            x_dataset_all[num_hum_dtctd] = pose_est_ind.human_est_to_numpy(human)
            '''for body_part in human.body_parts:
                x_dataset_all[num_hum_dtctd][body_part] = [human.body_parts[body_part].x, human.body_parts[body_part].y]
            '''
            num_hum_dtctd += 1
            #print(str(i) + '/' + str(len))
    if (show):
        cv2.destroyAllWindows()
    pose_est_ind.Finit_Input(input)
    x_dataset_all = x_dataset_all[:num_hum_dtctd] #copies only active dataset (remove padding zeros)
    print ('x_dataset_all size is: ' + str(x_dataset_all.shape))
    return x_dataset_all

def get_all_dataset_from_videos(videos, resolution, persons_height, h5file_all, show=0, rotate=-90):
    if os.path.exists(h5file_all):
        os.remove(h5file_all)
    with h5py.File(h5file_all, 'a') as hf:
        hf.create_dataset("X_Train_All", (0, 18, 2), 'float64', maxshape=(None, pose_est_ind.C_BACKGROUND, 4), chunks=True)
        hf.create_dataset("Y_Train_All", (0,), 'float64', maxshape=(None,), chunks=True)
        #for (video, person_height) in zip(videos, persons_height):
        for video in videos:
            vid_directory = 'videos\\' + video + '\\'
            dists = []
            video_index = 0
            for i_ in os.listdir(vid_directory):
                dists.insert(video_index, os.path.splitext(i_)[0])
                video_index = video_index + 1
            #dists = sorted(dists, key=float)
            for dist in dists:
                video_name = vid_directory + str(dist) + '.avi'
                if (not os.path.isfile(video_name)):
                    continue
                print(video_name)
                X_dataset_all = AqcData(acq_device=video_name, resolution=resolution, model=model, rotate=rotate, show=show)
                Y_dataset_all = np.zeros(X_dataset_all.shape[0])
                if pose_est_ind.is_number(dist): #distance
                    Y_dataset_all.fill(float(dist))
                else: #distracted
                    if dist.find('_nl'):
                        Y_dataset_all.fill(float(1))
                    elif dist.find('_l'):
                        Y_dataset_all.fill(float(0))
                #X_dataset_all = np.divide(X_dataset_all, persons_height[0])
                hf["X_Train_All"].resize((hf["X_Train_All"].shape[0] + X_dataset_all.shape[0]), axis=0)
                hf["X_Train_All"][-X_dataset_all.shape[0]:] = X_dataset_all
                hf["Y_Train_All"].resize((hf["Y_Train_All"].shape[0] + Y_dataset_all.shape[0]), axis=0)
                hf["Y_Train_All"][-Y_dataset_all.shape[0]:] = Y_dataset_all



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UL_GetDistance_Gen_Dataset')
    parser.add_argument('--analyse_videos', type=int, default=0)
    parser.add_argument('--video_file', type=str, default='')
    parser.add_argument('--h5_file', type=str, default='')
    parser.add_argument('--extract_features', type=int, default=1)
    parser.add_argument('--train', type=int, default=1)
    parser.add_argument('--show', type=int, default=0)
    parser.add_argument('--rotate', type=int, default=0)
    args = parser.parse_args()
    analyse_videos = args.analyse_videos
    data_extract_features = args.extract_features
    model = 'mobilenet_thin'
    resolution = '432x368'
    videos = ['buri']
    timestr = time.strftime("%Y%m%d-%H%M")
    if (not args.video_file):
        if (args.h5_file == ''):
            h5file_all = 'face_keypoints_dataset_ALL' + model + timestr + '.h5'
        else:
            h5file_all = args.h5_file
    else:
        h5file_all = str(args.video_file)
    if analyse_videos:
        get_all_dataset_from_videos(videos, resolution, 0, h5file_all, args.show, rotate=args.rotate)
    if data_extract_features:
        extract_features_pickel = 'face_keypoints_dataset_Features' + model + timestr + '.pickle'
        #h5file_extract_features = 'face_keypoints_dataset_Features' + model + timestr + '.h5'
        with h5py.File(h5file_all, 'r') as hf_all:
            X_dataset_all = hf_all.get('X_Train_All')
            Y_dataset_all = hf_all.get('Y_Train_All')

            #with h5py.File(h5file_extract_features, 'a') as hf_new:
            [X_dataset_mnplt, Y_dataset_mnplt] = pose_est_ind.ExtractDataFeatures (
                x_dataset_all=X_dataset_all, y_dataset_all=Y_dataset_all, plot=1, num_of_features=pose_est_ind.C_NUM_OF_FEATURES, features_name=pose_est_ind.C_FEATURES_NAME)
            # fill in missing data with avg
            X_dataset_mnplt_no_missing = copy.deepcopy(X_dataset_mnplt)
            for feature in range(pose_est_ind.C_NUM_OF_FEATURES):
                for y in sorted(np.unique(Y_dataset_mnplt)):
                    # Avg results by y and put it where X_dataset_mnplt is [nan,nan,nan]
                    y_first_occr = np.argmin(Y_dataset_mnplt[feature] < y)
                    y_last_occr = np.argmax(Y_dataset_mnplt[feature] > y)
                    if not y_last_occr:
                        y_last_occr = X_dataset_mnplt[feature].shape[0]
                    feature_avg = np.nanmean(X_dataset_mnplt[feature][y_first_occr:y_last_occr], axis=0)
                    if np.isnan(feature_avg).any():
                        print('Data is not complete - no data in ' + pose_est_ind.C_FEATURES_NAME[feature] + ' Dist: ' + str(y))
                        exit(-1)
                    np.place(X_dataset_mnplt_no_missing[feature][y_first_occr:y_last_occr],
                             np.isnan(X_dataset_mnplt[feature][y_first_occr:y_last_occr]), feature_avg)
                    np.place(X_dataset_mnplt_no_missing[feature][y_first_occr:y_last_occr],
                             X_dataset_mnplt[feature][y_first_occr:y_last_occr] == [0, 0, 0], feature_avg)
                    '''X_dataset_mnplt_no_missing[feature][y_first_occr:y_last_occr] = \
                        np.where(np.isnan(X_dataset_mnplt[feature][y_first_occr:y_last_occr]), feature_avg
                                 np.ma.array(X_dataset_mnplt[feature][y_first_occr:y_last_occr],
                                             mask=np.isnan(X_dataset_mnplt[feature][y_first_occr:y_last_occr])).mean(axis=0), X_dataset_mnplt[feature][y_first_occr:y_last_occr])
                    '''
                    print(np.isnan(X_dataset_mnplt_no_missing[feature][y_first_occr:y_last_occr]).any())
                    print([feature, y, y_first_occr, y_last_occr])

                print(np.isnan(X_dataset_mnplt_no_missing[feature]).any())
                    #feature_avg = np.nanmean(X_dataset_mnplt[feature][y_first_occr:y_last_occr], axis=0)
                    #X_dataset_mnplt[feature][y_first_occr:y_last_occr][np.isnan(X_dataset_mnplt[feature][y_first_occr:y_last_occr])] =  feature_avg
                    #np.place(X_dataset_mnplt_no_missing[feature][y_first_occr:y_last_occr], X_dataset_mnplt[feature][y_first_occr:y_last_occr] == [0, 0, 0], feature_avg)


            # Save data to pickle
            pickle_object = {}
            for feature in range(pose_est_ind.C_NUM_OF_FEATURES):
                pickle_object["X_Train_Missing"+str(feature)] = X_dataset_mnplt[feature]
                pickle_object["X_Train_No_Missing" + str(feature)] = X_dataset_mnplt_no_missing[feature]
                pickle_object["Y_Train" + str(feature)] = Y_dataset_mnplt[feature]

            pickle_object["NUM_OF_SUBFEATURES"] = pose_est_ind.C_NUM_OF_SUBFEATURES
            pickle_object["NUM_OF_FEATURES"] = pose_est_ind.C_NUM_OF_FEATURES
            pickle_object["FEATURES_NAMES"] = pose_est_ind.C_FEATURES_NAME
            pickle_object["NUM_OF_FRAMES"] = X_dataset_mnplt[feature].shape[0]
            #ascii_l = [n.encode("ascii", "ignore") for n in pose_est_ind.C_FEATURES_NAME]
            #hf_new["FEATURES_NAMES"] = ascii_l
            pickle.dump(pickle_object, open(extract_features_pickel,'wb'))

        print('Finished!')
        print('h5: ' + str(h5file_all))
        print('h5_extract_features:' + str(extract_features_pickel))

    if (args.train):

        distance_train.train_and_eval(extract_features_pickel, 1)
        distance_train.create_sink_model(extract_features_pickel)

        distance_train3.train_and_eval(h5file_all, 1)
