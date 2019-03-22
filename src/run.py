'''TODO:

   Smoother - causing to lose bbox
   Add an algorithm that takes the distance vector (from body parts) and produces distance (FC?)
   correct get_euc_body_parts_distance it doesn't gives real world coords

'''


import sys
import argparse
import time

import cv2
#import matplotlib
#matplotlib.use('cairo')
import pose_estimation_ind as pose_est_ind
import queue
#import gesture_listener as gest_est
#import pose_tracker as pose_trk
import numpy as np
from estimator import TfPoseEstimator, Human
import copy
import matplotlib.pyplot as plt

C_NUM_MISSING_GRACE =  3
C_HUMAN_LIST_Q_SIZE = 5
C_BBOX_SCALE_FACTOR = 1.1
C_CAMERA_FOV = pose_est_ind.C_CAMERA_CALIB_FOV_D#60 #deg diag
C_CAMERA_FOCAL_LEN = pose_est_ind.C_CAMERA_CALIB_FOCAL_LEN
#C_MAX_PERSON_TRACK = 10 #C_MAX_PERSON_TRACK- max num of people to track - CH: for debug only, need to consider later



def plot_image_header_annotation (image, humans, frame_w, frame_h, fps, dst_model, rotate, show_smoother):
    image_header_div = 4
    image_header_index = 1

    if (rotate):
        center = (frame_w / 2, frame_h / 2)
        M = cv2.getRotationMatrix2D(center, rotate, 1.0)
        image = cv2.warpAffine(image, M, (int(frame_w), int(frame_h)))
    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
    cv2.putText(image, 'FPS:' + str(round(fps,2)),
                (int(image_header_index*frame_w / image_header_div), 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                2)
    image_header_index+=1
    cv2.putText(image, 'Alg:' + str(dst_model),
                (int(image_header_index*frame_w / image_header_div), 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                2)
    image_header_index+=1
    if show_smoother:
        cv2.putText(image, 'SMOOTHER',
                    (int(image_header_index*frame_w / image_header_div), 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2)
    image_header_index += 1
    return image
#CH: DEBUG FUNC
def clear_image():
    image = np.zeros ([500,500,3], dtype=np.uint8)
    image.fill(255)
    return image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    #parser.add_argument('--image', type=str, default='0.jpg')
    parser.add_argument('--input', type=str, default='', help='input to play [#, .avi,.h5]. default=demo')
    parser.add_argument('--rotate', type=int, default=0, help='rotate video. default=0')
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--kpt_model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--dst_model', type=str, default='linear', help='linear,knn,lasso,ridge,mlp,svr_lin,aff')
    parser.add_argument('--camera_fov', type=int, default=0, help='camera fov (diagonal). default=calibration camera fov')
    parser.add_argument('--camera_focal_len', type=float, default=0, help='camera focal length (diagonal). default=calibration camera focal length')
    parser.add_argument('--scales', type=str, default='[None]', help='for multiple scales, eg. [1.0, (1.1, 0.05)]')
    parser.add_argument('--show', type=int, default=1, help='show video. default=0')
    parser.add_argument('--track', type=int, default=1, help='track person. default=1')
    parser.add_argument('--smooth', type=int, default=0, help='smoother person. default=0')

    #sys.stdout = pose_est_ind.Logger('Logs\_run.log')#open('Logs\sys.log', 'w')

    args = parser.parse_args()

    # Init input
    [input, camera_fov, camera_focal_len, fov_conv_factor, fps, frame_h, frame_w] = pose_est_ind.Init_Input(args)

    # App args init
    #TODO: create app inputs class and init func
    #person_dtct_list = []
    #smoother_person_dtct_list = []
    person_dtct_list_queue = queue.Queue(maxsize=C_HUMAN_LIST_Q_SIZE)
    smoother_person_dtct_list_queue = queue.Queue(maxsize=C_HUMAN_LIST_Q_SIZE)
    out = cv2.VideoWriter('plots\output_' + args.kpt_model + '_' + args.dst_model +'.avi', cv2.VideoWriter_fourcc(*'XVID'), int(fps/2), (int(frame_w), int(frame_h)))
    frame_index = 0
    t = time.time()
    person_dtct = 0
    smoother_person_dtct = 0
    show_smoother = args.smooth
    bbox_scale_factor = C_BBOX_SCALE_FACTOR
    num_missing_grace = 0
    real_fps = 0

    # Init tracking
    if (args.track):
        person_dtct_track = []
        smoother_person_dtct_track = []

    # CH: Debug
    dist_image = clear_image() #CH: debug
    f_dists = open('Logs\dists_run_' + str(args.dst_model) + '.csv', 'ab') #CH: debug

    # Run main-loop
    while (True):
        person_dtct_list = []
        smoother_person_dtct_list = []
        humans, image, succ = pose_est_ind.Infer_Humans(input, args)
        if (not succ):
            break
        for human in humans:
        ###DETECT AND CREATE PERSON/SMOOTHER FROM HUMAN###
            #track person by bbox or euclidan distance and add new persons
            bbox = pose_est_ind.GetBBox (human, bbox_resize_factor=bbox_scale_factor)
            if show_smoother:
                [new_person_id, person_old_distance] = pose_est_ind.find_person_id_by_oui(human=human, human_bbox=bbox, person_list_queue=smoother_person_dtct_list_queue,
                                                                    dst_model=args.dst_model, fov=camera_fov, focal_len=camera_focal_len, fov_conv_factor=fov_conv_factor,
                                                                    resolution=[frame_w, frame_h], fps=real_fps)
            else:
                [new_person_id, person_old_distance] = pose_est_ind.find_person_id_by_oui (human=human, human_bbox=bbox, person_list_queue=person_dtct_list_queue,
                                                                    dst_model=args.dst_model, fov=camera_fov,  focal_len=camera_focal_len, fov_conv_factor=fov_conv_factor,
                                                                    resolution=[frame_w, frame_h], fps=real_fps) #get new object or detected object from pool

            # Create new person instance w/wo unique new_person_id
            num_of_checked_persons = 0
            if new_person_id and person_dtct_list:
                for check_person_dtct in person_dtct_list:
                    if check_person_dtct.PersonId == new_person_id:
                        num_of_checked_persons += 1
            if num_of_checked_persons > 1:
                new_person_id = 0
            person_dtct = pose_est_ind.Person(PersonId=new_person_id, human=Human([]), person_detected=1)
            person_dtct.human.body_parts = copy.deepcopy(human.body_parts)
            person_dtct.bbox = copy.deepcopy(bbox)
            person_dtct.distance = copy.deepcopy(person_old_distance)
            #create ,if needed, set smoother person
            if show_smoother:
                smoother_person_dtct = pose_est_ind.smooth_human(person_dtct, person_dtct_list_queue)
                smoother_person_dtct.bbox = pose_est_ind.GetBBox(smoother_person_dtct.human, bbox_resize_factor=bbox_scale_factor)
                human = smoother_person_dtct.human

        ###SET FEATURES TO DETECTED PERSON/SMOOTHER###
            #get detected/smoother person features and show annotations
            bp_max_debug = pose_est_ind.set_person_features(person_dtct, args.dst_model, fov_conv_factor=fov_conv_factor)
            if show_smoother:
                pose_est_ind.set_person_features(smoother_person_dtct, args.dst_model, fov_conv_factor=fov_conv_factor)
        ###ARCHIVE###
            #save in list to show after and archive it for tracking and smoothing mech.(smoother_person_dtct_list_queue)
            person_dtct_list.append(person_dtct)
            if show_smoother:
                smoother_person_dtct_list.append(smoother_person_dtct)

        frame_index += 1

        #Tracking
        if (args.track and person_dtct):
            person_dtct_track.append(person_dtct.position)
            if (show_smoother and smoother_person_dtct):
                smoother_person_dtct_track.append(smoother_person_dtct.position)

        elapsed = time.time() - t
        real_fps = frame_index / elapsed
        #show image & persons annotations

        if (args.show and person_dtct): #CH: consider to set it all in a different thread
            # Plot image annotations
            image = plot_image_header_annotation (image, humans, frame_w, frame_h, real_fps, args.dst_model, args.rotate, show_smoother)
            # Plot Person (No/Smoother) features annotation
            if show_smoother:
               for smoother_person_dtct in smoother_person_dtct_list:
                    pose_est_ind.plot_person_features(smoother_person_dtct, image)
            else:
                for person_dtct in person_dtct_list:
                    pose_est_ind.plot_person_features(person_dtct, image)
            #CH: DEBUG!
            if person_dtct:
                dist_image = clear_image()
                cv2.putText(dist_image, str(person_dtct.PersonId) + ':' + str(person_dtct.distance)+'m', (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
                # debug plot
                if bp_max_debug > 0:
                    cv2.putText(dist_image, str(pose_est_ind.C_FEATURES_NAME[bp_max_debug]), (0, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            if show_smoother and smoother_person_dtct:
                cv2.putText(dist_image, str(person_dtct.PersonId) + ':' + str(smoother_person_dtct.distance)+'m', (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

            np.savetxt(f_dists, np.asarray(person_dtct.distance).reshape(1,1), delimiter=",")
            cv2.imshow('dist_image', dist_image)

            cv2.imshow('image', image)
            key = (cv2.waitKey(1) & 0xFF)
            if (key == 27):#escape
                break
            if (key == 115 or key == 83):#'s' or 'S'
                show_smoother = not show_smoother
            if (key >= 48 and ((key-48) < pose_est_ind.C_DIST_ALGS_NAMES_EX.__len__())):
                    args.dst_model = pose_est_ind.C_DIST_ALGS_NAMES_EX[key-48]
        #else:
        #    print('FPS:' + str(real_fps))

        #Write data to avi file
        if (args.show):
            out.write(image)

        '''#clear all persons which were not detected, with grace
        if (not humans and num_missing_grace >= C_NUM_MISSING_GRACE): #giving grace to tracking for C_NUM_MISSING_GRACE images to skip
            person_dtct_list = [x for x in person_dtct_list if x.person_detected != 0]
            num_missing_grace = 0
        else:
            person_dtct_list = [x for x in person_dtct_list if x.person_detected != 0]
            if num_missing_grace < C_NUM_MISSING_GRACE:
                num_missing_grace += 1
        #clear detected flag, for next loop
        if show_smoother or smoother_person_dtct_list:#reduce memory usage when smoother is not needed
            smoother_person_dtct_list = [x for x in smoother_person_dtct_list if x.person_detected != 0]
            for smoother_person_dtct in smoother_person_dtct_list:
                smoother_person_dtct.person_detected = 0
        for person_dtct in person_dtct_list:
            person_dtct.person_detected = 0
        '''

        #Push person to queue list for archiving and smoothing
        if (person_dtct_list_queue.full()):
            person_dtct_list_queue.get()
        if (person_dtct_list.__len__()):
            person_dtct_list_queue.put(copy.deepcopy(person_dtct_list))
        if show_smoother:
            if (smoother_person_dtct_list_queue.full()):
                smoother_person_dtct_list_queue.get()
            if (smoother_person_dtct_list.__len__()):
                smoother_person_dtct_list_queue.put(copy.deepcopy(smoother_person_dtct_list))



    #Main Loop - End: plot stats and run application end routin
    if (args.show and args.track):
        pose_est_ind.plot_person_course(person_dtct_track, smoother_person_dtct_track)
    print ('Finished!')
    print ('Number of frames:' + str(frame_index) + ' in ' + str(round(elapsed,2)) + 'seconds')
    out.release()
    pose_est_ind.Finit_Input(input)