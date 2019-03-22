import cv2
import numpy as np
import math
import pickle
from estimator import Human, BodyPart, TfPoseEstimator
from networks import get_graph_path, model_wh
import ast
import os
import copy
import sys
import matplotlib.pyplot as plt
#TODO: create person QUEUE class and add it - find_person_id_by_oui,
#TODO: add to Person class - Person(), smooth_human, set_person_features, plot_person_features as public and other as privet
#####BODY CONSTANTS#####
C_NOSE = 0
C_NECK = 1
C_RIGHT_SHOULDER = 2
C_RIGHT_ELBOW = 3
C_RIGHT_WRIST = 4
C_LEFT_SHOULDER = 5
C_LEFT_ELBOW = 6
C_LEFT_WRIST = 7
C_RIGHT_HIP = 8
C_RIGHT_KNEE = 9
C_RIGHT_ANKLE = 10
C_LEFT_HIP = 11
C_LEFT_KNEE = 12
C_LEFT_ANKLE = 13
C_RIGHT_EYE = 14
C_LEFT_EYE = 15
C_RIGHT_EAR = 16
C_LEFT_EAR = 17
C_BACKGROUND = 18

#####TRAINING CONSTANTS#####
C_NUM_OF_FEATURES = 9
C_NUM_OF_SUBFEATURES = 3
C_FEATURES_NAME= ['center_y_body_dist', 'r_shoulder_elbow_dist', 'l_shoulder_elbow_dist',
                               'r_elbow_wrist_dist', 'l_elbow_wrist_dist',
                               'r_hip_knee_dist', 'l_hip_knee_dist', 'r_knee_ankle_dist', 'l_knee_ankle_dist']
C_DIST_ALGS_NAMES = ["svr", "ridge", "linear", "lasso", "mlp", "poly", "aff"]
C_DIST_ALGS_NAMES_EX = ["svr", "ridge", "linear", "lasso", "mlp", "poly", "aff", "sf"]
#C_DIST_ALGS_NAMES = ["svr", "ridge", "linear", "lasso", "mlp", "poly"]

#####CAMERA CONSTANTS#####
C_IOU_MIN_OVRLP = 0.6
C_EUC_MIN_DIST = 5 #5 meters per seconds
C_CAMERA_CALIB_FOV_HV = [53.4,31.6] #[H,V] diag - 60 with aspect 16:9 # webcam c310
#C_CAMERA_CALIB_FOV_D = 78 #LG g4
C_CAMERA_CALIB_FOV_D  = 60 # webcam c310
#C_CAMERA_CALIB_FOCAL_LEN = 4.42/1000 #LG g4
C_CAMERA_CALIB_FOCAL_LEN = 4.4/1000 # webcam c310


#####CLASS#####
class Est_Object:
    e = 0
    scales = 0
    infer_res_w = 0
    infer_res_h = 0

class Person:
    PersonId = 0
    human = []
    bbox = []
    distance = 0
    IsApproching_Ind = 0
    IsDistracted_Ind = 0
    IsRightHandRaised_Ind = 0
    IsLeftHandRaised_Ind = 0
    person_detected = 0
    position = 0
    def __init__(self,PersonId=0, human=[], person_detected=0):
        if (PersonId):
            self.PersonId = PersonId
        else:
            Person.PersonId += 1
            self.PersonId = Person.PersonId
        self.human = human
        self.person_detected = person_detected

class Logger (object):
    def __init__(self, file):
        if (not file):
            file = 'Logs\sys.log'
        self.terminal = sys.stdout
        self.log = open(file, 'w')
    def write (self, message):
        self.terminal.write (message)
        self.log.write(message)
    def flush(self):
        pass

#####GLOBALS#####
gl_loaded_model = []
gl_loaded_model_sf = []
gl_loaded_model_distracted_sf = []
gl_sink_model = []
gl_loaded_model_name = ""
gl_est = 0

#####FUNCTIONS#####
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def conv_fov_diag_to_H_V (fov_diag_angle, aspect_ratio):
    #From fov_diag_angle to fovH and fovV
    '''
    FOV_Horizontal = 2 * atan(W/2/f) = 2 * atan2(W/2, f)  radians
    FOV_Vertical   = 2 * atan(H/2/f) = 2 * atan2(H/2, f)  radians
    FOV_Diagonal   = 2 * atan2(sqrt(W^2 + H^2)/2, f)    radians
    '''
    fovD_rad = math.radians(fov_diag_angle)
    diag_ratio_H = math.sqrt(1 + math.pow(1/aspect_ratio, 2))
    diag_ratio_V = math.sqrt(1 + math.pow(aspect_ratio, 2))
    fovH_rad = 2 * math.atan2(math.tan(fovD_rad / 2), diag_ratio_H)
    fovV_rad = 2 * math.atan2(math.tan(fovD_rad / 2), diag_ratio_V)

    return math.degrees(fovH_rad), math.degrees(fovV_rad)

def conv_2d_to_3d (point, fov, focal_len, resolution, distance):
    alpha = 1
    centerX = resolution[0] / 2
    centerY = resolution[1] / 2

    if (type(fov) == list): #if not list -> fov diag
        [fovH, fovV] = [fov[0], fov[1]]
    else:
        #[fovH, fovV] = conv_fov_diag_to_H_V(fov, resolution[0] / resolution[1])
        [fovH, fovV] = [fov, fov*resolution[1]/resolution[0]]

    FocalLenPixel = [centerX / np.tan(math.radians(fovH)/2),
                     centerY / np.tan(math.radians(fovV)/2)]
    A = [[FocalLenPixel[0], 0, centerX],
         [0, FocalLenPixel[1], centerY],
         [0, 0, 1]]
    pic_point = [point[0]*resolution[0], point[1]*resolution[1], 1]
    world_coord = np.linalg.inv(A).dot(alpha * pic_point) #get world cordinate for alpha meter distance
    world_coord_ret = (world_coord / np.linalg.norm(world_coord)) * distance #takes normalized vector multiplied with distance
    return world_coord_ret

def rotate(origin, point, angle):#radians
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

    return qx, qy

def get_eye_nose_ratio_xy(x_base, y_base, x1,y1,x2,y2):
    ratio = 0
    div = math.hypot(x1 - x_base, y1 - y_base)
    if (div):
        ratio = math.hypot(x2 - x_base, y2 - y_base) / div
    return ratio

def get_eye_nose_ratio (bp_nose, bp_right_eye, bp_left_eye):
    return get_eye_nose_ratio_xy (bp_nose.x,bp_nose.y, bp_right_eye.x,bp_right_eye.y, bp_left_eye.x,bp_left_eye.y)

def get_joints_distance_xy (x1, y1, x2, y2):
    dist = math.hypot(x1 - x2, y1 - y2)
    avg_coord_x = (x1 + x2)/2
    avg_coord_y = (y1 + y2) / 2
    return [dist, avg_coord_x, avg_coord_y]


def get_joints_distance (bp_joint1, bp_joint2):
    return get_joints_distance_xy (bp_joint1.x, bp_joint1.y, bp_joint2.x, bp_joint2.y)

def get_joints_angle_xy (x_base, y_base, x1, y1, x2, y2):
    v0 = np.array([x1, y1]) - np.array([x_base, y_base])
    v1 = np.array([x2, y2]) - np.array([x_base, y_base])
    angle = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
    return np.degrees(angle)


def get_joints_angle (bp_basejoint, bp_joint1, bp_joint2):
    return get_joints_angle_xy (bp_basejoint.x, bp_basejoint.y, bp_joint1.x, bp_joint1.y, bp_joint2.x, bp_joint2.y)


def get_subfeatures (x1, y1, x2, y2):
    dist = math.hypot(x1 - x2, y1 - y2)
    #CH: it's not correct!
    angle = np.radians(get_joints_angle_xy (x1, y1, x2, y2, x1, y2)) / math.pi
    valid_feature = 1
    return [dist, angle, valid_feature]


def get_eye_nose_ratio_center (nose, right_eye, left_eye):
    ratio = 0
    div = math.hypot(left_eye[0] - nose[0], left_eye[1] - nose[1])
    if (div):
        ratio = math.hypot(right_eye[0] - nose[0], right_eye[1] - nose[1]) / div
    return ratio

def get_joints_distance_center (joint1, joint2):
    dist = math.hypot(joint1[0] - joint2[0], joint1[1] - joint2[1])
    return dist

def get_joints_angle_center (basejoint, joint1, joint2):
    v0 = np.array([joint1[0], joint1[1]]) - np.array([basejoint[0], basejoint[1]])
    v1 = np.array([joint2[0], joint2[1]]) - np.array([basejoint[0], basejoint[1]])
    angle = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
    return np.degrees(angle)
    return dist

def get_skel_avg_size (human):
    num_of_parts_aq = 0
    total_parts_len = 0
    SkelAvgSize = 0
    if (C_RIGHT_HIP in human.body_parts and C_LEFT_HIP in human.body_parts):
        total_parts_len += get_joints_distance(human.body_parts[C_RIGHT_HIP], human.body_parts[C_LEFT_HIP])
        num_of_parts_aq += 1
    if (C_RIGHT_SHOULDER in human.body_parts and C_LEFT_SHOULDER in human.body_parts):
        total_parts_len += get_joints_distance(human.body_parts[C_RIGHT_SHOULDER], human.body_parts[C_LEFT_SHOULDER])
        num_of_parts_aq += 1
    if (C_NOSE in human.body_parts and C_NECK in human.body_parts):
        total_parts_len += get_joints_distance(human.body_parts[C_NOSE], human.body_parts[C_NECK])
        num_of_parts_aq += 1
    if (num_of_parts_aq):
        SkelAvgSize = total_parts_len / num_of_parts_aq
    return SkelAvgSize

def get_person_xy(body_parts):
    position_xy = []
    return position_xy

def get_feature_avg(x_dataset, y_dataset):
    # feature avg blobs
    avg_results = []
    for y in sorted(np.unique(y_dataset)):
        y_first_occr = np.argmin(y_dataset < y)
        y_last_occr = np.argmax(y_dataset > y)
        if not y_last_occr:
            y_last_occr = x_dataset.shape[0]
        avg_results.append([np.average(x_dataset[y_first_occr:y_last_occr][0], axis=0), y])
        # avg_results.append([np.average(x_dataset[feature,:,0][np.where(y_dataset[feature] == y)][...,]),y])
    return np.array(avg_results)

def calc_fov_conv_factor (camera_fov):
    camera_fov_rad = camera_fov * math.pi /180
    camera_calib_fov_rad = C_CAMERA_CALIB_FOV_D * math.pi / 180
    fov_conv_factor = math.fabs(math.tan(camera_fov_rad/2) / math.tan(camera_calib_fov_rad/2))
    return fov_conv_factor

def human_est_to_numpy(human):
    body_part_ret = np.zeros([C_BACKGROUND, 2], dtype='float64')
    for body_part in human.body_parts:
        body_part_ret[body_part] = [human.body_parts[body_part].x, human.body_parts[body_part].y]

    return body_part_ret

def fill_miss_dist_avg (ret_body_dist):
    for frame_indx, frame in enumerate(ret_body_dist):
        fill_dist_avg = np.nanmean(frame)
        for xx_indx, xx in enumerate(frame):
            if (math.isnan(xx)):
                ret_body_dist[frame_indx][xx_indx] = float(fill_dist_avg)
    return np.asarray(ret_body_dist)


def prep_data_sink_pred(x_test_features, loaded_model, dst_model, fov_conv_factor):
    if not np.isnan(x_test_features).any():  # if empty, the feature is 0 -> ignore the feature
        #x_test_features_factored = [x * fov_conv_factor for x in x_test_features]
        x_test_features[0][0] *= fov_conv_factor #conv only the first feature (body parts distance)
        if (dst_model.lower() == 'aff'):
            if x_test_features[:,0].any():
                ret_body_dist = 1 / (loaded_model * x_test_features[:,0])
            else:
                ret_body_dist = np.array([np.nan])
        else:
            ret_body_dist = loaded_model[0].predict(x_test_features)

    else:
        ret_body_dist = np.array([np.nan])
        #ret_body_dist = np.array(np.nan, dtype=np.float64)
    return ret_body_dist

def pred_data_sink(ret_body_dist, sink_model, dst_model):
    ret_body_dist_avg = []
    if not np.count_nonzero(~np.isnan(ret_body_dist)) == 0:
        # Fill missing distances with avg
        ret_body_dist = fill_miss_dist_avg (ret_body_dist)
        '''for frame_indx, frame in enumerate(ret_body_dist):
            fill_dist_avg = np.nanmean(frame)
            for xx_indx, xx in enumerate(frame):
                if (math.isnan(xx)):
                    ret_body_dist[frame_indx][xx_indx] = float(fill_dist_avg)

        ret_body_dist = np.asarray(ret_body_dist)
        '''
        ret_body_dist_avg.append(sink_model[C_DIST_ALGS_NAMES.index(dst_model)].predict(ret_body_dist))
    else:
        ret_body_dist_avg = np.zeros((1,1))
    ret_body_dist_avg = np.asarray(ret_body_dist_avg)
    return ret_body_dist_avg.reshape(ret_body_dist_avg.shape[1:])

def ExtractDataFeatures(x_dataset_all, y_dataset_all=None, plot = 0, num_of_features=0, features_name=[], default_val=np.nan):
    len = x_dataset_all.shape[0]
    x_dataset_aux = np.empty([x_dataset_all.shape[0], num_of_features, C_NUM_OF_SUBFEATURES], dtype='float64')
    x_dataset_aux[:] = default_val
    #x_dataset_aux = np.zeros([x_dataset_all.shape[0], num_of_features, C_NUM_OF_SUBFEATURES], dtype='float64')
    #x_dataset_aux = []#np.zeros([x_dataset_all.shape[0], num_of_features], dtype='object')
    if y_dataset_all is not None:
        y_dataset_aux = np.zeros([y_dataset_all.shape[0], num_of_features], dtype='float64')
    num_of_valid_data = 0
    for dataset_indx, body_parts in enumerate(x_dataset_all):
        eyes_nose_ang = default_val
        ## Eye-nose angle
        if (body_parts[C_RIGHT_EYE].all() and body_parts[C_LEFT_EYE].all() and body_parts[C_NOSE].all()):
            #eyes_dist = pose_est_ind.get_joints_distance_xy(body_parts[C_RIGHT_EYE][0], body_parts[C_RIGHT_EYE][1], body_parts[C_LEFT_EYE][0], body_parts[C_LEFT_EYE][1])
            #ratio = pose_est_ind.get_eye_nose_ratio_xy (body_parts[C_NOSE][0], body_parts[C_NOSE][1], body_parts[C_RIGHT_EYE][0], body_parts[C_RIGHT_EYE][1]
            #                                            , body_parts[C_LEFT_EYE][0], body_parts[C_LEFT_EYE][1])
            eyes_nose_ang = get_joints_angle_xy(body_parts[C_NOSE][0], body_parts[C_NOSE][1], body_parts[C_RIGHT_EYE][0],
                                                             body_parts[C_RIGHT_EYE][1]
                                                        , body_parts[C_LEFT_EYE][0], body_parts[C_LEFT_EYE][1])
            #nose_left_eye_dist = pose_est_ind.get_joints_distance_xy(body_parts[C_NOSE][0], body_parts[C_NOSE][1], body_parts[C_LEFT_EYE][0], body_parts[C_LEFT_EYE][1])
            #nose_right_eye_dist = pose_est_ind.get_joints_distance_xy(body_parts[C_NOSE][0], body_parts[C_NOSE][1], body_parts[C_RIGHT_EYE][0], body_parts[C_RIGHT_EYE][1])
        r_nose_eye_dist = default_val
        ## Right Eye-nose distance
        if (body_parts[C_RIGHT_EYE].all() and body_parts[C_NOSE].all()):
            r_nose_eye_dist = get_joints_distance_xy(body_parts[C_NOSE][0], body_parts[C_NOSE][1], body_parts[C_RIGHT_EYE][0],
                                                                  body_parts[C_RIGHT_EYE][1])
        l_nose_eye_dist = default_val
        ## Left Eye-nose distance
        if (body_parts[C_LEFT_EYE].all() and body_parts[C_NOSE].all()):
             l_nose_eye_dist = get_joints_distance_xy(body_parts[C_NOSE][0], body_parts[C_NOSE][1], body_parts[C_LEFT_EYE][0],
                                                                   body_parts[C_LEFT_EYE][1])
        eyes_dist = default_val
        ## Eyes distance
        if (body_parts[C_RIGHT_EYE].all() and body_parts[C_LEFT_EYE].all()):
            eyes_dist = get_joints_distance_xy(body_parts[C_RIGHT_EYE][0], body_parts[C_RIGHT_EYE][1], body_parts[C_LEFT_EYE][0],
                                                            body_parts[C_LEFT_EYE][1])
        l_eye_ear_dist = default_val
        ## Left Eye-Ear distance
        if (body_parts[C_LEFT_EYE].all() and body_parts[C_LEFT_EAR].all()):
             l_eye_ear_dist = get_joints_distance_xy(body_parts[C_LEFT_EAR][0], body_parts[C_LEFT_EAR][1], body_parts[C_LEFT_EYE][0],
                                                                  body_parts[C_LEFT_EYE][1])
        r_eye_ear_dist = default_val
        ## Right Eye-Ear distance
        if (body_parts[C_RIGHT_EYE].all() and body_parts[C_RIGHT_EAR].all()):
            r_eye_ear_dist = get_joints_distance_xy(body_parts[C_RIGHT_EAR][0], body_parts[C_RIGHT_EAR][1], body_parts[C_RIGHT_EYE][0],
                                                                 body_parts[C_RIGHT_EYE][1])
        r_shoulder_elbow_dist = np.empty(C_NUM_OF_SUBFEATURES, dtype='float64')
        r_shoulder_elbow_dist[:] = default_val
        #r_shoulder_elbow_dist = np.array([default_val, default_val, np.nan]) #np.zeros(C_NUM_OF_SUBFEATURES) #
        ## Right Shoulder-Elbow distance
        if (body_parts[C_RIGHT_SHOULDER].all() and body_parts[C_RIGHT_ELBOW].all()):
            r_shoulder_elbow_dist = get_subfeatures(body_parts[C_RIGHT_SHOULDER][0], body_parts[C_RIGHT_SHOULDER][1], body_parts[C_RIGHT_ELBOW][0],
                                                                        body_parts[C_RIGHT_ELBOW][1])
        r_elbow_wrist_dist = np.empty(C_NUM_OF_SUBFEATURES, dtype='float64')
        r_elbow_wrist_dist[:] = default_val
        #r_elbow_wrist_dist = np.array([np.nan, np.nan, np.nan]) #np.zeros(C_NUM_OF_SUBFEATURES)
        ## Right Wrist-Elbow distance
        if (body_parts[C_RIGHT_ELBOW].all() and body_parts[C_RIGHT_WRIST].all()):
            r_elbow_wrist_dist = get_joints_distance_xy(body_parts[C_RIGHT_SHOULDER][0], body_parts[C_RIGHT_SHOULDER][1], body_parts[C_RIGHT_WRIST][0],
                                                                     body_parts[C_RIGHT_WRIST][1])
        l_shoulder_elbow_dist = np.empty(C_NUM_OF_SUBFEATURES, dtype='float64')
        l_shoulder_elbow_dist[:] = default_val
        #l_shoulder_elbow_dist = np.array([np.nan, np.nan, np.nan]) #np.zeros(C_NUM_OF_SUBFEATURES)
        ## Left Shoulder-Elbow distance
        if (body_parts[C_LEFT_SHOULDER].all() and body_parts[C_LEFT_ELBOW].all()):
            l_shoulder_elbow_dist = get_subfeatures(body_parts[C_RIGHT_SHOULDER][0], body_parts[C_RIGHT_SHOULDER][1], body_parts[C_RIGHT_ELBOW][0],
                                                                        body_parts[C_RIGHT_ELBOW][1])
        l_elbow_wrist_dist = np.empty(C_NUM_OF_SUBFEATURES, dtype='float64')
        l_elbow_wrist_dist[:] = default_val
        #l_elbow_wrist_dist = np.array([np.nan, np.nan, np.nan]) #np.zeros(C_NUM_OF_SUBFEATURES)
        ## Left Wrist-Elbow distance
        if (body_parts[C_LEFT_ELBOW].all() and body_parts[C_LEFT_WRIST].all()):
            l_elbow_wrist_dist = get_subfeatures(body_parts[C_LEFT_SHOULDER][0], body_parts[C_LEFT_SHOULDER][1], body_parts[C_LEFT_WRIST][0],
                                                                     body_parts[C_LEFT_WRIST][1])
        r_hip_knee_dist = np.empty(C_NUM_OF_SUBFEATURES, dtype='float64')
        r_hip_knee_dist[:] = default_val
        #r_hip_knee_dist = np.array([np.nan, np.nan, np.nan]) #np.zeros(C_NUM_OF_SUBFEATURES)
        ## Right Hip-Knee distance
        if (body_parts[C_RIGHT_HIP].all() and body_parts[C_RIGHT_KNEE].all()):
            r_hip_knee_dist = get_subfeatures(body_parts[C_RIGHT_HIP][0],
                                                                        body_parts[C_RIGHT_HIP][1],
                                                                        body_parts[C_RIGHT_KNEE][0],
                                                                        body_parts[C_RIGHT_KNEE][1])
        r_knee_ankle_dist = np.empty(C_NUM_OF_SUBFEATURES, dtype='float64')
        r_knee_ankle_dist[:] = default_val
        #r_knee_ankle_dist = np.array([np.nan, np.nan, np.nan]) #np.zeros(C_NUM_OF_SUBFEATURES)
        ## Right Knee-Ankle distance
        if (body_parts[C_RIGHT_KNEE].all() and body_parts[C_RIGHT_ANKLE].all()):
            r_knee_ankle_dist = get_subfeatures(body_parts[C_RIGHT_KNEE][0],
                                                                     body_parts[C_RIGHT_KNEE][1],
                                                                     body_parts[C_RIGHT_ANKLE][0],
                                                                     body_parts[C_RIGHT_ANKLE][1])
        l_hip_knee_dist = np.empty(C_NUM_OF_SUBFEATURES, dtype='float64')
        l_hip_knee_dist[:] = default_val
        #l_hip_knee_dist = np.array([np.nan, np.nan, np.nan]) #np.zeros(C_NUM_OF_SUBFEATURES)
        ## Left Hip-Knee distance
        if (body_parts[C_LEFT_HIP].all() and body_parts[C_LEFT_KNEE].all()):
            l_hip_knee_dist = get_subfeatures(body_parts[C_LEFT_HIP][0],
                                                                        body_parts[C_LEFT_HIP][1],
                                                                        body_parts[C_LEFT_KNEE][0],
                                                                        body_parts[C_LEFT_KNEE][1])
        l_knee_ankle_dist = np.empty(C_NUM_OF_SUBFEATURES, dtype='float64')
        l_knee_ankle_dist[:] = default_val
        #l_knee_ankle_dist = np.array([np.nan, np.nan, np.nan]) #np.zeros(C_NUM_OF_SUBFEATURES)
        ## Left Knee=Ankle distance
        if (body_parts[C_LEFT_KNEE].all() and body_parts[C_LEFT_ANKLE].all()):
            l_knee_ankle_dist = get_subfeatures(body_parts[C_LEFT_KNEE][0],
                                                                     body_parts[C_LEFT_KNEE][1],
                                                                     body_parts[C_LEFT_ANKLE][0],
                                                                     body_parts[C_LEFT_ANKLE][1])

        center_y_body_dist = np.empty(C_NUM_OF_SUBFEATURES, dtype='float64')
        center_y_body_dist[:] = default_val
        #center_y_body_dist = np.array([np.nan, np.nan, np.nan]) #np.zeros(C_NUM_OF_SUBFEATURES)
        ## Center body y-distance
        if (body_parts[C_LEFT_SHOULDER].all() and body_parts[C_RIGHT_SHOULDER].all() and body_parts[C_LEFT_HIP].all()
                and body_parts[C_RIGHT_HIP].all()):
            upper_body_center = ((body_parts[C_LEFT_SHOULDER][0]+body_parts[C_RIGHT_SHOULDER][0])/2,
                                 (body_parts[C_LEFT_SHOULDER][1]+body_parts[C_RIGHT_SHOULDER][1])/2)
            lower_body_center = ((body_parts[C_LEFT_HIP][0] + body_parts[C_RIGHT_HIP][0]) / 2,
                                 (body_parts[C_LEFT_HIP][1] + body_parts[C_RIGHT_HIP][1]) / 2)
            center_y_body_dist = get_subfeatures(upper_body_center[0], upper_body_center[1], lower_body_center[0], lower_body_center[1])

        features_list = np.array([center_y_body_dist, r_shoulder_elbow_dist, l_shoulder_elbow_dist, r_elbow_wrist_dist, l_elbow_wrist_dist,
                                  r_hip_knee_dist, l_hip_knee_dist, r_knee_ankle_dist, l_knee_ankle_dist])
        debug_fix_bug = 0 #fixing the np.count_nonzero bug which counts on the first element type
        '''if features_list[0] == 0:
            features_list[0] = [0,0,0]
            debug_fix_bug = 1
        denum = np.count_nonzero(features_list) - debug_fix_bug
        '''
        #denum = np.count_nonzero(features_list)
        # if (not denum):
            #continue
        if np.isnan(features_list).all():
            continue

        x_dataset_aux[num_of_valid_data] = features_list
        #x_dataset_aux.append(features_list)
        if y_dataset_all is not None:
            #y_dataset_aux[num_of_valid_data] = y_dataset_all[num_of_valid_data]*(x_dataset_aux[num_of_valid_data] > 0) #insert only y where x exists
            #y_dataset_aux[num_of_valid_data] = y_dataset_all[num_of_valid_data] * (x_dataset_aux[num_of_valid_data].any() != 0)
            #y_dataset_aux.append(y_dataset_all[num_of_valid_data])
            y_dataset_aux[num_of_valid_data] = y_dataset_all[dataset_indx]
        num_of_valid_data += 1
        if (len > 1):
            print(str(num_of_valid_data) + '/' + str(len))

    #x_dataset = list(map(list,zip(*x_dataset_aux)))
    #x_dataset =  np.transpose(x_dataset_aux)
    x_dataset = np.transpose(x_dataset_aux[:num_of_valid_data], axes=[1, 0, 2])
    if y_dataset_all is not None:
        y_dataset = np.transpose(y_dataset_aux[:num_of_valid_data])
    else:
        y_dataset = 0

    '''if not x_dataset.shape[1] or not y_dataset.shape[1]:
        print('No features!!! exiting!')
        return
    '''
    '''for feature in range(x_dataset_aux[0].__len__()):
        #x_dataset.insert(feature, x_dataset_aux[np.where(x_dataset_aux[:num_of_valid_data, feature] > 0), feature])
        x_dataset.insert(feature, x_dataset_aux[:num_of_valid_data, feature])
    if y_dataset_all is not None:
        for feature in range(y_dataset_aux.shape[1]):
            #y_dataset.insert(feature, y_dataset_aux[np.where(y_dataset_aux[:num_of_valid_data, feature] > 0), feature])
            y_dataset.insert(feature, y_dataset_aux[:num_of_valid_data, feature])
    '''
    subfeature = 0
    if (plot and y_dataset_all is not None):
        # plot feature points
        jet = plt.get_cmap('jet')
        colors_features = iter(jet(np.linspace(0, 1, num_of_features)))
        for feature in range(num_of_features):
            feature_color = next(colors_features)
            plt.scatter(x_dataset[feature,:,subfeature], y_dataset[feature], color=feature_color,
                        label=features_name[feature])
            #plot feature avg blobs
            #avg_results = get_feature_avg(x_dataset[feature], y_dataset[feature])
            avg_results = []
            for y in sorted(np.unique(y_dataset[0])):
                y_first_occr = np.argmin(y_dataset[feature] < y)
                y_last_occr = np.argmax(y_dataset[feature] > y)
                if not y_last_occr:
                    y_last_occr = x_dataset[feature].shape[0]
                avg_results.append([np.nanmean(x_dataset[feature,y_first_occr:y_last_occr][:,subfeature], axis=0), y])
                #avg_results.append([np.average(x_dataset[feature,:,0][np.where(y_dataset[feature] == y)][...,]),y])
            avg_results = np.array(avg_results)

            plt.scatter(avg_results[:, 0], avg_results[:, 1], color=feature_color, label='feature_avg', s=15 ** 2)

        plt.legend()
        plt.savefig('plots/features.png', bbox_inches='tight')
        plt.show()
    return x_dataset, y_dataset



def IsHandRaised(human):
    left_hand_ind = (C_LEFT_SHOULDER in human.body_parts and C_LEFT_WRIST in human.body_parts and human.body_parts[C_LEFT_SHOULDER].y > human.body_parts[C_LEFT_WRIST].y)
                      #or C_LEFT_ELBOW in human.body_parts and C_LEFT_SHOULDER in human.body_parts and human.body_parts[C_LEFT_ELBOW].y - human.body_parts[C_LEFT_SHOULDER].y < 0.1)

    right_hand_ind = (C_RIGHT_SHOULDER in human.body_parts and C_RIGHT_WRIST in human.body_parts and human.body_parts[C_RIGHT_SHOULDER].y > human.body_parts[C_RIGHT_WRIST].y)
                     #or C_RIGHT_ELBOW in human.body_parts and C_RIGHT_SHOULDER in human.body_parts and human.body_parts[C_RIGHT_ELBOW].y - human.body_parts[C_RIGHT_SHOULDER].y < 0.1)
    return right_hand_ind, left_hand_ind

def IsDistracted(human):
    #test eye-nose distance rations, tolerances are reflected and opossite
    if (C_RIGHT_EYE in human.body_parts and C_LEFT_EYE in human.body_parts):
        ratio = get_eye_nose_ratio (human.body_parts[C_NOSE], human.body_parts[C_RIGHT_EYE], human.body_parts[C_LEFT_EYE])
        return (ratio < 0.8 or ratio > 1.2)
    else:
        return 0

def IsDistractedSF(human):
    global gl_loaded_model_distracted_sf

    # Init models
    if (not gl_loaded_model_distracted_sf):
        model_path = os.path.dirname(os.path.abspath(__file__))
        gl_loaded_model_distracted_sf = pickle.load(open(model_path + '\\train_mlp3_distracted.sav', 'rb'))
    loaded_model = gl_loaded_model_distracted_sf
    x_test = np.zeros([1, C_BACKGROUND, 2], dtype='float64')
    x_test[0] = human_est_to_numpy(human)

    x_test = x_test.reshape(x_test.shape[0], -1)
    distracted = loaded_model.predict(x_test)

    return distracted

global gl_debug_frame_count

def GetPersonDistance(human, old_body_dist, dst_model='linear', fov_conv_factor=1):#CH: need to get C_CAMERA_CALIB_FOV_D and FOCAL LENGTH from model pickel
    global gl_loaded_model
    global gl_sink_model
    global gl_loaded_model_name
    global gl_debug_frame_count #CH: debug

    # Init models
    if (not gl_loaded_model or gl_loaded_model_name != str(dst_model).lower()):
        gl_loaded_model_name = str(dst_model).lower()
        model_path = os.path.dirname(os.path.abspath(__file__))
        gl_loaded_model = pickle.load(open(model_path+'\\train_'+ gl_loaded_model_name + '_reg.sav', 'rb'))
        gl_sink_model = pickle.load(open('train_algs_sink.sav', 'rb'))
        gl_debug_frame_count = 0
    loaded_model = gl_loaded_model
    sink_model = gl_sink_model

    # Convert human to body_parts vectors
    x_test = np.zeros([1, C_BACKGROUND, 2], dtype='float64')
    x_test[0] = human_est_to_numpy(human)

    # Extract Features and predict
    [x_test_features, _,] = ExtractDataFeatures (
        x_dataset_all=x_test, y_dataset_all=None, num_of_features=C_NUM_OF_FEATURES, features_name=C_FEATURES_NAME) #, default_val=0
    if not x_test_features.shape[1]:
        return 0, -1, -1
    ret_old_body_dist = [None] * x_test_features.__len__()
    for feature in range(x_test_features.__len__()):
        ret_old_body_dist[feature] = prep_data_sink_pred(x_test_features[feature], loaded_model[feature],
                                                         dst_model, fov_conv_factor)
    ret_old_body_dist_max_bp = -1
    ret_old_body_dist = np.transpose(np.array(ret_old_body_dist))
    ret_old_body_dist_avg = pred_data_sink(ret_old_body_dist.tolist(), sink_model, dst_model)
    if (old_body_dist  < ret_old_body_dist_avg and old_body_dist >= 0):
        IsFigApproching = 1
    else:
        IsFigApproching = 0

    # CH: debug!
    '''ret_old_body_dist_avg = np.average(ret_old_body_dist[:int(ret_old_body_dist.shape[0] / 2)])
    gl_debug_frame_count += 1
    if np.abs(ret_old_body_dist_avg- 4) > 0.4:
        print(str(ret_old_body_dist_avg) +  ',' + str(gl_debug_frame_count) + ',')
    '''

    return IsFigApproching, round(ret_old_body_dist_avg[0],2), ret_old_body_dist_max_bp

import UL_GetDistance3_ML
def GetPersonDistanceSF (human, old_body_dist):#CH: need to get C_CAMERA_CALIB_FOV_D and FOCAL LENGTH from model pickel
    global gl_loaded_model_sf
    global gl_sink_model
    global gl_loaded_model_name
    global gl_debug_frame_count #CH: debug

    # Init models
    if (not gl_loaded_model_sf):
        model_path = os.path.dirname(os.path.abspath(__file__))
        gl_loaded_model_sf = pickle.load(open(model_path+'\\train_mlp3_single.sav', 'rb'))
        gl_debug_frame_count = 0
    loaded_model = gl_loaded_model_sf

    x_test = np.zeros([1, C_BACKGROUND, 2], dtype='float64')
    x_test[0] = human_est_to_numpy(human)
    '''human_avg = np.average(x_test[:, :, 0], axis=1)
    x_test[0] = UL_GetDistance3_ML.normalize_human_x(x_test[0], human_avg)'''
    x_test_normelize = UL_GetDistance3_ML.normalize_dataset(x_test)
    x_test_flat = x_test_normelize.reshape(x_test_normelize.shape[0], -1)

    ret_old_body_dist_avg = loaded_model.predict(x_test_flat)

    if (old_body_dist  < ret_old_body_dist_avg and old_body_dist >= 0):
        IsFigApproching = 1
    else:
        IsFigApproching = 0

    return IsFigApproching, round(ret_old_body_dist_avg[0],2), -1


def GetBBox (human, human_body_parts=[], bbox_resize_factor=1):
    [LeftX, TopY] = [1,1]
    [RightX, BottomY] = [0, 0]

    if not human_body_parts:
        human_body_parts = human.body_parts
    for body_part in human_body_parts:
        if not body_part in human.body_parts or body_part == C_RIGHT_WRIST or body_part == C_LEFT_WRIST \
                or body_part == C_RIGHT_ELBOW or body_part == C_LEFT_ELBOW:
            continue
        if (RightX < human.body_parts[body_part].x):
            RightX = human.body_parts[body_part].x * bbox_resize_factor
        if (BottomY < human.body_parts[body_part].y):
            BottomY = human.body_parts[body_part].y * bbox_resize_factor
        if (LeftX > human.body_parts[body_part].x):
            LeftX = human.body_parts[body_part].x / bbox_resize_factor
        if (TopY > human.body_parts[body_part].y):
            TopY = human.body_parts[body_part].y / bbox_resize_factor
    return {'x_left': LeftX, 'y_top': TopY, 'x_right': RightX, 'y_bottom': BottomY, 'resize_factor':bbox_resize_factor}

def get_iou (bbox1, bbox2):
    #parameters check for min < max
    if (bbox1['x_left'] > bbox1['x_right'] or bbox1['y_top'] > bbox1['y_bottom']
    or bbox2['x_left'] > bbox2['x_right'] or bbox2['y_top'] > bbox2['y_bottom']):
        return 0
    '''assert bbox1['x_left'] < bbox1['x_right']
    assert bbox1['y_top'] < bbox1['y_bottom']
    assert bbox2['x_left'] < bbox2['x_right']
    assert bbox2['y_top'] < bbox2['y_bottom']
    '''

    x_left = max(bbox1['x_left'], bbox2['x_left'])
    x_right = min(bbox1['x_right'], bbox2['x_right'])
    y_top = max(bbox1['y_top'], bbox2['y_top'])
    y_bottom = min(bbox1['y_bottom'], bbox2['y_bottom'])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    intersection_area = (x_right-x_left) * (y_bottom-y_top)
    bbox1_area = (bbox1['x_right'] - bbox1['x_left']) * (bbox1['y_bottom'] - bbox1['y_top'])
    bbox2_area = (bbox2['x_right'] - bbox2['x_left']) * (bbox2['y_bottom'] - bbox2['y_top'])
    iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def get_euc_bbox_distance (bbox1, bbox2):
    dist = np.array([bbox1['x_left'], bbox1['x_right'], bbox1['y_top'], bbox1['y_bottom']]) - np.array([bbox2['x_left'], bbox2['x_right'], bbox2['y_top'], bbox2['y_bottom']])
    return np.linalg.norm(dist)

def get_euc_body_parts_distance (body_parts1, body_parts2, distance1, distance2, fov, focal_len, resolution):
    euc_distance = []
    for body_part_idx in range(C_BACKGROUND):
        if body_part_idx not in body_parts1 or body_part_idx not in body_parts2:
            continue
        point1 = [body_parts1[body_part_idx].x, body_parts1[body_part_idx].y]
        point2 = [body_parts2[body_part_idx].x, body_parts2[body_part_idx].y]
        world_coord1 = conv_2d_to_3d(point1, fov, focal_len, resolution, distance1)
        world_coord2 = conv_2d_to_3d(point2, fov, focal_len, resolution, distance2)
        distance_temp = np.linalg.norm(world_coord1 - world_coord2)
        euc_distance.append(distance_temp)

    #take avg of the bottom half (closest) of all keypoints distance
    euc_distance.sort()
    ret_distance = np.average(euc_distance[:int(euc_distance.__len__()/2)])
    return ret_distance
    #and divide it by distance(????)

from mpl_toolkits.mplot3d import Axes3D

'''def plot_person_course(position, ax_3d, color):
    ax_3d.scatter(position[0],
               position[2],
               1, color=color)
    plt.pause(0)'''
def plot_person_course(person_dtct_track, smoother_person_dtct_track):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    fig.gca().set_xlim(0, 1)
    #fig_trendl = plt.figure()
    #ax_trendl = fig_trendl.gca(projection='3d')
    x = [row[0] for row in person_dtct_track]
    y = [row[2] for row in person_dtct_track]
    z = [1 for row in person_dtct_track]
    '''ax.plot([row[0] for row in person_dtct_track],
            [row[2] for row in person_dtct_track],
            [1 for row in person_dtct_track], color='blue')
    '''
    #ax.scatter(x, y, z, color='blue')

    '''
    trendl = np.polyfit(x, y, 2)
    trendl_val = np.poly1d(trendl)
    ax_trendl.plot(x, trendl_val(x), z, "b--")

    '''
    '''ax.plot([row[0] for row in smoother_person_dtct_track],
            [row[2] for row in smoother_person_dtct_track],
            [1 for row in smoother_person_dtct_track], color='yellow')
    '''
    if smoother_person_dtct_track:
        x = [row[0] for row in smoother_person_dtct_track]
        y = [row[2] for row in smoother_person_dtct_track]
        z = [1 for row in smoother_person_dtct_track]
        ax.scatter(x, y, z, color='red')
        '''
        trendl = np.polyfit(x, y, 1)
        trendl_val = np.poly1d(trendl)
        ax_trendl.plot(x, trendl_val(x), z, "y--")
        '''
    #fig_trendl.gca().set_xlim(0, 1)
    #ax.title('Track points')
    ax.set_xlabel('Image width')
    ax.set_ylabel('Distance [m]')
    ax.invert_yaxis()
    #fig.gca().title('Track points')
    #ax.title('Track Trendline')
    '''
    ax_trendl.set_xlabel('Image width')
    ax_trendl.set_ylabel('Distance [m]')
    ax_trendl.invert_yaxis()
    '''
    #plt.gca().invert_yaxis()
    plt.savefig('plots/course.png', bbox_inches='tight')
    # plt.gca().set_ylim(0, 15)
    plt.show()

def find_person_id_by_oui(human, human_bbox, person_list_queue=[], dst_model=0, fov=C_CAMERA_CALIB_FOV_D, focal_len=0, fov_conv_factor = 1, resolution=0, fps=1):
    max_iou = 0
    new_person_old_dist = -1
    min_euc_dist = float('inf')
    person_list_queue.queue.reverse() #reversing queue order to bring newer images from queue (from new to old)
    for person_list_idx, person_list in enumerate(person_list_queue.queue):
        for person_idx, person in enumerate(person_list):
            person.person_detected = 0
            new_bbox = GetBBox(human=person.human, human_body_parts=human.body_parts, bbox_resize_factor=person.bbox['resize_factor'])
            iou_by_bbox = get_iou(human_bbox, person.bbox)
            iou_by_same_body_parts_bbox = get_iou(human_bbox, new_bbox)
            iou = max(iou_by_bbox, iou_by_same_body_parts_bbox)
            euc_dist_by_same_body_parts = get_euc_bbox_distance (human_bbox, new_bbox)
            euc_dist_by_bbox = get_euc_bbox_distance(human_bbox, person.bbox)
            if str(dst_model) != 'sf':
                [_, human_distance, _] = GetPersonDistance(human, 0, dst_model=dst_model, fov_conv_factor=fov_conv_factor)
            else:
                [_, human_distance, _] = GetPersonDistanceSF(human, 0)
            euc_dist_by_body_parts = get_euc_body_parts_distance (human.body_parts, person.human.body_parts, distance1=human_distance, distance2=person.distance,
                                                                  fov=fov, focal_len=focal_len, resolution=resolution)
            euc_dist = min(euc_dist_by_bbox, euc_dist_by_same_body_parts, euc_dist_by_body_parts)
            if max_iou < iou:
                max_iou = iou
                max_iou_idx = person_idx - 1
                max_list_idx = person_list_idx
            if min_euc_dist > euc_dist:
                min_euc_dist = euc_dist
                max_euc_dist_idx = person_idx - 1
                max_list_idx = person_list_idx

    new_person_id = 0
    for person_list_idx, person_list in enumerate(person_list_queue.queue):
        if person_list_idx == max_list_idx:
            break
    if not fps:
        fps = 1
    try:
        if max_iou > C_IOU_MIN_OVRLP:
             new_person_id = person_list[max_iou_idx].PersonId
             new_person_old_dist = person_list[max_iou_idx].distance
        elif min_euc_dist < (C_EUC_MIN_DIST / fps):
            new_person_id = person_list[max_euc_dist_idx].PersonId
            new_person_old_dist = person_list[max_euc_dist_idx].distance
        else:
            print(max_iou, min_euc_dist, C_EUC_MIN_DIST/fps) #CH: debug - don't keep it in prod
    except:
        print('lll')
    person_list_queue.queue.reverse()  # re-reversing queue order to prev condition

    return new_person_id, new_person_old_dist

def plot_person_features(person_dtct=0, image=0):
    if not person_dtct:
        return -1
    image_h, image_w = image.shape[:2]
    factor = int(image_h * 0.01)
    x_min = int(image_w * person_dtct.bbox['x_left']) - 3 * factor
    x_max = int(image_w * person_dtct.bbox['x_right'])
    bbox_center = int((x_min + x_max) / 2)
    y_top = int(image_h * person_dtct.bbox['y_top'] - factor)

    cv2.putText(image, str(person_dtct.PersonId),
                (bbox_center, y_top),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                3)
    if (person_dtct.IsRightHandRaised_Ind):
        cv2.putText(image, 'Right Hand',
                    (x_min, y_top - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2)
    if (person_dtct.IsLeftHandRaised_Ind):
        cv2.putText(image, 'Left Hand',
                    (x_max, y_top - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2)
    if (not person_dtct.IsDistracted_Ind):
        cv2.putText(image, 'Looking at YOU!',
                    (bbox_center, y_top - 3 * factor),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2)
    if (person_dtct.IsApproching_Ind > 0):
        cv2.putText(image, 'Approching',
                    (bbox_center, y_top - 6 * factor),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2)
    cv2.putText(image, str(person_dtct.distance)+'m',
                (bbox_center, y_top + 5 * factor),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 255),
                2)
    #print scaled bbox
    center1 = (int(person_dtct.bbox['x_left'] * image_w), int(person_dtct.bbox['y_top'] * image_h))
    center2 = (int(person_dtct.bbox['x_right'] * image_w), int(person_dtct.bbox['y_bottom'] * image_h))
    cv2.rectangle(image, center1, center2, (255, 255, 255), 2)

    #print unscanled bbox points
    bbox = GetBBox(person_dtct.human, bbox_resize_factor=1)
    center1 = (int(bbox['x_left'] * image_w), int(bbox['y_top'] * image_h))
    center2 = (int(bbox['x_right'] * image_w), int(bbox['y_bottom'] * image_h))
    cv2.circle(image, center1, 2, (255, 255, 255), 2)
    cv2.circle(image, center2, 2, (255, 255, 255), 2)

def set_person_features (person_dtct=0, dst_model=0, fov_conv_factor=1):
    if not person_dtct or not dst_model:
        return -1
    [person_dtct.IsRightHandRaised_Ind, person_dtct.IsLeftHandRaised_Ind] = IsHandRaised(person_dtct.human)
    if str(dst_model) != 'sf':
        person_dtct.IsDistracted_Ind = IsDistracted(person_dtct.human)
        [person_dtct.IsApproching_Ind, person_dtct.distance, bp_max] = GetPersonDistance(human=person_dtct.human, old_body_dist=person_dtct.distance, dst_model=dst_model,
                                                                        fov_conv_factor=fov_conv_factor)
    else:
        person_dtct.IsDistracted_Ind = IsDistractedSF(person_dtct.human)
        #person_dtct.IsDistracted_Ind = IsDistracted(person_dtct.human)
        [person_dtct.IsApproching_Ind, person_dtct.distance, bp_max] = GetPersonDistanceSF(human=person_dtct.human,
                                                                                     old_body_dist=person_dtct.distance)

    '''if person_dtct.distance == 0:
        print ('debug')'''
    person_position = [np.average([person_dtct.bbox['x_left'], person_dtct.bbox['x_right']]),
                       np.average([person_dtct.bbox['y_top'], person_dtct.bbox['y_bottom']]),
                       person_dtct.distance]
    person_dtct.position = copy.deepcopy(person_position)
    return bp_max

def manipulate_image (image, args, resolution_w, resolution_h):
    image_h, image_w = image.shape[:2]
    if (args.rotate):
        center = (image_w / 2, image_h / 2)
        M = cv2.getRotationMatrix2D(center, args.rotate, 1.0)
        image = cv2.warpAffine(image, M, (int(image_w), int(image_h)))
    if (not image_w == resolution_w or not image_h == resolution_h):
        image_scaled = cv2.resize(image, (resolution_w, resolution_h), interpolation=cv2.INTER_AREA)
    else:
        image_scaled = image

    image_scaled = image
    return image_scaled

def Infer_Humans (input, args):
    global gl_est
    # Init model
    if not gl_est:
        gl_est = Est_Object()
        gl_est.scales = ast.literal_eval(args.scales)
        gl_est.infer_res_w, gl_est.infer_res_h = model_wh(args.resolution)
        gl_est.e = TfPoseEstimator(get_graph_path(args.kpt_model), target_size=(gl_est.infer_res_w, gl_est.infer_res_h))
    humans = []
    succ, image = input.read()
    if (succ):
        #[frame_w, frame_h] = image.shape[:2]
        image_scaled = manipulate_image(image, args, gl_est.infer_res_w, gl_est.infer_res_h)
        humans = gl_est.e.inference(image_scaled,  scales=gl_est.scales)

    return humans, image, succ

def Init_Input(args):
    if (args.camera_fov):
        camera_fov = args.camera_fov
    else:
        camera_fov = C_CAMERA_CALIB_FOV_D

    if (args.camera_focal_len):
        camera_focal_len = args.camera_focal_len
    else:
        camera_focal_len = C_CAMERA_CALIB_FOCAL_LEN
    if (args.input == ''):
        args.input = 0

    if (is_number(args.input)):
        video = cv2.VideoCapture(int(args.input))
        fps = 30
    elif (args.input):
        video = cv2.VideoCapture(args.input)
        fps = video.get(cv2.CAP_PROP_FPS)

    frame_w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fov_conv_factor = calc_fov_conv_factor (camera_fov)

    return video, camera_fov, camera_focal_len, fov_conv_factor, fps, frame_h, frame_w

def Finit_Input(input):
    input.release()


def merge_dict_fill(d1, d2):
    result = dict(d1)
    for k, v in d2.items():
        if k not in result:
            result.update({k: v})
    return result

def merge_dict_add_weights(d1, d2, w): #d1 is new person body parts to add, d2 is the person body parts history to smooth with (itr)
    result = copy.deepcopy(dict(d1))
    #result = {}
    for k,v in d2.items():
        if k not in result:
            x = copy.deepcopy(v.x)
            y = copy.deepcopy(v.y)
            score = copy.deepcopy(v.score)
        else:
            temp_x = (result[k].x * result[k].score + v.x * v.score * w)
            temp_y = (result[k].y * result[k].score + v.y * v.score * w)
            temp_score = (result[k].score + v.score * w)
            x = copy.deepcopy(temp_x / temp_score)
            y = copy.deepcopy(temp_y / temp_score)
            #x = copy.deepcopy((result[k].x + v.x * w) / (1+w))
            #y = copy.deepcopy((result[k].y + v.y * w) / (1+w))
            score = copy.deepcopy(temp_score / (1+w))
        result.update({k: BodyPart(v.uidx, v.part_idx, x, y, score)})
    return result

def avg_body_parts(body_parts, denum=1):
    for k in body_parts.keys():
        body_parts[k].x /= denum
        body_parts[k].y /= denum
        body_parts[k].score /= denum

    return body_parts


def smooth_human(person_dtct, person_list_queue):
    #smoother_person_dtct = Person(PersonId=1)
    #smoother_person_dtct.human.body_parts = copy.deepcopy(person_dtct.human.body_parts)
    if (not person_list_queue.full()):
        return person_dtct
    smoother_person_dtct = Person(PersonId=person_dtct.PersonId, human=Human([]), person_detected=1)
    #find specific person (detected person) from history (queue) and avg it by history
    num_person_dtct_in_queue = 0
    weights = np.arange(start=0, stop=1, step=1 / person_list_queue.qsize())#CH: think about diff weights for better results
    #weights = np.poly()
    #weights = np.ones(person_list_queue.qsize())
    #w_sum = 0
    for person_list in person_list_queue.queue:
        #w = 0
        for person in person_list:
            if person.PersonId == person_dtct.PersonId:
                smoother_person_dtct.human.body_parts = merge_dict_add_weights(person_dtct.human.body_parts, person.human.body_parts, weights[num_person_dtct_in_queue])
                num_person_dtct_in_queue += 1
                #smoother_person_dtct.human.body_parts = merge_dict_fill(person_dtct.human.body_parts, person.human.body_parts)
                break
    #avg deteced person list by weights
    #person_dtct.human.body_parts = avg_body_parts (person_dtct.human.body_parts, w_sum)
    if not smoother_person_dtct.human.body_parts:
        return person_dtct
    else:
        return smoother_person_dtct