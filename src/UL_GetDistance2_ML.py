import numpy as np
import math as math
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from scipy.stats import uniform as sp_rand
import matplotlib.pyplot as plt
import argparse
import pickle
import pose_estimation_ind as pose_est_ind
import UL_GetDistance_ML as UL_GetDistance_ML
import sys


def find_nearest_notnan(array):
    array = np.asarray(array)

    for val in array:
        if not np.isnan(val).any():
            return val
    avg = np.nanmean(array, axis=0)
    return avg


def find_nearest_notzero(array):
    array = np.asarray(array)

    for val in array:
        if not val.any():
            return val
    avg = np.nanmean(array, axis=0)
    return avg




def random_change_array (array, frame_percent, feature_percent, value):
    array = np.asarray(array)
    for frame_indx, frame in enumerate(array):
        rand_frame = np.random.uniform(0,1)
        if rand_frame <= frame_percent:
            for feature_index, feature in enumerate(frame):
                rand_feature = np.random.normal()
                if rand_feature <= feature_percent:
                    array[frame_indx][feature_index] = value

    return array

def train_and_eval(file, plot=0):
    # Init vars
    hf = pickle.load(open(file, "rb"))
    num_of_features = hf['NUM_OF_FEATURES']
    num_of_subfeatures = pose_est_ind.C_NUM_OF_SUBFEATURES#hf['NUM_OF_SUBFEATURES']
    features_names = hf['FEATURES_NAMES']
    num_of_frames = hf['NUM_OF_FRAMES']
    X_dataset = np.empty((num_of_frames,num_of_features, num_of_subfeatures), dtype='float64')
    #Y_dataset = np.empty((num_of_frames, num_of_features, num_of_subfeatures), dtype='float64')
    Y_dataset = np.empty((num_of_frames, ), dtype='float64')
    rand_percentage = [0.1, 0.25, 0.5]
    model_loss = np.zeros(rand_percentage.__len__(), dtype='float64')
    r2 = np.zeros(rand_percentage.__len__(), dtype='float64')
    var = np.zeros(rand_percentage.__len__(), dtype='float64')
    mse = np.zeros(rand_percentage.__len__(), dtype='float64')
    mae = np.zeros(rand_percentage.__len__(), dtype='float64')

    # Load data & Complete missing val by neighbours vals
    for feature in range(num_of_features):
        print(features_names[feature])
        X_dataset_feature = hf.get('X_Train_No_Missing' + str(feature))
        Y_dataset_feature = hf.get('Y_Train' + str(feature))

        for y in sorted(np.unique(Y_dataset_feature)):
            y_first_occr = np.argmin(Y_dataset_feature < y)
            y_last_occr = np.argmax(Y_dataset_feature > y)
            if not y_last_occr:
                y_last_occr = X_dataset_feature.shape[0]
            X_dataset_feature_y = X_dataset_feature[y_first_occr:y_last_occr]

            for X_indx, X in enumerate(X_dataset_feature_y):
                if np.isnan(X).any():
                    fill_X_prev = np.empty(num_of_subfeatures)
                    fill_X_prev[:] = np.nan
                    fill_X_next = np.empty(num_of_subfeatures)
                    fill_X_next[:] = np.nan

                    fill_X_prev = find_nearest_notnan(X_dataset_feature[:X_indx - 1])
                    fill_X_next = find_nearest_notnan(X_dataset_feature[X_indx + 1:])
                    X_dataset_feature[y_first_occr+X_indx] = np.nanmean(np.array([fill_X_prev, fill_X_next]), axis=0)
                if not X.any():
                    fill_X_prev = np.empty(num_of_subfeatures)
                    fill_X_prev[:] = np.nan
                    fill_X_next = np.empty(num_of_subfeatures)
                    fill_X_next[:] = np.nan
                    fill_X_prev = find_nearest_notzero(X_dataset_feature[:X_indx - 1])
                    fill_X_next = find_nearest_notzero(X_dataset_feature[X_indx + 1:])
                    X_dataset_feature[y_first_occr + X_indx] = np.nanmean(np.array([fill_X_prev, fill_X_next]), axis=0)

        if np.isnan(X_dataset_feature).any():
            print('Debug: Still nan!')

        X_dataset[:,feature,:] = X_dataset_feature
        #Y_dataset[:,feature,:] = X_dataset_feature

    Y_dataset[:] = Y_dataset_feature
    #ReTrain - Randomly select sub feature data, set it to zeros
    X_dataset_flat = X_dataset.reshape(X_dataset.shape[0], -1)
    #Y_dataset_flat = Y_dataset.reshape(Y_dataset.shape[0], -1)
    #Y_dataset_flat = np.append(Y_dataset_flat, Y_dataset_feature).reshape(Y_dataset.shape[0], -1)

    X_dataset_train, X_dataset_test, Y_dataset_train, Y_dataset_test = \
        train_test_split(X_dataset_flat, Y_dataset, test_size=0.20, random_state=42)
    # Scaling data to reduce MLP sensitivity
    #scalar = StandardScaler()
    #scalar.partial_fit(X_dataset_train)
    #X_dataset_train = scalar.transform(X_dataset_train)

    model = MLPRegressor(hidden_layer_sizes=5, max_iter=200, random_state=42)
    # Grid Search:
    [_, model] = UL_GetDistance_ML.reg_common_RS_CV(model, X_dataset_train, Y_dataset_train)

    # Run per missing precentage
    for per_indx, per in enumerate(rand_percentage):
        X_Per = random_change_array (X_dataset, per, per, np.zeros(num_of_subfeatures))
        X_Per_flat = X_Per.reshape(X_Per.shape[0], -1)
        # Split, Scale, Train and test
        X_dataset_train, X_dataset_test, Y_dataset_train, Y_dataset_test = \
            train_test_split(X_Per_flat, Y_dataset, test_size=0.20, random_state=42)
        #X_dataset_train = scalar.transform(X_dataset_train)
        #X_dataset_test = scalar.transform(X_dataset_test)

        # Training
        model.fit(X_dataset_train, Y_dataset_train)

        # Testing
        y_pred = model.predict(X_dataset_test)
        r2[per_indx] = r2_score(Y_dataset_test[:], y_pred[:])
        var[per_indx] = explained_variance_score(Y_dataset_test[:], y_pred)
        mse[per_indx] = mean_squared_error(Y_dataset_test[:], y_pred)
        mae[per_indx] = mean_absolute_error(Y_dataset_test[:], y_pred)
        model_loss[per_indx] = model.loss_
        '''
        [model, r2, var, mse, mae] = \
            UL_GetDistance_ML.reg_common(model, X_dataset_train, Y_dataset_train, X_dataset_test, Y_dataset_test, 0, 'MLP')
        '''

    # Print metrics
    print('R2: ' + str(r2))
    print('VAR: ' + str(var))
    print('MSE: ' + str(mse))
    print('MAE: ' + str(mae))
    print('LOSS: ' + str(model_loss))

    pickle.dump(model, open('train_mlp2_single.sav', 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UL_GetDistanceAndMore_ML')
    parser.add_argument('--h5_file', type=str, default='')
    args = parser.parse_args()
    sys.stdout = pose_est_ind.Logger('Logs\_MLAndMore.log')

    train_and_eval (args.h5_file, 1)
    plt.show()