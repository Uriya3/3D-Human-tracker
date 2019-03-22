import numpy as np
import copy
import math as math
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import uniform as sp_rand
import matplotlib.pyplot as plt
import argparse
import h5py
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

def normalize_human(human, human_x_avg, human_y_avg):
    if np.isnan(human_x_avg).any():
        human_x_avg = 0
    if np.isnan(human_y_avg).any():
        human_y_avg = 0
    human_ = np.array(human)
    for bp_indx, body_part in enumerate(human_):
        if body_part.any():
            #human_[bp_indx][0] = math.fabs(body_part[0] - human_avg)
            human_[bp_indx][0] = body_part[0] - human_x_avg
        else:
            human_[bp_indx][1] = body_part[1] + human_y_avg

    '''plt.scatter(human_[:, 0], human_[:, 1], c='b')
    plt.scatter(human[:, 0], human[:, 1], c='r')'''
    return human_


def normalize_dataset(X_dataset):
    X_dataset_normalize = np.empty(X_dataset.shape)
    humans_x_not_zeros = np.count_nonzero(X_dataset[:, :, 0], axis=1)
    humans_xs_sums = np.sum(X_dataset[:, :, 0], axis=1)
    humans_x_avgs = np.divide(humans_xs_sums, humans_x_not_zeros)
    humans_y_not_zeros = np.count_nonzero(X_dataset[:, :, 1], axis=1)
    humans_ys_sums = np.sum(X_dataset[:, :, 1], axis=1)
    humans_y_avgs = np.divide(humans_ys_sums, humans_y_not_zeros)

    # humans_avgs = np.count_nonzero(X_dataset[:, :, 0], axis=1)
    for human_indx, human in enumerate(X_dataset):
        X_dataset_normalize[human_indx] = normalize_human(human, humans_x_avgs[human_indx], humans_y_avgs[human_indx])
    return X_dataset_normalize

def get_metrics (model, X_dataset_test, Y_dataset_test):
    y_pred = model.predict(X_dataset_test)
    r2 = r2_score(Y_dataset_test[:], y_pred[:])
    var = explained_variance_score(Y_dataset_test[:], y_pred)
    mse = mean_squared_error(Y_dataset_test[:], y_pred)
    mae = mean_absolute_error(Y_dataset_test[:], y_pred)
    model_loss = model.loss_
    acc_score = model.score(X_dataset_test, Y_dataset_test)

    return [r2, var, mse, mae, model_loss, acc_score]


def train_and_eval(file, plot=0):
    # Init vars
    with h5py.File(file, 'r') as hf:
        X_dataset = hf.get('X_Train_All')
        Y_dataset = hf.get('Y_Train_All')
        rand_percentage = np.arange(0, 0.5, 0.05)
        num_of_pers = rand_percentage.__len__() #one before and one for end
        #rand_percentage = [0]
        model_loss = np.zeros(num_of_pers, dtype='float64')
        r2 = np.zeros(num_of_pers, dtype='float64')
        var = np.zeros(num_of_pers, dtype='float64')
        mse = np.zeros(num_of_pers, dtype='float64')
        mae = np.zeros(num_of_pers, dtype='float64')
        acc_score = np.zeros(num_of_pers, dtype='float64')

        model_loss_ = np.zeros(num_of_pers, dtype='float64')
        r2_ = np.zeros(num_of_pers, dtype='float64')
        var_ = np.zeros(num_of_pers, dtype='float64')
        mse_ = np.zeros(num_of_pers, dtype='float64')
        mae_ = np.zeros(num_of_pers, dtype='float64')
        acc_score_ = np.zeros(num_of_pers, dtype='float64')

        # Normalize and flatten Data type
        X_dataset_normalize = normalize_dataset(X_dataset)
        # Normilize
        #scaler = MinMaxScaler(feature_range=(0, 1))
        #X_dataset_normalize[:,:,0] = scaler.fit_transform(X_dataset_normalize[:,:,0])
        #X_dataset_normalize[:, :, 0] = X_dataset_normalize[:,:,0] + 0.5
        #X_dataset_normalize[:, :, 1] = scaler.fit_transform(X_dataset_normalize[:, :, 1])

        X_dataset_flat = np.array(X_dataset_normalize).reshape(X_dataset_normalize.shape[0], -1)

        #X_dataset_flat = np.array(X_dataset).reshape(X_dataset.shape[0], -1)
        #Y_dataset_flat = np.array([math.log(y, 10) for y in Y_dataset])
        Y_dataset_flat = np.array(Y_dataset)


        # Split and train for first time to find best estimator
        X_dataset_all_train, X_dataset_all_test, Y_dataset_all_train, Y_dataset_all_test = \
            train_test_split(X_dataset_flat, Y_dataset_flat, test_size=0.20, random_state=42)

        model = MLPRegressor(hidden_layer_sizes=10, max_iter=200, warm_start=True, random_state=0)
        # Grid Search:
        [_, model] = UL_GetDistance_ML.reg_common_RS_CV(model, X_dataset_all_train, Y_dataset_all_train)

        print (get_metrics(model, X_dataset_all_test, Y_dataset_all_test))

        # Run per missing percentage - Retrain
        for per_indx, per in enumerate(rand_percentage):
            X_Per = random_change_array (X_dataset, per, per, np.zeros(2))
            X_Per_normelize = normalize_dataset(X_Per)

            #X_Per = random_change_array(X_dataset, per, per, np.zeros(2))
            X_Per_flat = X_Per_normelize.reshape(X_Per_normelize.shape[0], -1)
            # Split, Scale, Train and test
            X_dataset_train, X_dataset_test, Y_dataset_train, Y_dataset_test = \
                train_test_split(X_Per_flat, Y_dataset_flat, test_size=0.20, random_state=42)

            # Training - Grid Search
            #model.partial_fit(X_dataset_train, Y_dataset_train)
            model.fit(X_dataset_train, Y_dataset_train)

            #[_, model] = UL_GetDistance_ML.reg_common_RS_CV(model, X_dataset_train, Y_dataset_train)

            # Testing on per
            [r2[per_indx], var[per_indx], mse[per_indx], mae[per_indx] ,model_loss[per_indx], acc_score[per_indx]] =\
                get_metrics (model, X_dataset_test, Y_dataset_test)

            # Testing on "real" data
            [r2_[per_indx], var_[per_indx], mse_[per_indx], mae_[per_indx], model_loss_[per_indx], acc_score_[per_indx]] = \
                get_metrics(model, X_dataset_all_test, Y_dataset_all_test)

            if acc_score_[per_indx] < 0.92:
                print (str(per) + ' Under score: ' + str(acc_score[per_indx]))
                break

            best_model = copy.deepcopy(model)

    '''[r2[num_of_pers-1], var[num_of_pers-1], mse[num_of_pers-1], mae[num_of_pers-1], model_loss[num_of_pers-1], acc_score[num_of_pers-1]] = \
        get_metrics(model, X_dataset_all_test, Y_dataset_all_test)
    '''
    # Print metrics
    print('Pers:            ' + str(rand_percentage))
    print('R2:              ' + str(r2[r2.nonzero()]))
    print('R2_:             ' + str(r2_[r2.nonzero()]))
    print('VAR:             ' + str(var[var.nonzero()]))
    print('VAR_:            ' + str(var_[var_.nonzero()]))
    print('MSE:             ' + str(mse[mse.nonzero()]))
    print('MSE_:            ' + str(mse_[mse_.nonzero()]))
    print('MAE:             ' + str(mae[mae.nonzero()]))
    print('MAE_:            ' + str(mae_[mae_.nonzero()]))
    print('LOSS:            ' + str(model_loss[model_loss.nonzero()]))
    print('LOSS_:           ' + str(model_loss_[model_loss_.nonzero()]))
    print('Accuracy Score:  ' + str(acc_score[acc_score.nonzero()]))
    print('Accuracy Score_: ' + str(acc_score_[acc_score_.nonzero()]))

    pickle.dump(best_model, open('train_mlp3_single.sav', 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UL_GetDistanceAndMore_ML')
    parser.add_argument('--h5_file', type=str, default='')
    args = parser.parse_args()
    sys.stdout = pose_est_ind.Logger('Logs\_MLAndMore.log')

    train_and_eval (args.h5_file, 1)
    plt.show()