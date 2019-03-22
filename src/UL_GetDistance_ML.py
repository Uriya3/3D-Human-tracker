import numpy as np
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from scipy.stats import uniform as sp_rand
import matplotlib.pyplot as plt
import argparse
import pickle
import pose_estimation_ind as pose_est_ind
import sys

def shuffle_in_unison(a,b):
    np.random.seed(0)
    rng_state = np.random.get_state()
    #a_permute = np.random.permutation(a)
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    #b_permute = np.random.permutation(b)
    np.random.shuffle(b)
    #return a_permute, b_permute

def scale_continuous_vars(data_array):
    if ((data_array.ndim) < 2):
        return data_array
    i = 0
    data_min = np.zeros(data_array.shape[1])
    data_max = np.zeros(data_array.shape[1])
    data_array_t = np.transpose(data_array)
    for row in data_array_t:
        data_min[i] = min(row)
        data_max[i] = max(row)
        i+=1
    for row in range(data_array_t.shape[0]):
        for feature in range(data_array_t.shape[1]):
            data_array_t[row,feature] = (data_array_t[row,feature] - data_min[row]) / (data_max[row] - data_min[row])
    return np.transpose(data_array_t)


def reg_common (regr, x_train, y_train, x_test, y_test, plot=0, alg_name=''):
    print('#####################################')
    print (alg_name+':')
    r2 = []
    var = []
    mse = []
    mae = []
    '''for feature in range(x_train.shape[1]):
        regr[feature].fit(x_train[:, feature, None], y_train[:, feature])
        y_pred = regr[feature].predict(x_test[:, feature, None])
        print(np.transpose(np.array([y_test[:, feature], y_pred, y_test[:, feature] - y_pred])))
    '''
    feature = 0
    regr[0].fit(x_train, y_train)
    y_pred = regr[0].predict(x_test)
    print(np.transpose(np.array([y_test, y_pred, y_test-y_pred, 100*np.abs(y_test-y_pred)/y_test])))

    '''if hasattr(regr, 'coef_'):
        print('Coefficients: \n', regr.coef_)'''
    r2.insert(feature, r2_score(y_test[1:], y_pred[1:]))
    #print('r_2 statistic: ' + str(r2))
    var.insert(feature, explained_variance_score(y_test[:], y_pred))
    #print('var statistic:  ' + str(var))
    mse.insert(feature, mean_squared_error(y_test[:], y_pred))
    mae.insert(feature, mean_absolute_error(y_test[:], y_pred))
    #print("Mean squared error: " + str(mse))
    '''if plot:
        # Plot outputs
        fig = plt.figure()
        fig.clear()
        #plt.gcf().clear()
        plt.scatter(x_test[:,0], y_test, color='black')
        plt.plot(x_test[:,0], y_pred, color='blue', linewidth=3)
        plt.xticks(())
        plt.yticks(())
        plt.title(alg_name+str(feature))
        plt.savefig(alg_name+str(feature))
        fig.show()
    '''
    r2 = np.average(np.array(r2))
    var = np.average(np.array(var))
    mse = np.average(np.array(mse))
    mae = np.average(np.array(mae))
    return [regr, r2, var, mse, mae]

def reg_common_RS_CV (regr, x_train, y_train, plot=0):
    param_grid = {'alpha': sp_rand()}
    rsearch = RandomizedSearchCV(estimator=regr, param_distributions=param_grid, verbose=10, random_state=42) #n_iter=100
    rsearch.fit(x_train, y_train)
    #print ('best score:' + str(rsearch.best_score_))
    #print('best alpha: ' + str(rsearch.best_estimator_.alpha))
    return [rsearch.best_score_, rsearch.best_estimator_]

def train_lasso_reg(x_train, y_train, x_test, y_test, plot=0):
    #print('LASSO: ')
    regr = linear_model.Lasso()
    [_, regr] = reg_common_RS_CV(regr, x_train, y_train)
    #regr = [linear_model.Lasso(alpha=best_est.alpha)]
    return reg_common ([regr], x_train, y_train, x_test, y_test, plot, 'LASSO')

def train_svr_lin_reg(x_train, y_train, x_test, y_test, plot=0):
    #print('SVR: ')
    #params = [{'C': [0.01, 0.05, 0.1, 1]}, {'n_estimators': [10, 100, 1000]}]
    regr = [SVR()]
    '''cv = []
    for param in params:
        best_m = GridSearchCV(regr, param)
        best_m.fit(x_train, y_train)
        s = mean_squared_error(y_train, best_m.predict(x_train))
        cv.append(s)
    print(np.mean(cv))'''
    return reg_common(regr, x_train, y_train, x_test, y_test, plot, 'SVR')

def train_ridge_reg(x_train, y_train, x_test, y_test, plot=0):
    #print('RIDGE: ')
    regr = Ridge()
    [_, regr] = reg_common_RS_CV (regr, x_train, y_train)
    #regr = [Ridge(alpha=best_est.alpha)]
    #regr = linear_model.RidgeCV(alphas=np.arange(0.01, 5, 0.5))#, cv=None, fit_intercept=True, scoring=None, normalize=False)#(alphas=0.05, cv=10)
    return reg_common([regr], x_train, y_train, x_test, y_test, plot, 'RIDGE')

def train_mlp_reg(x_train, y_train, x_test, y_test, plot=0):
    #print('MLP: ')
    regr = [MLPRegressor(hidden_layer_sizes=2)]
    #[_, best_est] = reg_common_RS_CV(regr, x_train, y_train)
    #regr = MLPRegressor(alpha=best_est.alpha)
    return reg_common(regr, x_train, y_train, x_test, y_test, plot, 'MLP')


def train_linear_reg(x_train, y_train, x_test, y_test, plot=0):
    #print('LINEAR: ')
    regr = [linear_model.LinearRegression()]
    return reg_common(regr, x_train, y_train, x_test, y_test, plot, 'LINEAR')

def train_poly_reg(x_train, y_train, x_test, y_test, plot=0):
    #print('POLY: ')
    poly_deg = 2
    regr = [make_pipeline(PolynomialFeatures(poly_deg),MLPRegressor(hidden_layer_sizes=2))]
    return reg_common(regr, x_train, y_train, x_test, y_test, plot, 'POLY')

def train_affine(x_train, y_train, x_test, y_test, plot=0):
    #affine_train = np.divide(y_train, x_train)
    affine_train = np.divide(1 / y_train, x_train[:,0])
    affine_train_avg = np.average(affine_train)
    #affine_test = np.divide(y_test, x_test)
    #r2 = r2_score(affine_train[:affine_test.shape[0]], affine_test)
    #var = explained_variance_score (affine_train[:affine_test.shape[0]], affine_test)
    #mse = mean_squared_error (affine_train[:affine_test.shape[0]], affine_test)
    r2=1
    var=0
    mse=0
    return affine_train_avg, r2, var, mse

'''
def train_sink (model, x_train, y_train, x_test, y_test):
    model_dist = []
    for feature in :
        if (model.lower() == 'aff'):
            model_dist.insert(feature, 1 / (model[feature] * x_test[feature]))
        else:
            model_dist.insert(feature, model[feature][0].predict(x_test[feature]))
     
    train_mlp_reg (model_dist, y_train)
    '''

def train_and_eval(file, plot=0):#CH: need to save C_CAMERA_CALIB_FOV_D and FOCAL LENGTH to model pickel
    # Init vars
    algs = pose_est_ind.C_DIST_ALGS_NAMES
    hf = pickle.load(open(file, "rb"))
    num_of_features = hf['NUM_OF_FEATURES']
    features_names = hf['FEATURES_NAMES']
    r2 = [None] * num_of_features
    mse = [None] * num_of_features
    var = [None] * num_of_features
    mae = [None] * num_of_features
    model_svr = [None] * num_of_features
    model_ridge = [None] * num_of_features
    model_linear = [None] * num_of_features
    model_lasso = [None] * num_of_features
    model_mlp = [None] * num_of_features
    model_poly = [None] * num_of_features
    model_affine = [None] * num_of_features

    # Train and Eval each feature model
    for feature in range(num_of_features):
        print(features_names[feature])
        X_dataset = hf.get('X_Train_No_Missing' + str(feature))
        Y_dataset = hf.get('Y_Train' + str(feature))
        X_dataset_train, X_dataset_test, Y_dataset_train, Y_dataset_test = \
            train_test_split(X_dataset, Y_dataset, test_size=0.20, random_state=42)

        r2[feature] = np.zeros(algs.__len__())
        var[feature] = np.zeros(algs.__len__())
        mse[feature] = np.ones(algs.__len__())
        mae[feature] = np.ones(algs.__len__())
        i = 0

        # Plot outputs
        if plot:
            fig = plt.figure()
            fig.clear()
            # plt.gcf().clear()
            jet = plt.get_cmap('jet')
            colors_features = iter(jet(np.linspace(0, 1, num_of_features)))
            feature_color = next(colors_features)
            plt.scatter(X_dataset_test[:,0], Y_dataset_test, color=feature_color, label=str(features_names[feature]))
            # plot feature avg blobs
            #avg_results = pose_est_ind.get_feature_avg(X_dataset_test, Y_dataset_test)
            avg_results = []
            for y in sorted(np.unique(Y_dataset_test)):
                avg_results.append([np.average(X_dataset_test[np.where(Y_dataset_test == y)][:,0],axis=0), y])
            avg_results = np.array(avg_results)

            plt.scatter(avg_results[:, 0], avg_results[:, 1], color=feature_color, label='feature_avg', s=15 ** 2)

            plt.legend()
            plt.savefig('plots/ml_' + str(features_names[feature]) + '.png', bbox_inches='tight')
            #plt.show()

        # Train data
        [model_svr[feature], r2[feature][i], var[feature][i], mse[feature][i], mae[feature][i]] = train_svr_lin_reg(
            X_dataset_train, Y_dataset_train, X_dataset_test, Y_dataset_test, plot)
        i += 1

        [model_ridge[feature], r2[feature][i], var[feature][i], mse[feature][i], mae[feature][i]] = train_ridge_reg(
            X_dataset_train, Y_dataset_train, X_dataset_test, Y_dataset_test, plot)     ### CV Checked
        i += 1

        [model_linear[feature], r2[feature][i], var[feature][i], mse[feature][i], mae[feature][i]] = train_linear_reg(
            X_dataset_train, Y_dataset_train, X_dataset_test, Y_dataset_test, plot)
        i += 1

         ### CV Checked
        [model_lasso[feature], r2[feature][i], var[feature][i], mse[feature][i], mae[feature][i]] = train_lasso_reg(
            X_dataset_train, Y_dataset_train, X_dataset_test, Y_dataset_test, plot)
        i += 1

        [model_mlp[feature], r2[feature][i], var[feature][i], mse[feature][i], mae[feature][i]] = train_mlp_reg(
            X_dataset_train, Y_dataset_train, X_dataset_test, Y_dataset_test, plot)
        i += 1

        [model_poly[feature], r2[feature][i], var[feature][i], mse[feature][i], mae[feature][i]] = train_poly_reg(
            X_dataset_train, Y_dataset_train, X_dataset_test, Y_dataset_test, plot)
        i += 1

        [model_affine[feature], r2[feature][i], var[feature][i], mse[feature][i]] = train_affine(
            X_dataset_train, Y_dataset_train, X_dataset_test, Y_dataset_test,
            plot)
        i += 1

    # Save feature model
    pickle.dump(model_svr, open('train_svr_reg.sav', 'wb'))
    pickle.dump(model_ridge, open('train_ridge_reg.sav', 'wb'))
    pickle.dump(model_linear, open('train_linear_reg.sav', 'wb'))
    pickle.dump(model_lasso, open('train_lasso_reg.sav', 'wb'))
    pickle.dump(model_mlp, open('train_mlp_reg.sav', 'wb'))
    pickle.dump(model_poly, open('train_poly_reg.sav', 'wb'))
    pickle.dump(model_affine, open('train_aff_reg.sav', 'wb'))

    # Show metrics
    for feature in range(num_of_features):
        for alg in range(algs.__len__()):
            print(algs[alg]+str(feature) + ' R2: ' + str(r2[feature][alg]) + ' Var: ' + str(var[feature][alg]) + ' MSE: ' + str(mse[feature][alg]) + ' MAE: ' + str(mae[feature][alg]))

    r2_avg = []
    var_avg = []
    mse_avg = []
    mae_avg = []
    for alg_indx in range(algs.__len__()):
        r2_avg.append(np.average(r2[alg_indx]))
        var_avg.append(np.average(var[alg_indx]))
        mse_avg.append(np.average(mse[alg_indx]))
        mae_avg.append(np.average(mae[alg_indx]))

    print('Best R2:     ' + algs[np.argmax(r2_avg)] +  ' Value: ' + str(np.max(r2_avg)))
    print('Best VAR:    ' + algs[np.argmax(var_avg)] + ' Value: ' + str(np.max(var_avg)))
    print('Best MSE:    ' + algs[np.argmin(mse_avg)] + ' Value: ' + str(np.min(mse_avg)))
    print('Best MAE:    ' + algs[np.argmin(mae_avg)] + ' Value: ' + str(np.min(mae_avg)))
    plt.show()

def create_sink_model (file):
    print('Creating distance sink model...')
    # Init vars
    algs = pose_est_ind.C_DIST_ALGS_NAMES
    fov_conv_factor = pose_est_ind.calc_fov_conv_factor(pose_est_ind.C_CAMERA_CALIB_FOV_D)
    hf = pickle.load(open(file, "rb"))
    num_of_features = hf['NUM_OF_FEATURES']
    model_algs_sink = [None] * algs.__len__()
    r2_sink = np.zeros(algs.__len__())
    var_sink = np.zeros(algs.__len__())
    mse_sink = np.ones(algs.__len__())
    mae_sink = np.ones(algs.__len__())

    # Sink for each alg type
    for alg_indx, alg in enumerate(algs):
        print(alg)
        # Init and load model for each alg type
        model = pickle.load(open('train_' + alg.lower() + '_reg.sav', 'rb'))
        f_dists = open('Logs\dists_ML_' + alg + '.csv', 'ab')
        dist_pred = [None] * num_of_features
        dist_real = [None] * num_of_features
        ret_train_body_dist = [None] * num_of_features
        ret_test_body_dist = [None] * num_of_features

        # Prepare train data for each feature model
        for feature in range(num_of_features):
            X_dataset = hf.get('X_Train_No_Missing' + str(feature))
            Y_dataset = hf.get('Y_Train' + str(feature))

            X_dataset_train, _, Y_dataset_train, _= \
                train_test_split(X_dataset, Y_dataset, test_size=0.20, random_state=42)

            ret_train_body_dist[feature] = [None] * X_dataset_train.shape[0]
            for x_i, x_features in enumerate(X_dataset_train):
                x_features = x_features.reshape(1,-1)
                ret_train_body_dist[feature][x_i] = pose_est_ind.prep_data_sink_pred(x_features, model[feature], alg, fov_conv_factor)

            if np.isnan(ret_train_body_dist[feature]).any():
                print('CH:')


        # Fix data type
        x_data_train = np.transpose(np.array(ret_train_body_dist))
        x_data_train = x_data_train.reshape(x_data_train.shape[1:])
        y_data_train = np.array(Y_dataset_train)

        # Train model
        model_algs_sink[alg_indx] = MLPRegressor(hidden_layer_sizes=1)
        [_, model_algs_sink[alg_indx]] = reg_common_RS_CV(model_algs_sink[alg_indx], x_data_train, y_data_train)
        model_algs_sink[alg_indx].fit(x_data_train, y_data_train)
        '''# Free Mem
        x_data_train = []
        y_data_train = []
        '''
        # Prepare test data for each feature model
        for feature in range(num_of_features):
            X_dataset = hf.get('X_Train_Missing' + str(feature))
            Y_dataset = hf.get('Y_Train' + str(feature))

            [X_dataset_test, Y_dataset_test] = [X_dataset, Y_dataset]

            ret_test_body_dist[feature] = [None] * X_dataset_test.shape[0]
            for x_i, x_features in enumerate(X_dataset_test):
                x_features = x_features.reshape(1, -1)
                ret_test_body_dist[feature][x_i] = pose_est_ind.prep_data_sink_pred(x_features, model[feature], alg, fov_conv_factor)

        # Fix data type
        x_data_test = np.transpose(np.array(ret_test_body_dist))
        x_data_test = x_data_test.reshape(x_data_test.shape[1:])
        y_data_test = np.array(Y_dataset_test)

        # Test model
        y_pred = pose_est_ind.pred_data_sink(x_data_test.tolist(), model_algs_sink, alg)

        # Print diff between real to pred and get metrics
        print(np.transpose(np.array([y_data_test, y_pred, y_data_test - y_pred, 100 * np.abs(y_data_test - y_pred) / y_data_test])))
        plt.plot(100 * np.abs(y_data_test - y_pred) / y_data_test)
        r2_sink[alg_indx] = r2_score(y_data_test, y_pred)
        var_sink[alg_indx] = explained_variance_score(y_data_test, y_pred)
        mse_sink[alg_indx] = mean_squared_error(y_data_test, y_pred)
        mae_sink[alg_indx] = mean_absolute_error(y_data_test, y_pred)

        plt.savefig('plots/ml_sink_' + alg + '.png', bbox_inches='tight')
        np.savetxt(f_dists, np.transpose([y_data_test, y_pred]), delimiter=",")


    # Print metrics
    print('Name:    ' + str(pose_est_ind.C_DIST_ALGS_NAMES))
    print('R2:      ' + str(r2_sink))
    print('VAR:     ' + str(var_sink))
    print('MSE:     ' + str(mse_sink))
    print('MAE:     ' + str(mae_sink))
    # Print Best metrics
    print('Best sink R2:     ' + algs[np.argmax(r2_sink)] + ' Value: ' + str(np.max(r2_sink)))
    print('Best sink VAR:    ' + algs[np.argmin(var_sink)] + ' Value: ' + str(np.min(var_sink)))
    print('Best sink MSE:    ' + algs[np.argmin(mse_sink)] + ' Value: ' + str(np.min(mse_sink)))
    print('Best sink MAE:    ' + algs[np.argmin(mae_sink)] + ' Value: ' + str(np.min(mae_sink)))

    # Dump sink model
    pickle.dump(model_algs_sink, open('train_algs_sink.sav', 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UL_GetDistance_ML')
    parser.add_argument('--h5_file', type=str, default='')
    #parser.add_argument('--shuffle', type=int, default=1)
    #parser.add_argument('--normalize', type=int, default=1)
    args = parser.parse_args()
    sys.stdout = pose_est_ind.Logger('Logs\_ML.log')
    #args.h5_file = 'face_keypoints_dataset_Featuresmobilenet_thin20180924-0249.pickle'
    train_and_eval (args.h5_file, 1)
    create_sink_model (args.h5_file)
    plt.show()