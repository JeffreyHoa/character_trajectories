# Copyright 2017 Jeffrey Hoa. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================


from collections import OrderedDict as odict

import numpy as np
import scipy.io as scio
from sklearn import mixture
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BaseNB
from sklearn.preprocessing import LabelBinarizer


def MaxMinNormalization(x, Max, Min):
    x = (x - Min) / (Max - Min);
    return x;


def refinePoints(x):
    #############################
    # Cut zero in trace
    #############################
    x_list_cutted = np.trim_zeros(x)

    #############################
    # Interpolation
    #############################
    x_interpolation = []

    for idx in range(len(x_list_cutted) - 1):
        interval = (x_list_cutted[idx + 1] - x_list_cutted[idx]) / g_timeSeqNormalize_len

        for i in range(g_timeSeqNormalize_len):
            x_interpolation.append(x_list_cutted[idx] + interval * i)

    x_interpolation.append(x_list_cutted[-1])

    x_norm = []
    sampling_invertal = len(x_list_cutted) - 1

    for idx in range(len(x_interpolation)):
        if (idx % sampling_invertal == 0):
            x_norm.append(x_interpolation[idx])

    return x_norm


def preprocess_norm(X):
    X_norm = np.zeros((X.shape[0], 3, g_num_check_points))

    for i, x in enumerate(X):
        X_norm[i] = [refinePoints(_) for _ in x]

    return X_norm


class checkPointMGaussianMixture(BaseNB):
    def __init__(self, n_classes, g_num_check_points):
        self.n_classes = n_classes
        self.g_num_check_points = g_num_check_points

    def fit(self, X, y):
        self.classes_ = np.arange(self.n_classes, dtype=int)
        self.class_prior_ = y.mean(axis=0)
        models = odict()
        for label_idx in range(self.n_classes):
            sub_models = odict()

            # choose one time point
            for check_idx in range(self.g_num_check_points):
                x_3d_list_gmm = []

                # check this time point through all relevant samples
                for sample_idx in range(len(X)):
                    sample_keyIdx = y[sample_idx].tolist().index(1)
                    if (sample_keyIdx == label_idx):
                        # Get one 3d point:
                        # the horizontal and vertical axes, and the pen tip force.
                        x_3d_points = X[sample_idx].T
                        x_3d_list_gmm.append(x_3d_points[check_idx])

                gmm = mixture.GaussianMixture(n_components=1, covariance_type='full')
                gmm.fit(x_3d_list_gmm)

                '''
                For one label, each sub-models contains <count of time points> gmms.
                '''
                sub_models[check_idx] = gmm

            '''
            For whole model, it contains <count of labels> sub-models
            '''
            models[label_idx] = sub_models

        self.models = models
        return self

    def _joint_log_likelihood(self, X):
        joint_log_likelihood = []

        for sample_idx in range(len(X)):
            trace_3d_points = X[sample_idx].T
            total_log_likelihood_list = []

            for label_index in range(self.n_classes):
                log_likelihood_list = []

                for checkPoint_idx in range(self.g_num_check_points):
                    gmm = self.models[label_index][checkPoint_idx]
                    log_likelihood = gmm.score(trace_3d_points[checkPoint_idx].reshape(1, -1))
                    log_likelihood_list.append(log_likelihood)

                total_log_likelihood_list.append(sum(log_likelihood_list[1:-1])/(g_num_check_points-2))

            joint_log_likelihood.append(total_log_likelihood_list)

        joint_log_likelihood = np.array(joint_log_likelihood)

        return joint_log_likelihood

    def predict(self, X):
        multiclass_pred = super(checkPointMGaussianMixture, self).predict(X)
        one_hot_pred = np.zeros((X.shape[0], self.n_classes))
        one_hot_pred[np.arange(X.shape[0]), multiclass_pred.ravel()] = 1
        return one_hot_pred


def train_model(xtrain, ytrain, cnt_class):
    checkPointM = checkPointMGaussianMixture(cnt_class, g_num_check_points)
    checkPointM.fit(xtrain, ytrain)

    return checkPointM


def predictive_performance(xdata, ydata, class_pred):
    correct = np.zeros(xdata.shape[0])

    for i, x in enumerate(xdata):
        correct[i] = np.all(ydata[i, :] == class_pred[i, :])
    accuracy = correct.mean()

    return accuracy

g_timeSeqNormalize_len = 99  # 99 intervals --> 100 points.
g_num_check_points = g_timeSeqNormalize_len + 1

if __name__ == '__main__':
    #################################
    # Load training data
    #################################
    train_file = '../data/trajectories_train.mat'
    test_file = '../data/trajectories_xtest.mat'
    train_data = scio.loadmat(train_file)
    test_data = scio.loadmat(test_file)

    xtrain = train_data['xtrain'][0]  # [sample]
    ytrain = train_data['ytrain'][0]  # [sample]
    ktrain = train_data['key'][0]  # [classes]

    cnt_sample = len(xtrain)
    cnt_class = len(ktrain)

    xtrain_norm = preprocess_norm(xtrain)
    ytrain = LabelBinarizer().fit_transform(np.reshape(ytrain, (-1, 1)) - 1)
    # Train-Test spliting
    xtrain_norm, xtest_norm, ytrain, ytest = train_test_split(xtrain_norm, ytrain, test_size=0.2, random_state=0)
    cnt_train = len(xtrain_norm)
    cnt_test = len(xtest_norm)

    checkPointM = train_model(xtrain_norm, ytrain, cnt_class)
    pred = checkPointM.predict(xtest_norm)

    accuracy = predictive_performance(xtest_norm, ytest, pred)
    print('Average test accuracy=' + str(accuracy))
