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


import numpy as np
import scipy.io as scio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.externals import joblib
import matplotlib.pyplot as pl
from scipy.misc import logsumexp
from scipy.stats import mode

import gmm
import checkPoint


def ensemble_predict(X_mat, model_gmm, X_norm, model_checkPoint):
    class_prob_gmm = model_gmm.predict_log_proba(X_mat)
    class_prob_checkPoint = model_checkPoint._joint_log_likelihood(X_norm)
    # normalisation
    for i in range(class_prob_checkPoint.shape[0]):
        class_prob_checkPoint[i] -= logsumexp(class_prob_checkPoint[i])

    n_x = X_mat.shape[0]
    n_classes = class_prob_gmm[0].shape[0]
    class_prob_ensemble = np.ndarray(shape=[n_x, n_classes], dtype=np.float32)
    pred = np.zeros((n_x, n_classes))
    w = [0.5, 0.5]

    for i in range(n_x):
        for j in range(n_classes):
            class_prob_ensemble[i][j] = class_prob_gmm[i][j] * w[0] + class_prob_checkPoint[i][j] * w[1]

        pred[i][np.argmax(class_prob_ensemble[i])] = 1

    return np.array(pred), class_prob_ensemble, class_prob_gmm, class_prob_checkPoint


def ensemble_predict_top_n(X_mat, model_gmm, X_norm, model_checkPoint):
    class_prob_gmm = model_gmm.predict_log_proba(X_mat)
    class_prob_checkPoint = model_checkPoint._joint_log_likelihood(X_norm)
    # normalisation
    for i in range(class_prob_checkPoint.shape[0]):
        class_prob_checkPoint[i] -= logsumexp(class_prob_checkPoint[i])

    n_x = X_mat.shape[0]
    n_classes = class_prob_gmm[0].shape[0]
    pred = np.zeros((n_x, n_classes))

    for i in range(n_x):
        gmm_top_n = np.argsort(class_prob_gmm[i])[-1:][::-1]
        checkPoint_top_n = np.argsort(class_prob_checkPoint[i])[-1:][::-1]
        y = gmm_top_n if class_prob_gmm[i][gmm_top_n] > class_prob_checkPoint[i][checkPoint_top_n] else checkPoint_top_n
        pred[i][y] = 1

    return np.array(pred), class_prob_gmm, class_prob_checkPoint

def predictive_performance(ydata, class_pred):
    correct = np.zeros(ydata.shape[0])

    for i in range(ydata.shape[0]):
        correct[i] = np.all(ydata[i, :] == class_pred[i, :])
    accuracy = correct.mean()

    return accuracy


def show_mat(mat, mat_size):
    pl.imshow(np.reshape(mat, (mat_size, mat_size)), cmap='gray')
    pl.show()


if __name__ == '__main__':
    train_file = '../data/trajectories_train.mat'
    test_file = '../data/trajectories_xtest.mat'
    train_data = scio.loadmat(train_file)
    test_data = scio.loadmat(test_file)

    xtrain = train_data['xtrain'][0]
    ytrain = train_data['ytrain'][0]
    key = train_data['key'][0]

    # preprocessing
    # one-hot encoding for y
    ytrain = LabelBinarizer().fit_transform(np.reshape(ytrain, (-1, 1)) - 1)

    # train-test spliting
    xtrain, xtest, ytrain, ytest = train_test_split(xtrain, ytrain, test_size=0.2, random_state=0)

    xtrain_mat = gmm.preprocess_mat(xtrain)
    xtrain_norm = checkPoint.preprocess_norm(xtrain)
    xtest_mat = gmm.preprocess_mat(xtest)
    xtest_norm = checkPoint.preprocess_norm(xtest)
    np.savez_compressed('data.npz',
                        xtrain_mat=xtest_mat,
                        xtrain_norm=xtrain_norm,
                        xtest_mat=xtest_mat,
                        xtest_norm=xtest_norm,
                        ytrain=ytrain,
                        ytest=ytest,
                        key=key)
    # load_data = np.load('data.npz')
    # xtrain_mat = load_data['xtrain_mat']
    # xtrain_norm = load_data['xtrain_norm']
    # ytrain = load_data['ytrain']
    # xtest_mat = load_data['xtest_mat']
    # xtest_norm = load_data['xtest_norm']
    # ytest = load_data['ytest']
    # key = load_data['key']
    n_classes = len(key)

    # training
    model_gmm = gmm.train_model(xtrain_mat, ytrain)
    model_checkPoint = checkPoint.train_model(xtrain_norm, ytrain, n_classes)

    joblib.dump(model_gmm, 'gmm.pkl')
    joblib.dump(model_checkPoint, 'checkPoint.pkl')

    # model_gmm = joblib.load('gmm.pkl')
    # model_checkPoint = joblib.load('checkPoint.pkl')

    # prediction and evaluation
    pred, prob, prob_gmm, prob_checkPoint = ensemble_predict(xtest_mat, model_gmm, xtest_norm, model_checkPoint)
    # pred, prob_gmm, prob_checkPoint = ensemble_predict_top_n(xtest_mat, model_gmm, xtest_norm, model_checkPoint)
    pred_gmm = model_gmm.predict(xtest_mat)
    pred_checkPoint = model_checkPoint.predict(xtest_norm)

    accuracy = predictive_performance(ytest, pred)
    print('Average test accuracy=' + str(accuracy))

    for i in range(len(pred)):
        if np.argmax(pred[i]) != np.argmax(ytest[i]):
            print('true: {}\npred-ens: {} (log-prob: {})\npred-gmm: {} (log-prob: {})\npred-ckp: {} (log-prob: {})'.format(
                key[np.argmax(ytest[i])],
                # key[np.argmax(pred[i])], 0,
                key[np.argmax(pred[i])], prob[i][np.argmax(pred[i])],
                key[np.argmax(pred_gmm[i])], prob_gmm[i][np.argmax(pred_gmm[i])],
                key[np.argmax(pred_checkPoint[i])], prob_checkPoint[i][np.argmax(pred_checkPoint[i])]))
            show_mat(xtest_mat[i], 20)
