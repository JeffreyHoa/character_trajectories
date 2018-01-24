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
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.naive_bayes import BaseNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer


def preprocess_mat(X, mat_size=20):
    return np.array([to_mat(_, mat_size) for _ in X])


def generate_point_sequence(data, mat_size):
    points = [(0, 0)]
    for vx, vy, df in data:
        x, y = points[-1]
        points.append((x + vx, y - vy))
        if df < -1:
            break

    points = np.array(points)
    # translation and scaling
    points -= np.min(points, axis=0)
    points *= (mat_size - 1) / np.max(points)
    # centring
    points += (mat_size - np.max(points, axis=0)) * 0.5
    # type casting
    int_points = points.astype(np.int32)

    return int_points


def to_mat(data, mat_size):
    points = generate_point_sequence(data.T, mat_size)
    img = np.zeros((mat_size, mat_size))
    for x, y in points:
        img[y, x] = 1

    return np.reshape(img, (mat_size * mat_size))


class CharCondGaussianMixture(BaseNB):
    def __init__(self, n_components=10, covariance_type='diag', n_init=1, n_classes=20):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_init = n_init
        self.n_classes = n_classes

    def fit(self, X, y):
        self.classes_ = np.arange(self.n_classes, dtype=int)
        self.class_prior_ = y.mean(axis=0)
        class_cond_models = []

        for i in range(len(self.classes_)):
            this_idx = y[:, i] == 1
            this_x = X[this_idx, :]
            this_y = y[this_idx, :]
            this_model = GaussianMixture(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                n_init=self.n_init,
            )
            this_model.fit(this_x, this_y)
            class_cond_models.append(this_model)

        self.class_cond_models = class_cond_models
        return self

    def _joint_log_likelihood(self, X):
        joint_log_likelihood = []
        for i in range(np.size(self.classes_)):
            jointi = np.log(self.class_prior_[i])
            l_ij = self.class_cond_models[i].score_samples(X)
            joint_log_likelihood.append(jointi + l_ij)

        joint_log_likelihood = np.array(joint_log_likelihood).T
        return joint_log_likelihood

    def predict(self, X):
        multiclass_pred = super(CharCondGaussianMixture, self).predict(X)
        one_hot_pred = np.zeros((X.shape[0], self.n_classes))
        one_hot_pred[np.arange(X.shape[0]), multiclass_pred.ravel()] = 1
        return one_hot_pred


def train_model(xtrain, ytrain, covariance_type='full', n_init=1):
    pca = PCA(
        svd_solver='randomized',
        n_components=10)
    nbgmm = CharCondGaussianMixture(
        covariance_type=covariance_type,
        n_init=n_init,
        n_components=10
    )
    pipe = Pipeline(steps=[('pca', pca), ('nbgmm', nbgmm)])
    # Prediction
    n_pca_components = np.arange(10, 50)
    n_mixture_components = np.arange(1, 30)

    estimator = RandomizedSearchCV(
        pipe,
        cv=5,
        param_distributions=dict(
            pca__n_components=n_pca_components,
            nbgmm__n_components=n_mixture_components
        ),
        n_iter=50,
        scoring='accuracy',
        random_state=0
    )
    estimator.fit(xtrain, ytrain)

    return estimator


def predictive_performance(xdata, ydata, class_pred):
    correct = np.zeros(xdata.shape[0])

    for i, x in enumerate(xdata):
        correct[i] = np.all(ydata[i, :] == class_pred[i, :])

    accuracy = correct.mean()

    return accuracy


if __name__ == '__main__':
    train_file = '../data/trajectories_train.mat'
    test_file  = '../data/trajectories_xtest.mat'
    train_data = scio.loadmat(train_file)
    test_data  = scio.loadmat(test_file)

    xtrain = train_data['xtrain'][0]
    ytrain = train_data['ytrain'][0]
    key = train_data['key'][0]

    # preprocessing
    # restoring x
    xtrain = preprocess_mat(xtrain)
    # print(FunctionTransformer(func=preprocess_mat).fit_transform(xtrain))
    # one-hot encoding for y
    ytrain = LabelBinarizer().fit_transform(np.reshape(ytrain, (-1, 1)) - 1)

    # train-test spliting
    xtrain, xtest, ytrain, ytest = train_test_split(xtrain, ytrain, test_size=0.2, random_state=0)

    # training
    model = train_model(xtrain, ytrain)

    # prediction and evaluation
    class_pred = model.predict(xtest)
    accuracy = predictive_performance(xtest, ytest, class_pred)
    print('Average test accuracy=' + str(accuracy))

    # display mis-classifications
    """
    for i in range(len(class_pred)):
        if np.argmax(class_pred[i]) != np.argmax(ytest[i]):
            print('true', key[np.argmax(ytest[i])], 'pred', key[np.argmax(class_pred[i])])
            show_mat(xtest[i])
    """
