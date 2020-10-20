import warnings
import sys

warnings.filterwarnings('ignore')
sys.path.append('../src')

import numpy as np
import time
from utils import bags_to_2D
from sklearn_transformers import LeadLag, AddTime
import tslearn.metrics as metrics
from sklearn.model_selection import GridSearchCV, train_test_split
from gaKer_fast import GA_kernels_mmd
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


from joblib import Parallel, delayed
from tslearn.utils import to_time_series, to_time_series_dataset, ts_size, \
    check_equal_size


def model(X, y, ll=None, at=False, mode='krr', NUM_TRIALS=5, cv=3, grid={}, lambdas=[1.]):
    """Performs a DR-GA kernel-based distribution regression on ensembles (of possibly unequal cardinality)
       of univariate or multivariate time-series.

       Input:
              X (list): list of lists such that

                        - len(X) = n_samples

                        - for any i, X[i] is a list of arrays of shape (length, dim)

                        - for any j, X[i][j] is an array of shape (length, dim)

              y (np.array): array of shape (n_samples,)

              ll (list of ints): dimensions to lag
              at (bool): if True pre-process the input path with add-time

              mode (str): "krr" -> Kernel Ridge Regression, 'svr' -> Support Vector Regresion

              NUM_TRIALS, cv : parameters for nested cross-validation

       Output: mean MSE (and std) (both scalars) of regression performance on a cv-folds cross-validation (NUM_TRIALS times)
    """

    assert mode in ['svr', 'krr'], "mode must be either 'svr' or 'krr' "

    # possibly augment the state space of the time series
    if ll is not None:
        X = LeadLag(ll).fit_transform(X)
    if at:
        X = AddTime().fit_transform(X)

    if mode == 'krr':
        parameters = {'clf__kernel': ['precomputed'], 'clf__alpha': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3],
                      'rbf_ga__gamma': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]}

        # check if the user has not given an irrelevant entry
        assert len(list(set(parameters.keys()) & set(grid.keys()))) == len(
            list(grid.keys())), "keys should be in " + ' '.join([str(e) for e in parameters.keys()])
        # merge the user grid with the default one
        parameters.update(grid)
        # print(parameters)
        clf = KernelRidge

    else:
        parameters = {'clf__kernel': ['precomputed'], 'clf__C': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3],
                      'rbf_ga__gamma': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]}

        # check if the user has not given an irrelevant entry
        assert len(list(set(parameters.keys()) & set(grid.keys()))) == len(
            list(grid.keys())), "keys should be in " + ' '.join(
            [str(e) for e in parameters.keys()])

        # merge the user grid with the default one
        parameters.update(grid)

        clf = SVR

    ''' Parallelisation: Here we precompute and cythonize the whole gram matrix computation like we do for KES. '''
    list_kernels = []
    for n,lambda_ in enumerate(lambdas):
       K_full = np.zeros((len(X), len(X)),dtype=np.float64)
       indices = np.triu_indices(len(X),k=0,m=len(X))
       K_full[indices] = Parallel(n_jobs=-1,verbose=3)(
           delayed(ExpectedKernel)(X[i],X[j],sigma=1, gamma=lambda_)
           for i in range(len(X))
           for j in range(i,len(X))
       )
       indices = np.tril_indices(len(X), k=-1, m=len(X))
       K_full[indices] = K_full.T[indices]
       diag = np.diag(K_full)

       mmd = -2. * K_full + np.tile(diag,(K_full.shape[0],1)) + np.tile(diag[:,None],(1,K_full.shape[0]))
       list_kernels.append(mmd)

    # Loop for each trial
    scores = np.zeros(NUM_TRIALS)
    results = {}

    for i in range(NUM_TRIALS):

        best_scores_train = np.zeros(len(lambdas))

        # will only retain the MSE (mean + std) corresponding to the model achieving the best score (on the train set)
        # i.e. the test set is not used to decide the hyperparameters

        MSE_test = np.zeros(len(lambdas))

        for n, lambda_ in enumerate(lambdas):
            results_tmp = {}
            if not np.isnan(list_kernels[n]).any():
                pipe = Pipeline([('rbf_ga', RBF_GA_Kernel(K_full=list_kernels[n])),
                                 ('clf', clf())])

                X_train, X_test, y_train, y_test = train_test_split(np.arange(len(y)), y, test_size=0.2, random_state=i)

                # parameter search
                model = GridSearchCV(pipe, parameters, verbose=0, n_jobs=-1, scoring='neg_mean_squared_error', cv=cv,
                                     error_score=np.nan)

                model.fit(X_train, y_train)

                best_scores_train[n] = -model.best_score_

                y_pred = model.predict(X_test)
                results_tmp[n] = {'pred': y_pred, 'true': y_test}
                MSE_test[n] = mean_squared_error(y_pred, y_test)
            else:
                best_scores_train[n] = 100000
                MSE_test[n] = 100000
        # pick the model with the best performances on the train set
        best_score = 100000
        index = 0
        for n, lambda_ in enumerate(lambdas):
            if (best_scores_train[n] < best_score):
                best_score = best_scores_train[n]
                index = n

        scores[i] = MSE_test[index]
        results[i] = results_tmp[index]
        print('best lambda (cv on the train set): ', lambdas[index])
    return scores.mean(), scores.std(), results


class RBF_GA_Kernel(BaseEstimator, TransformerMixin):
    def __init__(self, K_full=None, gamma=1.0):
        super(RBF_GA_Kernel, self).__init__()
        self.gamma = gamma
        self.K_full = K_full

    def transform(self, X):
        alpha = 1. / (2 * self.gamma ** 2)
        K = self.K_full[X][:, self.ind_train].copy()
        return np.exp(-alpha * K)  

    def fit(self, X, y=None, **fit_params):
        self.ind_train = X
        return self


def ExpectedKernel(bag_i, bag_j, sigma=1, gamma=1):
    K_ij = GA_kernels_mmd(np.array(bag_i,dtype=np.float64), np.array(bag_j,dtype=np.float64), gamma=sigma)
    K_ij = np.exp(-gamma * np.array(K_ij))
    return np.mean(K_ij)


