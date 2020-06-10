import numpy as np
from tqdm import tqdm as tqdm

import warnings
warnings.filterwarnings('ignore')

from sklearn_transformers import AddTime, LeadLag, ExpectedSignatureTransform

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


def model(X, y, depths=[2], ll=None, at=False, mode='krr', NUM_TRIALS=5, cv=3, grid={}):
    """Performs a kernel based distribution regression on ensembles (of possibly unequal cardinality)
       of univariate or multivariate time-series (of possibly unequal lengths)

       Input:
              X (list): list of lists such that

                        - len(X) = n_samples

                        - for any i, X[i] is a list of arrays of shape (length, dim)

                        - for any j, X[i][j] is an array of shape (length, dim)

              y (np.array): array of shape (n_samples,)

              depths (list of ints): signature levels to cross-validate
              ll (list of ints): dimensions to lag (set to None by default)
              at (bool): if True pre-process the input path with add-time

              mode (str): "krr" -> Kernel Ridge Regression, 'svr' -> Support Vector Regresion

              NUM_TRIALS, cv : parameters for cross-validation

              grid (dict): a dictionary to specify the hyperparameter grid for the gridsearch. Unspecified entries will be set by default

       Output: mean MSE (and std) (both scalars) of regression performance on a cv-folds cross-validation (NUM_TRIALS times)
    """

    assert mode in ['svr', 'krr'], "mode must be either 'svr' or 'krr' "

    if X[0][0].shape[1] == 1:
        assert ll is not None or at == True, "must add one dimension to the time-series, via ll=[0] or at=True"
        
    # possibly augment the state space of the time series
    if ll is not None:
        X = LeadLag(ll).fit_transform(X)
    if at:
        X = AddTime().fit_transform(X)

    if mode == 'krr':

        # default grid
        parameters = {'clf__kernel': ['rbf'],
                      'clf__gamma': [gamma(1e-3), gamma(1e-2), gamma(1e-1), gamma(1), gamma(1e1), gamma(1e2),
                                     gamma(1e3)],
                      'clf__alpha': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3],
                      'std_scaler': [None, StandardScaler()]}

        # check if the user has not given an irrelevant entry
        assert len(list(set(parameters.keys()) & set(grid.keys()))) == len(
            list(grid.keys())), "keys should be in " + ' '.join([str(e) for e in parameters.keys()])

        # merge the user grid with the default one
        parameters.update(grid)

        clf = KernelRidge

    else:

        # default grid
        parameters = [{'clf__kernel': ['rbf'],
                       'clf__gamma': [gamma(1e-3),gamma(1e-2), gamma(1e-1), gamma(1), gamma(1e1), gamma(1e2),gamma(1e3)], 
                       'clf__C': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3],
                       'std_scaler': [None, StandardScaler()]
                       }]

        # check if the user has not given an irrelevant entry
        assert len(list(set(parameters.keys()) & set(grid.keys()))) == len(
            list(grid.keys())), "keys should be in " + ' '.join([str(e) for e in parameters.keys()])

        # merge the user grid with the default one
        parameters.update(grid)
        clf = SVR

    # building the estimator
    pipe = Pipeline([('std_scaler', StandardScaler()),
                     ('clf', clf())
                     ])

    scores = np.zeros(NUM_TRIALS)

    # Loop for each trial
    for i in tqdm(range(NUM_TRIALS)):

        best_scores_train = np.zeros(len(depths))

        # will only retain the MSE (mean + std) corresponding to the model achieving the best score (on the train set)
        # i.e. the test set is not used to decide the truncation level.
        # the truncation level cannot be placed in a pipeline otherwise the signatures are recomputed even when not needed

        MSE_test = np.zeros(len(depths))

        for n, depth in enumerate(depths):
            
            ES = ExpectedSignatureTransform(order=depth).fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(np.array(ES), np.array(y), test_size=0.2,
                                                                random_state=i)

            # parameter search
            model = GridSearchCV(pipe, parameters, verbose=0, n_jobs=-1, scoring='neg_mean_squared_error', cv=cv,
                                 error_score=np.nan)

            model.fit(X_train, y_train)
            best_scores_train[n] = -model.best_score_

            y_pred = model.predict(X_test)
            MSE_test[n] = mean_squared_error(y_pred, y_test)

        # pick the model with the best performances on the train set
        best_score = 100000
        index = None
        for n, depth in enumerate(depths):
            if (best_scores_train[n] < best_score):
                best_score = best_scores_train[n]
                index = n

        scores[i] = MSE_test[index]
        print('best truncation level (cv on the train set): ', depths[index])
    return scores.mean(), scores.std()


def gamma(l):
    '''
        The rbf kernel from sklearn is parametrized by gamma:
                k_rbf(x,x')=exp(-gamma||x-x'||^2)
        For our implementation of RBF-RBF we used another parametrization
                k_rbf(x,x')=exp(-(1/2(l^2))||x-x'||^2)
        This function transforms ell into gamma.
        gamma = 1/(2l^2)
    '''
    return 1. / (2 * l ** 2)