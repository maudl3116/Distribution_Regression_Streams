import numpy as np
from tqdm import tqdm as tqdm

import warnings
warnings.filterwarnings('ignore')

import iisignature
from utils.addtime import AddTime, LeadLag

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV, ShuffleSplit, cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


# epsilon=0 -> Kernel Ridge Regression
# epsilon>0 -> SVM (Regression)
SVM_parameters = [{'svm__kernel': ['rbf'], 'svm__epsilon':[0., 0.1],
                   'svm__gamma': [1e-6, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e6, 'auto'], 
                   'svm__C': [1e-6, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e6]}]


def poly_SVM(degree, X, y, ll=None, at=False, ss=False, cv=3):
    
    """Performs a poly(degree)-SVM distribution regression on ensembles (of possibly unequal size) 
       of univariate or multivariate time-series equal of unequal lengths 
    
       Input: degree (int): degree of the polynomial feature map
       
              X (list): list of lists such that
              
                        - len(X) = n_samples
                        
                        - for any i, X[i] is a list of arrays of shape (length, dim)
                        
                          !!! note that all arrays in the list must have same length and same dim !!!
                          
                        - for any j, X[i][j] is an array of shape (length, dim)
                        
              y (np.array): array of shape (n_samples,)
              
              ll (list of ints): dimensions to lag
              at (bool): if True pre-process the input path with add-time
              ss (bool): if True StandardScale the ESignature features
              
              cv (int or list): if int it performs a cross-validation with cv train-test random splits,
                                otherwise, it needs to be a list tuples representing train-test splits  
              
       Output: mean MSE (and std) (both scalars) of regression performance on a 5-fold cross-validation
    
    """
    
    X_poly = []
    
    # take polynomial feature map
    for bag in tqdm(X):
        if ll is not None:
            bag = LeadLag(ll).fit_transform(bag)
        if at:
            bag = AddTime().fit_transform(bag)
        bag = [x.reshape(-1) for x in bag]
        X_poly.append(PolynomialFeatures(degree).fit_transform(bag).mean(0))
        
    X_poly = np.array(X_poly)
                
    # building poly-SVM estimator
    pipe = Pipeline([('std_scaler', StandardScaler()), ('svm', SVR())])
    
    # set-up grid-search over cv random or pre-specified folds
    clf = GridSearchCV(pipe, SVM_parameters, verbose=1, n_jobs=-1, scoring='neg_mean_squared_error', cv=cv)
    
    # find best estimator via grid search 
    clf.fit(X_poly, y)
    
#     score = cross_val_score(clf.best_estimator_, X, mus, scoring='neg_mean_squared_error')
    if not isinstance(cv, int):
        cv = len(cv)    
    scores = np.zeros(cv)
    for i in range(cv):
        scores[i] = np.max(clf.cv_results_['split'+str(i)+'_test_score'])
    
    return -scores.mean(), scores.std()


def ESig_SVM(depth, X, y, ll=[0], at=True, ss=False, cv=3):
    
    """Performs a ESig(depth)-SVM distribution regression on ensembles (of possibly unequal size) 
       of univariate or multivariate time-series equal of unequal lengths 
    
       Input: depth (int): truncation of the signature
       
              X (list): list of lists such that
              
                        - len(X) = n_samples
                        
                        - for any i, X[i] is a list of arrays of shape (length, dim)
                                                  
                        - for any j, X[i][j] is an array of shape (length, dim)
                        
              y (np.array): array of shape (n_samples,)
              
              ll (list of ints): dimensions to lag
              at (bool): if True pre-process the input path with add-time
              ss (bool): if True StandardScale the ESignature features
              
              cv (int or list): if int it performs a cross-validation with cv train-test random splits,
                                otherwise, it needs to be a list tuples representing train-test splits 
              
       Output: mean MSE (and std) (both scalars) of regression performance on a 5-fold cross-validation
    
    """
    
    X_esig = []
    
    # take Esig feature map
    for bag in tqdm(X):
        if ll is not None:
            bag = LeadLag(ll).fit_transform(bag)
        if at:
            bag = AddTime().fit_transform(bag)
        try:
            bag = iisignature.sig(bag, depth)
        except:
            bag = np.array([iisignature.sig(p, depth) for p in bag])
        if bag.shape[0]>0:
            X_esig.append(bag.mean(0))
        
    X_esig = np.array(X_esig)
                    
    # building ESig-SVM estimator
    if ss:
        pipe = Pipeline([('std_scaler', StandardScaler()), ('svm', SVR())])
    else:
        pipe = Pipeline([('svm', SVR())])

    # set-up grid-search over cv random or pre-specified folds
    clf = GridSearchCV(pipe, SVM_parameters, verbose=1, n_jobs=-1, scoring='neg_mean_squared_error', cv=cv)

    # find best estimator via grid search 
    clf.fit(X_esig, y)
    
#     score = cross_val_score(clf.best_estimator_, X, mus, scoring='neg_mean_squared_error')
    if not isinstance(cv, int):
        cv = len(cv)    
    scores = np.zeros(cv)
    for i in range(cv):
        scores[i] = np.max(clf.cv_results_['split'+str(i)+'_test_score'])
    
    return -scores.mean(), scores.std()


def SigESig_LinReg(depth1, depth2, X, y, ll=[0], at=False, ss=False, cv=3):
    
    """Performs a SigESig(depth)-Linear distribution regression on ensembles (of possibly unequal size) 
       of univariate or multivariate time-series equal of unequal lengths 
    
       Input: depth1 (int): truncation of the signature 1
              depth2 (int): truncation of the signature 2
       
              X (list): list of lists such that
              
                        - len(X) = n_samples
                        
                        - for any i, X[i] is a list of arrays of shape (length, dim)
                                                  
                        - for any j, X[i][j] is an array of shape (length, dim)
                        
              y (np.array): array of shape (n_samples,)
              
              ll (list of ints): dimensions to lag
              at (bool): if True pre-process the input path with add-time
              ss (bool): if True StandardScale the ESignature features
              
              cv (int or list): if int it performs a cross-validation with cv train-test random splits,
                                otherwise, it needs to be a list tuples representing train-test splits 
              
       Output: mean MSE (and std) (both scalars) of regression performance on a 5-fold cross-validation
    
    """
    
    
    X_sigEsig = []
    
    # take sigEsig feature map
    for bag in tqdm(X):
        intermediate = []
        for path in bag:
            if ll is not None:
                path = LeadLag(ll).fit_transform([path])[0]
            if at:
                path = AddTime().fit_transform([path])[0]
            sig_path = iisignature.sig(path, depth1, 2) 
            intermediate.append(sig_path)
        
        try:
            intermediate = iisignature.sig(intermediate, depth2)
        except:
            intermediate_intermediate = []
            for p in intermediate:
                try:
                    intermediate_intermediate.append(iisignature.sig(p, depth2))
                except:
                    pass
            intermediate = np.array(intermediate_intermediate)
            
        X_sigEsig.append(intermediate.mean(0))
    X_sigEsig = np.array(X_sigEsig)
    
    # parameters for grid search 
    parameters = [{'lin_reg__alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2], 
                   'lin_reg__fit_intercept' : [True, False], 
                   'lin_reg__normalize' : [True, False]}]
                        
    # building pathwise-Esig estimator
    if ss:
        pipe = Pipeline([('std_scaler', StandardScaler()),('lin_reg', Lasso(max_iter=1000))])
    else:
        pipe = Pipeline([('lin_reg', Lasso(max_iter=1000))])

    # set-up grid-search over cv random or pre-specified folds
    clf = GridSearchCV(pipe, parameters, verbose=1, n_jobs=-1, scoring='neg_mean_squared_error', cv=cv)

    # find best estimator via grid search 
    clf.fit(X_sigEsig, y)
    
#     score = cross_val_score(clf.best_estimator_, X, mus, scoring='neg_mean_squared_error')
    if not isinstance(cv, int):
        cv = len(cv)    
    scores = np.zeros(cv)
    for i in range(cv):
        scores[i] = np.max(clf.cv_results_['split'+str(i)+'_test_score'])
    
    return -scores.mean(), scores.std()


def RBF_RBF_SVM(X, y, ll=None, at=False, ss=True, cv=3):
    
    """Performs a RBF-RBF-SVM distribution regression on ensembles (of possibly unequal size)
       of univariate or multivariate time-series equal of unequal lengths

       Input:
              X (list): list of lists such that

                        - len(X) = n_samples

                        - for any i, X[i] is a list of arrays of shape (length, dim)

                        - for any j, X[i][j] is an array of shape (length, dim)

              y (np.array): array of shape (n_samples,)
              
              ll (list of ints): dimensions to lag
              at (bool): if True pre-process the input path with add-time
              ss (bool): if True StandardScale the ESignature features
              
              cv (int or list): if int it performs a cross-validation with cv train-test random splits,
                                otherwise, it needs to be a list tuples representing train-test splits 
              
       Output: mean MSE (and std) (both scalars) of regression performance on a 5-fold (or custom spatial) cross-validation
    """
    
    X_new = []
    for bag in X:
        if ll is not None:
            bag = LeadLag(ll).fit_transform(bag)
        if at:
            bag = AddTime().fit_transform(bag)
        X_new.append(bag)
    X = X_new
    
    # transforming the data into a 2d array (N_bags, N_items_max x length_min x dim)
    X, max_items, common_T, dim_path = bags_to_2D(X)
    
#     parameters = [{'svm__kernel': ['precomputed'], 'svm__epsilon':[0.,0.1],
#                    'rbf_rbf__gamma_emb':[1e-6, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e6, 1./(common_T*dim_path)],
#                    'rbf_rbf__gamma_top':[1e-6, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e6, 1./(common_T*dim_path)],
#                    'svm__C':[1e-6, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e6]}]

    parameters = [{'svm__kernel': ['precomputed'],
                   'rbf_rbf__gamma_emb':[1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1./(common_T*dim_path)],
                   'rbf_rbf__gamma_top':[1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1./(common_T*dim_path)],
                   'svm__C':[1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]}]

    # building RBF-RBF-SVM estimator
    if ss:
        pipe = Pipeline([('std_scaler', StandardScaler()), 
                         ('rbf_rbf', RBF_RBF(max_items = max_items, size_item=dim_path*common_T)),
                         ('svm', SVR())])
    else:
        pipe = Pipeline([('rbf_rbf', RBF_RBF(max_items = max_items, size_item=dim_path*common_T)),
                         ('svm', SVR())])

    # set-up grid-search over cv random or pre-specified folds
    clf = GridSearchCV(pipe, parameters, verbose=1, n_jobs=-1, scoring='neg_mean_squared_error', cv=cv)

    # find best estimator via grid search
    clf.fit(X, y)

#     score = cross_val_score(clf.best_estimator_, X, mus, scoring='neg_mean_squared_error')
    if not isinstance(cv, int):
        cv = len(cv)    
    scores = np.zeros(cv)
    for i in range(cv):
        scores[i] = np.max(clf.cv_results_['split'+str(i)+'_test_score'])
    
    return -scores.mean(), scores.std()


# The RBF-RBF kernel
class RBF_RBF(BaseEstimator, TransformerMixin):
    def __init__(self, max_items=None, size_item=None, gamma_emb=1.0, gamma_top=1.0):
        super(RBF_RBF, self).__init__()
        self.gamma_emb = gamma_emb
        self.gamma_top = gamma_top
        self.size_item = size_item
        self.max_items = max_items

    def transform(self, X):

        alpha = 1. / (2 * self.gamma_top ** 2)
        x = X.reshape(-1, self.size_item)
        K = rbf_mmd_mat(x, self.x_train, gamma=self.gamma_emb, max_items=self.max_items)

        return np.exp(-alpha * K)

    def fit(self, X, y=None, **fit_params):
        self.X_train_ = X
        x_train = X.reshape(-1, self.size_item)  # x_train is [bag1_item1,bag1_item2,....bagN_itemN] some items are nans
        self.x_train = x_train

        return self


def rbf_mmd_mat(X, Y, gamma=None, max_items=None):
    M = max_items

    alpha = 1. / (2 * gamma ** 2)

    if X.shape[0] == Y.shape[0]:

        XX = np.dot(X, X.T)
        X_sqnorms = np.diagonal(XX)
        K_XX = np.exp(-alpha * (
                -2 * XX + X_sqnorms[:, np.newaxis] + X_sqnorms[np.newaxis, :]))

        K_XX_blocks = [K_XX[i * M:(i + 1) * M, i * M:(i + 1) * M] for i in range(K_XX.shape[0] // M)]
        K_XX_means = [np.nanmean(bag) for bag in K_XX_blocks]

        K_XY_means = np.nanmean(K_XX.reshape(X.shape[0] // M, M, Y.shape[0] // M, M), axis=(1, 3))
        mmd = np.array(K_XX_means)[:, np.newaxis] + np.array(K_XX_means)[np.newaxis, :] - 2 * K_XY_means

    else:

        XX = np.dot(X, X.T)
        XY = np.dot(X, Y.T)
        YY = np.dot(Y, Y.T)

        X_sqnorms = np.diagonal(XX)
        Y_sqnorms = np.diagonal(YY)

        K_XY = np.exp(-alpha * (
                -2 * XY + X_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]))
        K_XX = np.exp(-alpha * (
                -2 * XX + X_sqnorms[:, np.newaxis] + X_sqnorms[np.newaxis, :]))
        K_YY = np.exp(-alpha * (
                -2 * YY + Y_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]))

        # blocks of bags
        K_XX_blocks = [K_XX[i * M:(i + 1) * M, i * M:(i + 1) * M] for i in range(K_XX.shape[0] // M)]
        K_YY_blocks = [K_YY[i * M:(i + 1) * M, i * M:(i + 1) * M] for i in range(K_YY.shape[0] // M)]

        # nanmeans
        K_XX_means = [np.nanmean(bag) for bag in K_XX_blocks]  # n_bags_test
        K_YY_means = [np.nanmean(bag) for bag in K_YY_blocks]  # n_bags_train

        K_XY_means = np.nanmean(K_XY.reshape(X.shape[0] // M, M, Y.shape[0] // M, M), axis=(1, 3))

        mmd = np.array(K_XX_means)[:, np.newaxis] + np.array(K_YY_means)[np.newaxis, :] - 2 * K_XY_means

    return mmd


def bags_to_2D(input_):

    '''
    For the RBF-RBF model, we have defined a customed kernel in sklearn. An sklearn pipeline takes in input
    a 2D array (n_samples, n_features). Whilst we have data in the form of bags of items, where each item is a D-dimensional time series
    represented as a list of list of (LxD) matrices, where L is the length of the time series.

    This function transforms lists of lists of D-dimensional time series into a 2D array.

       Input:
              input_ (list): list of lists of (length,dim) arrays

                        - len(X) = n_bags

                        - for any i, input_[i] is a list of n_items arrays of shape (length, dim)

                        - for any j, input_[i][j] is an array of shape (length, dim)

       Output: a 2D array of shape (n_bags,n_items x length x dim)

    '''

    dim_path = input_[0][0].shape[1]

    # adjust time and concatenate the dimensions
    T = [e[0].shape[0] for e in input_]
    common_T = min(T)  # for other datasets would consider padding. Here we are just missing 10 values from time to time

    unlist_items = [np.concatenate([item[None, :common_T, :] for item in bag], axis=0) for bag in input_]

    # stack dimensions (we get a list of bags, where each bag is N_items x 3T)
    dim_stacked = [np.concatenate([bag[:, :common_T, k] for k in range(dim_path)], axis=1) for bag in unlist_items]

    # stack the items, we have item1(temp-hum-rain) - items2(temp-hum-rain) ...
    items_stacked = [bag.flatten() for bag in dim_stacked]

    # retrieve the maximum number of items
    max_ = [bag.shape for bag in items_stacked]
    max_ = np.max(max_)
    max_items = int(max_ / (dim_path * common_T))

    # pad the vectors with nan items such that we can obtain a 2d array.
    items_naned = [np.append(bag, (max_ - len(bag)) * [np.nan]) for bag in items_stacked]  # np.nan

    X = np.array(items_naned)

    return X, max_items, common_T, dim_path