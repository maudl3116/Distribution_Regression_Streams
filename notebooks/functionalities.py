import sys
sys.path.append('../')

import numpy as np
from tqdm import tqdm_notebook as tqdm

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

tuned_parameters = [{'svm__kernel': ['rbf'], 'svm__gamma': [1e-6, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e6, 'auto'], 
                     'svm__C': [1e-6, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e6]}]


def poly_SVM(degree, X, y):
    
    """Performs a poly(degree)-SVM distribution regression on ensembles (of possibly unequal size) 
       of univariate or multivariate time-series equal of unequal lengths 
    
       Input: degree (int): degree of the polynomial feature map
       
              X (list): list of lists such that
              
                        - len(X) = n_samples
                        
                        - for any i, X[i] is a list of arrays of shape (length, dim)
                        
                          !!! note that all arrays in the list must have same length and same dim !!!
                          
                        - for any j, X[i][j] is an array of shape (length, dim)
                        
              y (np.array): array of shape (n_samples,)
              
       Output: mean MSE (and std) (both scalars) of regression performance on a 5-fold cross-validation
    
    """
    
    X_poly = []
    
    # take polynomial feature map
    for bag in tqdm(X):
        X_poly.append(PolynomialFeatures(degree).fit_transform([x.reshape(-1) for x in bag]).mean(0))
        
    X_poly = np.array(X_poly)
                
    # building poly-SVM estimator
    pipe = Pipeline([('std_scaler', StandardScaler()), ('svm', SVR())])
    
    # set-up grid-search over 5 random folds
    clf = GridSearchCV(pipe, tuned_parameters, verbose=1, n_jobs=-1, scoring='neg_mean_squared_error')
    
    # find best estimator via grid search 
    clf.fit(X_poly, y)
    
    return clf.best_estimator_
    
#     # cross-validation across 5 folds with best estimator
#     score = cross_val_score(clf.best_estimator_, X_poly, y, scoring='neg_mean_squared_error')
#     return -score.mean(), score.std()


def ESig_SVM(depth, X, y):
    
    """Performs a ESig(depth)-SVM distribution regression on ensembles (of possibly unequal size) 
       of univariate or multivariate time-series equal of unequal lengths 
    
       Input: depth (int): truncation of the signature
       
              X (list): list of lists such that
              
                        - len(X) = n_samples
                        
                        - for any i, X[i] is a list of arrays of shape (length, dim)
                                                  
                        - for any j, X[i][j] is an array of shape (length, dim)
                        
              y (np.array): array of shape (n_samples,)
              
       Output: mean MSE (and std) (both scalars) of regression performance on a 5-fold cross-validation
    
    """
    
    X_esig = []
    
    # take Esig feature map
    for bag in tqdm(X):
        intermediate = bag
#         intermediate = LeadLag([0]).fit_transform(bag)
#         intermediate = AddTime().fit_transform(intermediate)
        try:
            intermediate = iisignature.sig(intermediate, depth)
        except:
            intermediate = np.array([iisignature.sig(p, depth) for p in intermediate])
        if intermediate.shape[0]>0:
            X_esig.append(intermediate.mean(0))
        
    X_esig = np.array(X_esig)
                    
    # building ESig-SVM estimator
#     pipe = Pipeline([('svm', SVR())])
    pipe = Pipeline([('std_scaler', StandardScaler()), ('svm', SVR())])
    
    # set-up grid-search over 5 random folds
    clf = GridSearchCV(pipe, tuned_parameters, verbose=1, n_jobs=-1, scoring='neg_mean_squared_error')
    
    # find best estimator via grid search 
    clf.fit(X_esig, y)
    
#     return clf.best_estimator_
    
    # cross-validation across 5 folds with best estimator
    score = cross_val_score(clf.best_estimator_, X_esig, y, scoring='neg_mean_squared_error')
    return -score.mean(), score.std()


def SigESig_LinReg(depth1, depth2, X, y):
    
    """Performs a SigESig(depth)-Linear distribution regression on ensembles (of possibly unequal size) 
       of univariate or multivariate time-series equal of unequal lengths 
    
       Input: depth1 (int): truncation of the signature 1
              depth2 (int): truncation of the signature 2
       
              X (list): list of lists such that
              
                        - len(X) = n_samples
                        
                        - for any i, X[i] is a list of arrays of shape (length, dim)
                                                  
                        - for any j, X[i][j] is an array of shape (length, dim)
                        
              y (np.array): array of shape (n_samples,)
              
       Output: mean MSE (and std) (both scalars) of regression performance on a 5-fold cross-validation
    
    """
    
    
    X_sigEsig = []
    
    # take sigEsig feature map
    for bag in tqdm(X):
        intermediate = []
        for path in bag:
#             path = LeadLag([0]).fit_transform([path])[0]
#             path = AddTime().fit_transform([path])
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
                        
    # building ESig-SVM estimator
    pipe = Pipeline([('lin_reg', Lasso(max_iter=1000))])
#     pipe = Pipeline([('std_scaler', StandardScaler()),('lin_reg', Lasso(max_iter=1000))])
    
    # set-up grid-search over 5 random folds
    clf = GridSearchCV(pipe, parameters, verbose=1, n_jobs=-1, scoring='neg_mean_squared_error')
    
    # find best estimator via grid search 
    clf.fit(X_sigEsig, y)
    
#     return clf.best_estimator_
    
    # cross-validation across 5 folds with best estimator
    score = cross_val_score(clf.best_estimator_, X_sigEsig, y, scoring='neg_mean_squared_error')
    return -score.mean(), score.std()


def RBF_RBF_SVM(X, y,region_labels=None):
    """Performs a RBF-RBF-SVM distribution regression on ensembles (of possibly unequal size)
       of univariate or multivariate time-series equal of unequal lengths

       Input:
              X (list): list of lists such that

                        - len(X) = n_samples

                        - for any i, X[i] is a list of arrays of shape (length, dim)

                        - for any j, X[i][j] is an array of shape (length, dim)

              y (np.array): array of shape (n_samples,)
              region_labels (dictionary): if crop example and we want to perform a spatial cross-validation

       Output: mean MSE (and std) (both scalars) of regression performance on a 5-fold (or custom spatial) cross-validation

       Careful + TO DO: The implementation of the RBF-RBF kernel requires to have train-test splits of different cardinality (in terms of bags)
                How can we incorporate a standard scaler? To do: spatial cross_val

    """
    tuned_parameters = [{'svm__kernel': ['precomputed'], 'rbf_rbf__gamma_emb': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3],'rbf_rbf__gamma_top': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3],
                         'svm__C': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]}]

    # transforming the data into a 2d array (N_bags, N_items_max x length_min x dim)
    X, max_items, common_T, dim_path = bags_to_2D(X)

    # building RBF-RBF-SVM estimator
    #pipe = Pipeline([('std_scaler', StandardScaler()), ('svm', SVR())])
    pipe = Pipeline([('rbf_rbf', RBF_RBF(max_items = max_items, size_item=dim_path*common_T)),('svm', SVR())])

    # set-up grid-search over 5 random folds
    clf = GridSearchCV(pipe, tuned_parameters, verbose=1, n_jobs=-1, scoring='neg_mean_squared_error')

    # find best estimator via grid search
    clf.fit(X, y)

    #     return clf.best_estimator_

    # cross-validation across 5 folds with best estimator

    score = cross_val_score(clf.best_estimator_, X, y, scoring='neg_mean_squared_error')


    return -score.mean(), score.std()


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
    Can handle paths of different lengths (naive truncation to smallest lengths, but ok for real data we consider), and bags of varying cardinality

       Input:
              input_ (list): list of lists such that

                        - len(X) = n_samples

                        - for any i, X[i] is a list of arrays of shape (length, dim)

                        - for any j, X[i][j] is an array of shape (length, dim)

       Output: The input in the form of a 2d array, the maximum number of items, the common length at which paths have been truncated, the dimensionality of the paths

    '''

    dim_path = input_[0][0].shape[1]

    # adjust time and concatenate the dimensions
    T = [e[0].shape[0] for e in input_]
    common_T = min(T)  # for other datasets would consider padding. Here we are just missing 10 values from time to time

    unlist_items = [np.concatenate([item[None, :common_T, :] for item in bag[:30]], axis=0) for bag in input_]

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