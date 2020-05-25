import sys
sys.path.append('../')

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

from sklearn.cluster import KMeans
import pickle

# epsilon=0 -> Kernel Ridge Regression
# epsilon>0 -> SVM (Regression)
tuned_parameters = [{'svm__kernel': ['rbf'], 'svm__epsilon':[0., 0.1],
                     'svm__gamma': [1e-6, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e6, 'auto'], 
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
    
#     score = cross_val_score(clf.best_estimator_, X, mus, scoring='neg_mean_squared_error')
    scores = np.zeros(3)
    for i in range(3):
        scores[i] = np.max(clf.cv_results_['split'+str(i)+'_test_score'])
    
    return -scores.mean(), scores.std()


def ESig_SVM(depth, X, y, ll=True, at=True, ss=False,targets_dico=None,spatial_CV=False,temporal_CV=False):
    
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
        if ll:
            bag = LeadLag([0]).fit_transform(bag)
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

    if spatial_CV:  ## TO ADD
        splits = spatial_CV(nb_folds=7, targets_dico=targets_dico)
        clf = GridSearchCV(pipe, tuned_parameters, verbose=1, cv=splits, n_jobs=-1, scoring='neg_mean_squared_error')
    elif temporal_CV:
        splits = temporal_CV(targets_dico=targets_dico)
        clf = GridSearchCV(pipe, tuned_parameters, verbose=1, cv=splits, n_jobs=-1, scoring='neg_mean_squared_error')
    else:
        clf = GridSearchCV(pipe, tuned_parameters, verbose=1, n_jobs=-1, scoring='neg_mean_squared_error')

    # find best estimator via grid search 
    clf.fit(X_esig, y)
    
#     score = cross_val_score(clf.best_estimator_, X, mus, scoring='neg_mean_squared_error')
    scores = np.zeros(3)
    for i in range(3):
        scores[i] = np.max(clf.cv_results_['split'+str(i)+'_test_score'])
    
    return -scores.mean(), scores.std()


def SigESig_LinReg(depth1, depth2, X, y, ll=True, at=False, ss=False,targets_dico=None,spatial_CV=False,temporal_CV=False):
    
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
            if ll:
                path = LeadLag([0]).fit_transform([path])[0]
            if at:
                path = AddTime().fit_transform([path])
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
    if ss:
        pipe = Pipeline([('std_scaler', StandardScaler()),('lin_reg', Lasso(max_iter=1000))])
    else:
        pipe = Pipeline([('lin_reg', Lasso(max_iter=1000))])

    if spatial_CV:  ## TO ADD
        splits = spatial_CV(nb_folds=7, targets_dico=targets_dico)
        clf = GridSearchCV(pipe, tuned_parameters, verbose=1, cv=splits, n_jobs=-1, scoring='neg_mean_squared_error')
    elif temporal_CV:
        splits = temporal_CV(targets_dico=targets_dico)
        clf = GridSearchCV(pipe, tuned_parameters, verbose=1, cv=splits, n_jobs=-1, scoring='neg_mean_squared_error')
    else:
        clf = GridSearchCV(pipe, tuned_parameters, verbose=1, n_jobs=-1, scoring='neg_mean_squared_error')

    # find best estimator via grid search 
    clf.fit(X_sigEsig, y)
    
#     score = cross_val_score(clf.best_estimator_, X, mus, scoring='neg_mean_squared_error')
    scores = np.zeros(3)
    for i in range(3):
        scores[i] = np.max(clf.cv_results_['split'+str(i)+'_test_score'])
    
    return -scores.mean(), scores.std()


def RBF_RBF_SVM(X, y,region_labels=None,targets_dico=None,spatial_CV=False,temporal_CV=False):
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
    #pipe = Pipeline([('std_scaler', StandardScaler()), ('svm', SVR())])
    pipe = Pipeline([('std_scaler', StandardScaler()), 
                     ('rbf_rbf', RBF_RBF(max_items = max_items, size_item=dim_path*common_T)),
                     ('svm', SVR())])

    if spatial_CV:  ## TO ADD
        splits = spatial_CV(nb_folds=7, targets_dico=targets_dico)
        clf = GridSearchCV(pipe, tuned_parameters, verbose=1, cv=splits, n_jobs=-1, scoring='neg_mean_squared_error')
    elif temporal_CV:
        splits = temporal_CV(targets_dico=targets_dico)
        clf = GridSearchCV(pipe, tuned_parameters, verbose=1, cv=splits, n_jobs=-1, scoring='neg_mean_squared_error')
    else:
        clf = GridSearchCV(pipe, tuned_parameters, verbose=1, n_jobs=-1, scoring='neg_mean_squared_error')

    # find best estimator via grid search
    clf.fit(X, y)

#     score = cross_val_score(clf.best_estimator_, X, mus, scoring='neg_mean_squared_error')
    scores = np.zeros(3)
    for i in range(3):
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


def spatial_CV(nb_folds, targets_dico):  # TO ADD

    # create spatial train/test splits
    dico_geom = pickle.load(open('../data/crops/dico_geom.obj', 'rb'))

    clusters = KMeans(n_clusters=nb_folds, n_jobs=-1, random_state=2)
    clusters.fit(list(dico_geom.values()))

    v_lookup = {}
    for i in range(len(clusters.labels_)):
        v_lookup[list(dico_geom.keys())[i]] = clusters.labels_[i]

    splits = []
    for i in range(nb_folds):

        # train indices : all but cluster i
        train_indices = []
        # test indices : cluster i
        test_indices = []

        for k in range(len(targets_dico)):
            # get region for the point
            region = list(targets_dico.keys())[k][1]
            if v_lookup[region] != i:  # gives in which cluster the region is
                train_indices.append(k)
            else:
                test_indices.append(k)

        splits.append((train_indices, test_indices))

    return splits


def temporal_CV(targets_dico):  # TO ADD

    years = [int(key[0]) for key in targets_dico.keys()]
    years = np.sort(np.unique(np.array(years)))

    splits = []
    for i in range(len(years) - 1):

        # train indices : all previous years
        train_indices = []
        # test indices : year to predict
        test_indices = []

        for k in range(len(targets_dico)):
            # get region for the point
            year = int(list(targets_dico.keys())[k][0])
            if year < years[i + 1]:  # gives in which cluster the region is
                train_indices.append(k)
            else:
                test_indices.append(k)

        splits.append((train_indices, test_indices))

    return splits