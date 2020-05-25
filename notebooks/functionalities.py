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