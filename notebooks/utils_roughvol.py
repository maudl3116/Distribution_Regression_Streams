import sys
sys.path.append('../')

import numpy as np
from tqdm import tqdm_notebook as tqdm

import warnings
warnings.filterwarnings('ignore')

from fbm import FBM
import iisignature
from utils.addtime import AddTime, LeadLag

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV, ShuffleSplit, cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.pipeline import Pipeline

tuned_parameters = [{'svm__kernel': ['poly'], 'svm__gamma': ['auto'], 
                     'svm__C': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3], 'svm__degree': [1, 2, 3, 4, 5, 6]},
                    {'svm__kernel': ['rbf'], 'svm__gamma': ['auto'], 
                     'svm__C': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]}]

def fOU_generator(a,n=0.3,h=0.3,length=100):
    fbm_increments = np.diff(FBM(length, h).fbm())
    # X(t+1) = X(t) - a(X(t)-m) + n(W(t+1)-W(t))
    x0 = np.random.normal(1,0.1)
    x0 = 0.5
    m = x0
    price = [x0]
    for i in range(length):
        p = price[i] - a*(price[i]-m) + n*fbm_increments[i]
        price.append(p)
    return np.array(price)


def poly_SVM(degree, mus, n_bags, n_items, tuned_parameters):
    
    X = []
    
    for a in tqdm(mus):
        intermediate = []
        for n in range(n_items):
            intermediate.append(np.exp(fOU_generator(a)))
        intermediate = PolynomialFeatures(degree).fit_transform(np.array(intermediate))
        X.append(intermediate.mean(0))
    
    pipe = Pipeline([('std_scaler', StandardScaler()), ('svm', SVR())])
    clf = GridSearchCV(pipe, tuned_parameters, verbose=1, n_jobs=-1, scoring='neg_mean_squared_error')
    clf.fit(X, mus)

    score = cross_val_score(clf.best_estimator_, X, mus, scoring='neg_mean_squared_error')
    return -score.mean(), score.std()


def ESig_SVM(depth, mus, n_bags, n_items, tuned_parameters):
    
    X = []
    
    for a in tqdm(mus):
        intermediate = []
        for n in range(n_items):
            intermediate.append(np.exp(fOU_generator(a)).reshape(-1,1))
        intermediate = LeadLag([0]).fit_transform(intermediate)
        intermediate = AddTime().fit_transform(intermediate)
        intermediate = iisignature.sig(intermediate, depth)
        X.append(intermediate.mean(0))
        
    pipe = Pipeline([('svm', SVR())])
    clf = GridSearchCV(pipe, tuned_parameters, verbose=1, n_jobs=-1, scoring='neg_mean_squared_error')
    clf.fit(X, mus)

    score = cross_val_score(clf.best_estimator_, X, mus, scoring='neg_mean_squared_error')
    return -score.mean(), score.std()


def SigESig_SVM(depth1, depth2, mus, n_bags, n_items):
    
    X = []
    
    for a in tqdm(mus):
        intermediate = []
        for n in range(n_items):
            path = np.exp(fOU_generator(a)).reshape(-1,1)
            path = LeadLag([0]).fit_transform([path])[0]
            path = AddTime().fit_transform([path])
            sig_path = iisignature.sig(path, depth1, 2) 
            intermediate.append(sig_path)
        intermediate = iisignature.sig(intermediate, depth2)
        X.append(intermediate.mean(0))
    X = np.array(X)[:,0,:]
        
    parameters = [{'lin_reg__alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2], 
                   'lin_reg__fit_intercept' : [True, False], 
                   'lin_reg__normalize' : [True, False]}]
    
    pipe = Pipeline([('lin_reg', Lasso(max_iter=1000))])
    
    clf = GridSearchCV(pipe, parameters, verbose=1, n_jobs=-1, scoring='neg_mean_squared_error')
    clf.fit(X, mus)

    score = cross_val_score(clf.best_estimator_, X, mus, cv=5, scoring='neg_mean_squared_error')
    return -score.mean(), score.std()