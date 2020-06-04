import numpy as np
from tqdm import tqdm as tqdm

import warnings
warnings.filterwarnings('ignore')

import iisignature
from utils import AddTime, LeadLag

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin




def model(X, y, depth1, depth2, ll=[0], at=False, ss=False, NUM_TRIALS=5, cv=3):
    
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
    X = np.array(X_sigEsig)
    
    # parameters for grid search 
    parameters = [{'lin_reg__alpha': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6], 
                   'lin_reg__fit_intercept' : [True, False], 
                   'lin_reg__normalize' : [True, False]}]
                        
    # building pathwise-Esig estimator
    if ss:
        pipe = Pipeline([('std_scaler', StandardScaler()),('lin_reg', Lasso(max_iter=1000))])
    else:
        pipe = Pipeline([('lin_reg', Lasso(max_iter=1000))])

        
    scores = np.zeros(NUM_TRIALS)
    
    # Loop for each trial
    for i in tqdm(range(NUM_TRIALS)):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

        # parameter search
        model = GridSearchCV(pipe, parameters, verbose=0, n_jobs=-1, scoring='neg_mean_squared_error', cv=cv)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        scores[i] = mean_squared_error(y_pred, y_test)
            
    return scores.mean(), scores.std()
