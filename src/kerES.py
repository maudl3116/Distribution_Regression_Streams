import numpy as np
from tqdm import tqdm as tqdm
from itertools import chain, combinations

import warnings
warnings.filterwarnings('ignore')

from utils import bags_to_2D
from sklearn_transformers import AddTime, LeadLag, ExpectedSignatureTransform
from SigKerPDESolver import SigKerMMD

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


def model(X, y, mode='krr', NUM_TRIALS=5,  cv=3):
    
    """Performs a ESig-RBF Kernel based distribution regression on ensembles (of possibly unequal size)
       of univariate or multivariate time-series equal of unequal lengths

       Input:
              X (list): list of lists such that

                        - len(X) = n_samples

                        - for any i, X[i] is a list of arrays of shape (length, dim)

                        - for any j, X[i][j] is an array of shape (length, dim)

              y (np.array): array of shape (n_samples,)
                                
              mode (str): "krr" -> Kernel Ridge Regression, 'svr' -> Support Vector Regresion
              
              NUM_TRIALS, cv : parameters for nested cross-validation
              
       Output: mean MSE (and std) (both scalars) of regression performance on a 5-fold (or custom spatial) cross-validation
    """
    
    assert mode in ['svr', 'krr'], "mode must be either 'svr' or 'krr' "

    # for the lead-lag transformation
    ll = [x for x in powerset(list(np.arange(X[0][0].shape[1])))][:-1]
    ll+=[None]


    if mode == 'krr':

        parameters = [{'clf__kernel': ['rbf'],
                           'clf__gamma': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3],
                           'clf__alpha': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3],
                           'lead_lag_transform__dimensions_to_lag': ll,
                           'add_time_transform':[None,AddTime()],
                           'std_scaler': [None,StandardScaler()],
                           'expected_signature_transform__order': [2,3,4,5,6,7,8]}
                           ]

        clf = KernelRidge
       
    else:     

        parameters = [{'clf__kernel': ['rbf'],
                       'clf__gamma': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3],
                       'clf__C': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3],
                       'lead_lag_transform__dimensions_to_lag': ll,
                       'add_time_transform':[None,AddTime()],
                       'expected_signature_transform__order': [2,3,4,5,6,7,8],
                       'std_scaler': [None,StandardScaler()]
                       }]

        clf = SVR

    # building the estimator

    pipe = Pipeline([('lead_lag_transform', LeadLag(dimensions_to_lag=[0])),
                     ('add_time_transform', AddTime()),
                     ('expected_signature_transform', ExpectedSignatureTransform(order=2)),
                     ('std_scaler', StandardScaler()),
                     ('clf', clf())
                     ])
      
    scores = np.zeros(NUM_TRIALS)
    
    # Loop for each trial
    for i in tqdm(range(NUM_TRIALS)):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        # parameter search
        model = GridSearchCV(pipe, parameters, verbose=0, n_jobs=-1, scoring='neg_mean_squared_error', cv=cv, error_score=np.nan)
        model.fit(X_train, y_train)
        print(model.best_params_)
        y_pred = model.predict(X_test)
        
        scores[i] = mean_squared_error(y_pred, y_test)
            
    return scores.mean(), scores.std()


def powerset(seq):
    """
    Returns all the subsets of this set. This is a generator.
    """
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]]+item
            yield item

# The ESig_RBF kernel
# class ESig_RBF_Kernel(BaseEstimator, TransformerMixin):
#     def __init__(self, dim_path=None, max_items=None, size_item=None, gamma=1.0):
#         super(ESig_RBF_Kernel, self).__init__()
#         self.gamma = gamma
#         self.size_item = size_item # L*dim_path
#         self.max_items = max_items
#         self.dim_path = dim_path
#
#
#     def transform(self, X):
#         '''
#         X is a 2D array of shape (N_bags,n_max_items*L*dim_path)
#         1) reshape X into (Nbags*max_items,L*dim_path)
#         2) extract bags in the form (max_items,L,dim_path)
#         3) remove nanned items
#         '''
#
#         alpha = 1. / (2 * self.gamma ** 2)
#         x = X.reshape(-1, self.max_items,self.size_item)
#
#         MMD_mat = np.zeros((X.shape[0],self.X_train_.shape[0]))
#         for i in range(X.shape[0]):
#             bag_i = x[i][~np.isnan(x[i]).all(axis=1)]
#             bag_i = np.split(bag_i,self.dim_path,axis=1)  # turn (N_items,dim_path*common_T) into  (N_items,common_T,dim_path)
#             bag_i = np.concatenate([item[:,:,None] for item in bag_i],axis=2)
#             for j in range(self.X_train_.shape[0]):
#                 bag_j = self.x_train[j][~np.isnan(self.x_train[j]).all(axis=1)]
#                 bag_j = np.split(bag_j,self.dim_path,axis=1)  # turn (N_items,dim_path*common_T) into  (N_items,common_T,dim_path)
#                 bag_j = np.concatenate([item[:,:,None] for item in bag_j],axis=2)
#                 MMD_mat[i][j] = SigKerMMD(bag_i,bag_j)
#
#         return np.exp(-alpha * MMD_mat)
#
#     def fit(self, X, y=None, **fit_params):
#         self.X_train_ = X
#         x_train = X.reshape(-1,self.max_items, self.size_item)  # x_train is [bag1_item1,bag1_item2,....bagN_itemN] some items are nans
#         self.x_train = x_train
#
#         return self
#
#
