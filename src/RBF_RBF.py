import numpy as np
from tqdm import tqdm as tqdm

import warnings
warnings.filterwarnings('ignore')

from utils import AddTime, LeadLag, bags_to_2D

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin



def model(X, y, ll=None, at=False, mode='krr', NUM_TRIALS=5,  cv=3):
    
    """Performs a RBF-RBF Kenrel based distribution regression on ensembles (of possibly unequal size)
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
                                
              mode (str): "krr" -> Kernel Ridge Regression, 'svr' -> Support Vector Regresion
              
              NUM_TRIALS, cv : parameters for nested cross-validation
              
       Output: mean MSE (and std) (both scalars) of regression performance on a 5-fold (or custom spatial) cross-validation
    """
    
    assert mode in ['svr', 'krr'], "mode must be either 'svr' or 'krr' "
    
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

    if mode == 'krr':
        parameters = [{'clf__kernel': ['precomputed'], 'clf__alpha':[1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3],
                       'rbf_rbf__gamma_emb':[1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3],
                       'rbf_rbf__gamma_top':[1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]}]
        
        clf = KernelRidge
       
    else:     
        parameters = [{'clf__kernel': ['precomputed'], 'clf__C':[1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3],
                       'rbf_rbf__gamma_emb':[1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1./(common_T*dim_path)],
                       'rbf_rbf__gamma_top':[1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1./(common_T*dim_path)]}]
        
        clf = SVR

    # building RBF-RBF estimator
    pipe = Pipeline([('rbf_rbf', RBF_RBF_Kernel(max_items = max_items, size_item=dim_path*common_T)),
                     ('clf', clf())])
          
    scores = np.zeros(NUM_TRIALS)
    
    # Loop for each trial
    for i in tqdm(range(NUM_TRIALS)):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

        # parameter search
        model = GridSearchCV(pipe, parameters, verbose=0, n_jobs=-1, scoring='neg_mean_squared_error', cv=cv)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        scores[i] = mean_squared_error(y_pred, y_test)
            
    return scores.mean(), scores.std()


# The RBF-RBF kernel
class RBF_RBF_Kernel(BaseEstimator, TransformerMixin):
    def __init__(self, max_items=None, size_item=None, gamma_emb=1.0, gamma_top=1.0):
        super(RBF_RBF_Kernel, self).__init__()
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