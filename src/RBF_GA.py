import warnings
warnings.filterwarnings('ignore')
import numpy as np
import time
from utils import bags_to_2D
from sklearn_transformers import LeadLag, AddTime
import tslearn.metrics as metrics
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


def model(X, y, ll=None, at=False, mode='krr', NUM_TRIALS=5, cv=3, grid={}):
    """Performs a RBF-RBF kernel-based distribution regression on ensembles (of possibly unequal cardinality)
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

    # compute distances between some points of the training set

    #dist_med = compute_distances(data=[bag[:5] for bag in X],n_points=10)


    # transforming the data into a 2d array (N_bags, N_items_max x length_max x dim)
    # X, max_items, common_T, dim_path = bags_to_2D(X,pad_method='pad_nans')

    if mode == 'krr':
        parameters = {'clf__kernel': ['precomputed'], 'clf__alpha': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3],
                      'rbf_ga__gamma':[1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]}
                      # 'rbf_tka__sigma': [0.1*sigma,0.5*sigma],
                      # 'rbf_tka__triangular': [0.2*triangular,0.5*triangular]}

        # check if the user has not given an irrelevant entry
        assert len(list(set(parameters.keys()) & set(grid.keys()))) == len(
            list(grid.keys())), "keys should be in " + ' '.join([str(e) for e in parameters.keys()])
        # merge the user grid with the default one
        parameters.update(grid)
        clf = KernelRidge

    else:
        parameters = {'clf__kernel': ['precomputed'], 'clf__C': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3],
                      'rbf_ga__gamma': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]}
                      # 'rbf_tka__sigma': [0.1*sigma,0.5*sigma],
                      # 'rbf_tka__triangular': [0.2*triangular,0.5*triangular]}

        # check if the user has not given an irrelevant entry
        assert len(list(set(parameters.keys()) & set(grid.keys()))) == len(
            list(grid.keys())), "keys should be in " + ' '.join(
            [str(e) for e in parameters.keys()])

        # merge the user grid with the default one
        parameters.update(grid)

        clf = SVR



    X_nans,n_items_max = fill_nans_items(X)

    sigma = metrics.sigma_gak(dataset=[item for sublist in X for item in sublist],n_samples=len([item for sublist in X for item in sublist]))
    
    K_full = ga_gram(X=X_nans, sigma= sigma)


    K_full = tka_mmd_mat_from_K_full(K_full,n_items_max)


    pipe = Pipeline([('rbf_ga', RBF_GA_Kernel(K_full = K_full)),
                     ('clf', clf())])

    # Loop for each trial
    scores = np.zeros(NUM_TRIALS)
    results = {}
    for i in range(NUM_TRIALS):

        X_train, X_test, y_train, y_test = train_test_split(np.arange(len(y)), y, test_size=0.2, random_state=i)

        model = GridSearchCV(pipe, parameters, verbose=0, n_jobs=-1, scoring='neg_mean_squared_error',cv=cv,error_score=0)
        model.fit(X_train, y_train)


        y_pred = model.predict(X_test)

        scores[i] = mean_squared_error(y_pred, y_test)
        results[i]={'pred':y_pred,'true':y_test}
    return scores.mean(), scores.std(), results


class RBF_GA_Kernel(BaseEstimator, TransformerMixin):
    def __init__(self, K_full=None,gamma=1.0, sigma=1.,triangular=1.0):
        super(RBF_GA_Kernel, self).__init__()
        self.gamma = gamma
        self.sigma = sigma
        self.triangular = triangular
        self.K_full = K_full


    def transform(self, X):
        #print('transform', X.shape)
        alpha = 1. / (2 * self.gamma ** 2)
        K = self.K_full[X][:,self.ind_train].copy()
        return np.exp(-alpha * K)  # #alpha*X #

    def fit(self, X, y=None, **fit_params):
        self.ind_train = X
        return self


def ga_gram(X, sigma):
    X_flat = [item for sublist in X for item in sublist]
    #print('one item',X_flat[0])
    mat = metrics.cdist_gak(X_flat, sigma=sigma, n_jobs=-1)
    return mat



def tka_mmd_mat_from_K_full(K,N_max):

    K_blocks = [K[i * N_max:(i + 1) * N_max, i * N_max:(i + 1) * N_max] for i in range(K.shape[0] // N_max)]
    K_XX_means = [np.nanmean(bag) for bag in K_blocks]

    K_XY_means = np.nanmean(K.reshape(K.shape[0] //  N_max,  N_max, K.shape[0] //  N_max,  N_max), axis=(1, 3))
    mmd = np.array(K_XX_means)[:, np.newaxis] + np.array(K_XX_means)[np.newaxis, :] - 2 * K_XY_means
    return mmd





def fill_nans_items(X):
    # X is M lists of list of (T_j,D) matrices
    # return M lists of N_items_max lists of (T_j,D) matrices

    # check if not same number of items
    n_items = [len(bag) for bag in X]
    n_items_max = max(n_items)
    if len(list(set(n_items)))==1:
        return X, n_items_max
    else:
        new_X = []
        D = X[0][0].shape[1]

        for bag in X:
            new_bag = bag.copy()
            for missing_item in range(n_items_max-len(bag)):
                new_bag.append(np.nan*np.zeros((1,D)))
            new_X.append(new_bag)

        return new_X, n_items_max
