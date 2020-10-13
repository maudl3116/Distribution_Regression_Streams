import warnings
import sys

warnings.filterwarnings('ignore')
sys.path.append('../src')
# sys.path.append('../src/TGA_python_wrapper')
import numpy as np
import time
from utils import bags_to_2D
from sklearn_transformers import LeadLag, AddTime
import tslearn.metrics as metrics
from sklearn.model_selection import GridSearchCV, train_test_split
from gaKer_fast import GA_kernels_mmd
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
# import global_align as ga

from joblib import Parallel, delayed
from tslearn.utils import to_time_series, to_time_series_dataset, ts_size, \
    check_equal_size


def model(X, y, ll=None, at=False, mode='krr', NUM_TRIALS=5, cv=3, grid={}, lambdas=[1.]):
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

    # sig_cuturi = metrics.sigma_gak(dataset=[item for sublist in X for item in sublist],
    #                           n_samples=len([item for sublist in X for item in sublist]))


    if mode == 'krr':
        parameters = {'clf__kernel': ['precomputed'], 'clf__alpha': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3],
                      'rbf_ga__gamma': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]}

        # check if the user has not given an irrelevant entry
        assert len(list(set(parameters.keys()) & set(grid.keys()))) == len(
            list(grid.keys())), "keys should be in " + ' '.join([str(e) for e in parameters.keys()])
        # merge the user grid with the default one
        parameters.update(grid)
        # print(parameters)
        clf = KernelRidge

    else:
        parameters = {'clf__kernel': ['precomputed'], 'clf__C': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3],
                      'rbf_ga__gamma': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]}

        # check if the user has not given an irrelevant entry
        assert len(list(set(parameters.keys()) & set(grid.keys()))) == len(
            list(grid.keys())), "keys should be in " + ' '.join(
            [str(e) for e in parameters.keys()])

        # merge the user grid with the default one
        parameters.update(grid)

        clf = SVR


    # Loop for each trial
    scores = np.zeros(NUM_TRIALS)
    results = {}
    X_nans, n_items_max = fill_nans_items(X)
    list_kernels = []

    ''' Method Parallelisation 1: The GA gram matrix computed via soft-DTW has not been parallelized, cdist_soft_dtw is just a double for-loop.'''
    # Here we parallelize using joblib in our own cdist_soft_dtw.
    # for n, lambda_ in enumerate(lambdas):
    #     K_full = soft_dtw_gram(X=X_nans, sigma=1, gamma=lambda_)
    #     K_full = tka_mmd_mat_from_K_full(K_full, n_items_max)
    #
    #     list_kernels.append(K_full)

    ''' Method Parallelisation 2: One kernel evaluation is done in cython. Here we cythonize the whole gram matrix computation like we do for KES. '''
    for n,lambda_ in enumerate(lambdas):
       K_full = np.zeros((len(X), len(X)),dtype=np.float64)
       indices = np.triu_indices(len(X),k=0,m=len(X))
       K_full[indices] = Parallel(n_jobs=-1,verbose=3)(
           delayed(ExpectedKernel)(X[i],X[j],sigma=1, gamma=lambda_)
           for i in range(len(X))
           for j in range(i,len(X))
       )
       indices = np.tril_indices(len(X), k=-1, m=len(X))
       K_full[indices] = K_full.T[indices]
       diag = np.diag(K_full)

       mmd = -2. * K_full + np.tile(diag,(K_full.shape[0],1)) + np.tile(diag[:,None],(1,K_full.shape[0]))
       list_kernels.append(mmd)

    # Loop for each trial
    for i in range(NUM_TRIALS):

        best_scores_train = np.zeros(len(lambdas))

        # will only retain the MSE (mean + std) corresponding to the model achieving the best score (on the train set)
        # i.e. the test set is not used to decide the truncation level.
        # the truncation level cannot be placed in a pipeline otherwise the signatures are recomputed even when not needed

        MSE_test = np.zeros(len(lambdas))

        for n, lambda_ in enumerate(lambdas):
            results_tmp = {}
            if not np.isnan(list_kernels[n]).any():
                pipe = Pipeline([('rbf_ga', RBF_GA_Kernel(K_full=list_kernels[n])),
                                 ('clf', clf())])

                X_train, X_test, y_train, y_test = train_test_split(np.arange(len(y)), y, test_size=0.2, random_state=i)

                # parameter search
                model = GridSearchCV(pipe, parameters, verbose=0, n_jobs=-1, scoring='neg_mean_squared_error', cv=cv,
                                     error_score=np.nan)

                model.fit(X_train, y_train)

                best_scores_train[n] = -model.best_score_

                y_pred = model.predict(X_test)
                results_tmp[n] = {'pred': y_pred, 'true': y_test}
                MSE_test[n] = mean_squared_error(y_pred, y_test)
            else:
                best_scores_train[n] = 100000
                MSE_test[n] = 100000
        # pick the model with the best performances on the train set
        best_score = 100000
        index = 0
        for n, lambda_ in enumerate(lambdas):
            if (best_scores_train[n] < best_score):
                best_score = best_scores_train[n]
                index = n

        scores[i] = MSE_test[index]
        results[i] = results_tmp[index]
        print('best sigma (cv on the train set): ', lambdas[index])
    return scores.mean(), scores.std(), results


class RBF_GA_Kernel(BaseEstimator, TransformerMixin):
    def __init__(self, K_full=None, gamma=1.0, sigma=1., triangular=1.0):
        super(RBF_GA_Kernel, self).__init__()
        self.gamma = gamma
        self.sigma = sigma
        self.triangular = triangular
        self.K_full = K_full

    def transform(self, X):
        alpha = 1. / (2 * self.gamma ** 2)
        K = self.K_full[X][:, self.ind_train].copy()
        return np.exp(-alpha * K)  # #alpha*X #

    def fit(self, X, y=None, **fit_params):
        self.ind_train = X
        return self

''' CODE for parallelization 1 '''
def soft_dtw_gram(X, sigma=1., gamma=1.):
    X_flat = [item for sublist in X for item in sublist]
    # print('one item',X_flat[0])
    # mat = metrics.cdist_soft_dtw(X_flat, gamma=sigma)
    mat = cdist_soft_dtw(X_flat, gamma=sigma)
    return np.exp(-gamma * mat)

def cdist_soft_dtw(dataset1, dataset2=None, gamma=1.):
    # from _cdist_generic

    dataset1 = to_time_series_dataset(dataset1, dtype=np.float64)
    if dataset2 is None:
        dataset2 = dataset1
        dataset2_was_none = True
    else:
        dataset2_was_none = False
        dataset2 = to_time_series_dataset(dataset2, dtype=np.float64)

    dists = np.empty((dataset1.shape[0], dataset2.shape[0]))

    if dataset2_was_none:
        indices = np.triu_indices(len(dataset1), k=0, m=len(dataset1))
        dists[indices] = Parallel(n_jobs=-1,
                                  verbose=3)(
            delayed(soft_dtw)(
                dataset1[i], dataset1[j], gamma=gamma
            )
            for i in range(len(dataset1))
            for j in range(i,
                           len(dataset1))
        )
        indices = np.tril_indices(len(dataset1), k=-1, m=len(dataset1))
        dists[indices] = dists.T[indices]
        return dists
    else:
        dataset2 = to_time_series_dataset(dataset2, dtype=np.float64)
        matrix = Parallel(n_jobs=-1, verbose=0)(
            delayed(soft_dtw)(
                dataset1[i], dataset2[j], gamma=gamma)
            for i in range(len(dataset1)) for j in range(len(dataset2))
        )
        return np.array(matrix).reshape((len(dataset1), -1))


def soft_dtw(ts1, ts2, gamma=1.):
    if np.isnan(ts1).all() or np.isnan(ts2).all():
        return np.nan
    if gamma == 0.:
        return dtw(ts1, ts2) ** 2
    return metrics.soft_dtw(ts1, ts2, gamma=gamma)

def tka_mmd_mat_from_K_full(K, N_max):
    K_blocks = [K[i * N_max:(i + 1) * N_max, i * N_max:(i + 1) * N_max] for i in range(K.shape[0] // N_max)]
    K_XX_means = [np.nanmean(bag) for bag in K_blocks]
    K_XY_means = np.nanmean(K.reshape(K.shape[0] // N_max, N_max, K.shape[0] // N_max, N_max), axis=(1, 3))
    mmd = np.array(K_XX_means)[:, np.newaxis] + np.array(K_XX_means)[np.newaxis, :] - 2 * K_XY_means
    return mmd


''' Code for parallelization 2 '''
def ExpectedKernel(bag_i, bag_j, sigma=1, gamma=1):
    K_ij = GA_kernels_mmd(np.array(bag_i,dtype=np.float64), np.array(bag_j,dtype=np.float64), gamma=sigma)
    K_ij = np.exp(-gamma * np.array(K_ij))
    return np.mean(K_ij)




# def tka_mat_from_K_full(K,N_max):
#   K_XY_means = np.nanmean(K.reshape(K.shape[0] //  N_max,  N_max, K.shape[0] //  N_max,  N_max), axis=(1, 3))
#  return K_XY_means


# def tka_gram(X, sigma=1., lambda_=1., triangular=0):
#     X_flat = [item for sublist in X for item in sublist]
#     # print('one item',X_flat[0])
#     mat = cdist_tka(X_flat, sigma=sigma, lambda_=lambda_, triangular=triangular, n_jobs=-1)
#     return mat
#
#
# def tka_unormalized(s1, s2, sigma=1., lambda_=1., triangular=0):
#     ''' for two time series '''
#     s1 = to_time_series(s1, remove_nans=True)
#     s2 = to_time_series(s2, remove_nans=True)
#     gram = ga.tga_dissimilarity(s1, s2, sigma=sigma, lambda_=lambda_, triangular=triangular, normalized=0)
#     gak_val = np.exp(-gram)
#     return gak_val


# def cdist_tka(dataset1, dataset2=None, sigma=1., lambda_=1., triangular=0, n_jobs=None, verbose=0):
#     unnormalized_matrix = _cdist_generic(dist_fun=tka_unormalized,
#                                          dataset1=dataset1,
#                                          dataset2=dataset2,
#                                          n_jobs=n_jobs,
#                                          verbose=verbose,
#                                          sigma=sigma,
#                                          lambda_=lambda_,
#                                          triangular=triangular,
#                                          compute_diagonal=True)
#
#     dataset1 = to_time_series_dataset(dataset1)
#     if dataset2 is None:
#         diagonal = np.diag(np.sqrt(1. / np.diag(unnormalized_matrix)))
#         diagonal_left = diagonal_right = diagonal
#     else:
#         dataset2 = to_time_series_dataset(dataset2)
#         diagonal_left = Parallel(n_jobs=n_jobs,
#                                  prefer="threads",
#                                  verbose=verbose)(
#             delayed(tka_unormalized)(dataset1[i], dataset1[i], sigma=sigma, lambda_=lambda_, triangular=triangular)
#             for i in range(len(dataset1))
#         )
#         diagonal_right = Parallel(n_jobs=n_jobs,
#                                   prefer="threads",
#                                   verbose=verbose)(
#             delayed(tka_unormalized)(dataset2[j], dataset2[j], sigma=sigma, lambda_=lambda_, triangular=triangular)
#             for j in range(len(dataset2))
#         )
#         diagonal_left = np.diag(1. / np.sqrt(diagonal_left))
#         diagonal_right = np.diag(1. / np.sqrt(diagonal_right))
#     # return unnormalized_matrix #(diagonal_left.dot(unnormalized_matrix)).dot(diagonal_right)
#     return (diagonal_left.dot(unnormalized_matrix)).dot(diagonal_right)


# def _cdist_generic(dist_fun, dataset1, dataset2, n_jobs, verbose,
#                    compute_diagonal=True, dtype=np.float, *args, **kwargs):
#     dataset1 = to_time_series_dataset(dataset1, dtype=dtype)
#
#     if dataset2 is None:
#         # Inspired from code by @GillesVandewiele:
#         # https://github.com/rtavenar/tslearn/pull/128#discussion_r314978479
#         matrix = np.zeros((len(dataset1), len(dataset1)))
#         indices = np.triu_indices(len(dataset1),
#                                   k=0 if compute_diagonal else 1,
#                                   m=len(dataset1))
#         matrix[indices] = Parallel(n_jobs=n_jobs,
#                                    prefer="threads",
#                                    verbose=verbose)(
#             delayed(dist_fun)(
#                 dataset1[i], dataset1[j],
#                 *args, **kwargs
#             )
#             for i in range(len(dataset1))
#             for j in range(i if compute_diagonal else i + 1,
#                            len(dataset1))
#         )
#         indices = np.tril_indices(len(dataset1), k=-1, m=len(dataset1))
#         matrix[indices] = matrix.T[indices]
#         return matrix
#     else:
#         dataset2 = to_time_series_dataset(dataset2, dtype=dtype)
#         matrix = Parallel(n_jobs=n_jobs, prefer="threads", verbose=verbose)(
#             delayed(dist_fun)(
#                 dataset1[i], dataset2[j],
#                 *args, **kwargs
#             )
#             for i in range(len(dataset1)) for j in range(len(dataset2))
#         )
#         return np.array(matrix).reshape((len(dataset1), -1))

# def ga_gram(X, sigma):
#     X_flat = [item for sublist in X for item in sublist]
#     # print('one item',X_flat[0])
#     mat = metrics.cdist_gak(X_flat, sigma=sigma, n_jobs=-1)
#     return mat

def fill_nans_items(X):
    # X is M lists of list of (T_j,D) matrices
    # return M lists of N_items_max lists of (T_j,D) matrices

    # check if not same number of items
    n_items = [len(bag) for bag in X]
    n_items_max = max(n_items)
    if len(list(set(n_items))) == 1:
        return X, n_items_max
    else:
        new_X = []
        D = X[0][0].shape[1]

        for bag in X:
            new_bag = bag.copy()
            for missing_item in range(n_items_max - len(bag)):
                new_bag.append(np.nan * np.zeros((1, D)))
            new_X.append(new_bag)

        return new_X, n_items_max