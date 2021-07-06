import numpy as np
from tqdm import tqdm as tqdm
from tqdm import trange as trange
from sklearn_transformers import AddTime, LeadLag, SketchExpectedSignatureTransform, SketchpwCKMETransform
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR, SVC
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import itertools
import torch 
import sigkernel


def model(X, y, alphas=[0.5], rbf=True, dyadic_order=1, ll=None, at=False, mode='krr', NUM_TRIALS=1, cv=3, grid={}):
    """Performs a kernel based distribution regression on ensembles (of possibly unequal cardinality)
       of univariate or multivariate time-series (of possibly unequal lengths)
       Input:
              X (list): list of lists such that
                        - len(X) = n_samples
                        - for any i, X[i] is a list of arrays of shape (length, dim)
                        - for any j, X[i][j] is an array of shape (length, dim)
              y (np.array): array of shape (n_samples,) 
              alphas (list of floats): RBF kernel scaling parameter to cross-validate
              dyadic_order (int): dyadic order of PDE solver
              ll (list of ints): dimensions to lag (set to None by default)
              at (bool): if True pre-process the input path with add-time
              mode (str): "krr" -> Kernel Ridge Regression, 'svr' -> Support Vector Regresion
              NUM_TRIALS, cv : parameters for cross-validation
              grid (dict): a dictionary to specify the hyperparameter grid for the gridsearch. Unspecified entries will be set by default
       Output: mean MSE (and std) (both scalars) of regression performance on a cv-folds cross-validation (NUM_TRIALS times) as well results (a dictionary containing the predicted labels and true labels)
    """
    
    use_gpu = torch.cuda.is_available()

    assert mode in ['svr', 'krr'], "mode must be either 'svr' or 'krr' "

    # possibly augment the state space of the time series
    if ll is not None:
        X = LeadLag(ll).fit_transform(X)
    if at:
        X = AddTime().fit_transform(X)
    
    if mode == 'krr':

        # default grid
        parameters = {'clf__kernel': ['precomputed'],
                      'rbf_mmd__gamma':[1e3, 1e2, 1e1, 1, 1e-1,1e-2,1e-3],
                      'clf__alpha': [1e3, 1e2, 1e1, 1e-1, 1e-2, 1e-3]}

        # check if the user has not given an irrelevant entry
        assert len(list(set(parameters.keys()) & set(grid.keys()))) == len(
            list(grid.keys())), "keys should be in " + ' '.join([str(e) for e in parameters.keys()])

        # merge the user grid with the default one
        parameters.update(grid)

        clf = KernelRidge

    else:

        # default grid
        parameters = {'clf__kernel': ['precomputed'],
                       'clf__gamma': [1e3, 1e2, 1e1, 1, 1e-1,1e-2,1e-3], 
                      'rbf_mmd__gamma':[1e3, 1e2, 1e1, 1, 1e-1,1e-2,1e-3],
                       'clf__C': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
                       }

        # check if the user has not given an irrelevant entry
        assert len(list(set(parameters.keys()) & set(grid.keys()))) == len(
            list(grid.keys())), "keys should be in " + ' '.join([str(e) for e in parameters.keys()])

        # merge the user grid with the default one
        parameters.update(grid)
        clf = SVR

    list_kernels = []

    # Precompute the Gram matrices for the different scaling parameters, to avoid recomputing them for each grid search step
    for alpha in alphas:

        mmd = np.zeros((len(X),len(X)))

        if rbf:
            static_kernel = sigkernel.RBFKernel(sigma=alpha,add_time=X[0][0].shape[0]-1)
        else:
            static_kernel = sigkernel.LinearKernel(add_time=X[0][0].shape[0]-1)

        signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order)

        for i in trange(len(X)):
            mu = torch.tensor(X[i])
            if use_gpu:
                mu = mu.cuda()
            for j in range(i,len(X)):
                nu = torch.tensor(X[j])
                if use_gpu:
                    nu = nu.cuda()
                mmd[i,j] = signature_kernel.compute_mmd(mu,nu)
                mmd[j,i] = mmd[i,j]
        if np.isnan(mmd).any():
            list_kernels.append(np.eye(len(X)))
        else:
            list_kernels.append(mmd)

    
    scores = np.zeros(NUM_TRIALS)
    results = {}
    # Loop for each trial
    for i in tqdm(range(NUM_TRIALS)):

        best_scores_train = np.zeros(len(alphas))

        # will only retain the MSE (mean + std) corresponding to the model achieving the best score (on the train set)
        # i.e. the test set is not used to decide the hyperparameters.
   
        MSE_test = np.zeros(len(alphas))
        results_tmp = {}
        models = []
        for n,alpha in enumerate(alphas):
            
            ind_train, ind_test, y_train, y_test = train_test_split(np.arange(len(y)), np.array(y), test_size=0.2,
                                                                random_state=i)
            

            # building the estimator
            pipe = Pipeline([('rbf_mmd', RBF_Sig_MMD_Kernel(K_full=list_kernels[n])),
                    ('clf', clf())
                    ])
            # parameter search
            model = GridSearchCV(pipe, parameters, refit=True,verbose=0, n_jobs=-1, scoring='neg_mean_squared_error', cv=cv,
                                 error_score=np.nan)

            model.fit(ind_train, y_train)
            best_scores_train[n] = -model.best_score_
            y_pred = model.predict(ind_test)
        
            results_tmp[n]={'pred':y_pred,'true':y_test}
            MSE_test[n] = mean_squared_error(y_pred, y_test)
            models.append(model)

        # pick the model with the best performances on the train set
        best_score = 100000
        index = None
        for n,alpha in enumerate(alphas):
            if (best_scores_train[n] < best_score):
                best_score = best_scores_train[n]
                index = n

        scores[i] = MSE_test[index]
        results[i] = results_tmp[index]
       
        print('best scaling parameter (cv on the train set): ', alphas[index])
        print('best mse score (cv on the train set): ', best_scores_train[index])
    return scores.mean(), scores.std(), results

def model_sketch(X, y, k2='rbf', alphas=[0.5], depths=[2], ncompos = [100], rbf=True, dyadic_order=1, ll=None, at=False, mode='krr', NUM_TRIALS=1, cv=3, grid={}):
    """Performs a kernel based distribution regression on ensembles (of possibly unequal cardinality)
       of univariate or multivariate time-series (of possibly unequal lengths)
       Input:
              X (list): list of lists such that
                        - len(X) = n_samples
                        - for any i, X[i] is a list of arrays of shape (length, dim)
                        - for any j, X[i][j] is an array of shape (length, dim)
              y (np.array): array of shape (n_samples,) 
              alphas (list of floats): RBF kernel scaling parameter to cross-validate
              dyadic_order (int): dyadic order of PDE solver
              ll (list of ints): dimensions to lag (set to None by default)
              at (bool): if True pre-process the input path with add-time
              mode (str): "krr" -> Kernel Ridge Regression, 'svr' -> Support Vector Regresion
              NUM_TRIALS, cv : parameters for cross-validation
              grid (dict): a dictionary to specify the hyperparameter grid for the gridsearch. Unspecified entries will be set by default
       Output: mean MSE (and std) (both scalars) of regression performance on a cv-folds cross-validation (NUM_TRIALS times) as well results (a dictionary containing the predicted labels and true labels)
    """
    
    use_gpu = torch.cuda.is_available()

    assert mode in ['svr', 'krr'], "mode must be either 'svr' or 'krr' "

    # possibly augment the state space of the time series
    if ll is not None:
        X = LeadLag(ll).fit_transform(X)
    if at:
        X = AddTime().fit_transform(X)
    
    if mode == 'krr':

        # default grid
        parameters = {'clf__kernel': ['precomputed'],
                      'rbf_mmd__gamma':[1e3, 1e2, 1e1, 1, 1e-1,1e-2,1e-3],
                      'clf__alpha': [1e3, 1e2, 1e1, 1e-1, 1e-2, 1e-3]}

        # check if the user has not given an irrelevant entry
        assert len(list(set(parameters.keys()) & set(grid.keys()))) == len(
            list(grid.keys())), "keys should be in " + ' '.join([str(e) for e in parameters.keys()])

        # merge the user grid with the default one
        parameters.update(grid)

        clf = KernelRidge

    else:

        # default grid
        parameters = {'clf__kernel': ['precomputed'],
                       'clf__gamma': [1e3, 1e2, 1e1, 1, 1e-1,1e-2,1e-3], 
                      'rbf_mmd__gamma':[1e3, 1e2, 1e1, 1, 1e-1,1e-2,1e-3],
                       'clf__C': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
                       }

        # check if the user has not given an irrelevant entry
        assert len(list(set(parameters.keys()) & set(grid.keys()))) == len(
            list(grid.keys())), "keys should be in " + ' '.join([str(e) for e in parameters.keys()])

        # merge the user grid with the default one
        parameters.update(grid)
        clf = SVR

    list_kernels = []
    hyperparams = list(itertools.product(alphas, depths, ncompos))
    # Precompute the Gram matrices for the different scaling parameters, to avoid recomputing them for each grid search step
    for (alpha, depth, ncompo) in hyperparams:

        mmd = np.zeros((len(X),len(X)))

        
        ES = SketchExpectedSignatureTransform(order=depth, ncompo=ncompo, rbf=rbf, lengthscale=alpha).fit_transform(X)  #(M,D)
        
        mmd = -2*ES@ES.T
        mmd += np.diag(mmd)[:,None] + np.diag(mmd)[None,:]

        if np.isnan(mmd).any():
            list_kernels.append(np.eye(len(X)))
        else:
            list_kernels.append(mmd)

    
    scores = np.zeros(NUM_TRIALS)
    results = {}
    # Loop for each trial
    for i in tqdm(range(NUM_TRIALS)):

        best_scores_train = np.zeros(len(alphas))

        # will only retain the MSE (mean + std) corresponding to the model achieving the best score (on the train set)
        # i.e. the test set is not used to decide the hyperparameters.
   
        MSE_test = np.zeros(len(alphas))
        results_tmp = {}
        models = []
        for n,(alpha, depth, ncompo) in enumerate(hyperparams):
            
            ind_train, ind_test, y_train, y_test = train_test_split(np.arange(len(y)), np.array(y), test_size=0.2,
                                                                random_state=i)
            

            # building the estimator
            pipe = Pipeline([('rbf_mmd', RBF_Sig_MMD_Kernel(K_full=list_kernels[n],k2=k2)),
                    ('clf', clf())
                    ])
            # parameter search
            model = GridSearchCV(pipe, parameters, refit=True,verbose=0, n_jobs=-1, scoring='neg_mean_squared_error', cv=cv,
                                 error_score=np.nan)

            model.fit(ind_train, y_train)
            best_scores_train[n] = -model.best_score_
            y_pred = model.predict(ind_test)
        
            results_tmp[n]={'pred':y_pred,'true':y_test}
            MSE_test[n] = mean_squared_error(y_pred, y_test)
            models.append(model)

        # pick the model with the best performances on the train set
        best_score = 100000
        index = None
        for n,(alpha, depth, ncompo) in enumerate(hyperparams):
            if (best_scores_train[n] < best_score):
                best_score = best_scores_train[n]
                index = n

        scores[i] = MSE_test[index]
        results[i] = results_tmp[index]
       
        print('best scaling parameter (cv on the train set): ', hyperparams[index])
        print('best mse score (cv on the train set): ', best_scores_train[index])
    return scores.mean(), scores.std(), results


def model_higher_rank(X, y, alphas0=[0.5], alphas1=[0.5], lambdas=[0.1], rbf=True, dyadic_order0=1, dyadic_order1=1, ll=None, at=False, mode='krr', NUM_TRIALS=1, cv=3, grid={}):
    """Performs a kernel based distribution classification on ensembles (of possibly unequal cardinality)
       of univariate or multivariate time-series (of possibly unequal lengths)
       We use the RBF embedding throughout. 
       Input:
              X (list): list of lists such that
                        - len(X) = n_samples
                        - for any i, X[i] is a list of arrays of shape (length, dim)
                        - for any j, X[i][j] is an array of shape (length, dim)
              y (np.array): array of shape (n_samples,)
              rank (int): order of the DR kernel 
              alphas0 (list of floats): RBF kernel scaling parameter to cross-validate for order 0
              alphas1 (list of floats): RBF kernel scaling parameter to cross-validate for order 1
              lambdas (list of floats): conditional signature mean embedding regularizer to cross-validate for order 1 
              dyadic_order0 (int): dyadic order of PDE solver for order 0
              dyadic_order1 (int): dyadic order of PDE solver for order 1
              ll (list of ints): dimensions to lag (set to None by default)
              at (bool): if True pre-process the input path with add-time
              mode (str): "krr" -> Kernel Ridge Regression, 'svr' -> Support Vector Regresion
              NUM_TRIALS, cv : parameters for cross-validation
              grid (dict): a dictionary to specify the hyperparameter grid for the gridsearch. Unspecified entries will be set by default
       Output: mean MSE (and std) (both scalars) of regression performance on a cv-folds cross-validation (NUM_TRIALS times) as well results (a dictionary containing the predicted labels and true labels)
    """
    
    use_gpu = torch.cuda.is_available()

    assert mode in ['svr', 'krr'], "mode must be either 'svr' or 'krr' "

    # possibly augment the state space of the time series
    if ll is not None:
        X = LeadLag(ll).fit_transform(X)
    if at:
        X = AddTime().fit_transform(X)
    
    if mode == 'krr':

        # default grid
        parameters = {'clf__kernel': ['precomputed'],
                      'rbf_mmd__gamma':[1e3, 1e2, 1e1, 1, 1e-1,1e-2,1e-3],
                      'clf__alpha': [1e3, 1e2, 1e1, 1e-1, 1e-2, 1e-3]}

        # check if the user has not given an irrelevant entry
        assert len(list(set(parameters.keys()) & set(grid.keys()))) == len(
            list(grid.keys())), "keys should be in " + ' '.join([str(e) for e in parameters.keys()])

        # merge the user grid with the default one
        parameters.update(grid)

        clf = KernelRidge

    else:

        # default grid
        parameters = {'clf__kernel': ['precomputed'],
                       'clf__gamma': [1e3, 1e2, 1e1, 1, 1e-1,1e-2,1e-3], 
                      'rbf_mmd__gamma':[1e3, 1e2, 1e1, 1, 1e-1,1e-2,1e-3],
                       'clf__C': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
                       }

        # check if the user has not given an irrelevant entry
        assert len(list(set(parameters.keys()) & set(grid.keys()))) == len(
            list(grid.keys())), "keys should be in " + ' '.join([str(e) for e in parameters.keys()])

        # merge the user grid with the default one
        parameters.update(grid)
        clf = SVR

    list_kernels = []
    hyperparams = list(itertools.product(alphas0, alphas1, lambdas))
    # Precompute the Gram matrices for the different scaling parameters, to avoid recomputing them for each grid search step
    for (scale1,scale2,lambda_) in hyperparams:

        mmd = np.zeros((len(X),len(X)))
        if rbf:
            static_kernel = sigkernel.RBFKernel(sigma=scale1,add_time=X[0][0].shape[0]-1)
            static_kernel_1 = sigkernel.RBFKernel(sigma=scale2,add_time=X[0][0].shape[0]-1)
        else:
            static_kernel = sigkernel.LinearKernel(add_time=X[0][0].shape[0]-1)
            static_kernel_1 = sigkernel.LinearKernel(add_time=X[0][0].shape[0]-1)
        signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order0, static_kernel_1=static_kernel_1, dyadic_order_1=dyadic_order1)

        for i in trange(len(X)):
            mu = torch.tensor(X[i])
            if use_gpu:
                mu = mu.cuda()
            for j in range(i,len(X)):
                nu = torch.tensor(X[j])
                if use_gpu:
                    nu = nu.cuda()
                mmd[i,j] = signature_kernel.compute_mmd_rank_1(mu,nu,lambda_=lambda_)
                mmd[j,i] = mmd[i,j]
        if np.isnan(mmd).any():
            list_kernels.append(np.eye(len(X)))
        else:
            list_kernels.append(mmd)

    
    scores = np.zeros(NUM_TRIALS)
    results = {}
    # Loop for each trial
    for i in tqdm(range(NUM_TRIALS)):

        best_scores_train = np.zeros(len(hyperparams))

        # will only retain the MSE (mean + std) corresponding to the model achieving the best score (on the train set)
        # i.e. the test set is not used to decide the hyperparameters.
   
        MSE_test = np.zeros(len(hyperparams))
        results_tmp = {}
        models = []
        for n,(scale1, scale2, lambda_) in enumerate(hyperparams):
            
            ind_train, ind_test, y_train, y_test = train_test_split(np.arange(len(y)), np.array(y), test_size=0.2,
                                                                random_state=i)
            
            # building the estimator
            pipe = Pipeline([('rbf_mmd', RBF_Sig_MMD_Kernel(K_full=list_kernels[n])),
                    ('clf', clf())
                    ])
            # parameter search
            model = GridSearchCV(pipe, parameters, refit=True,verbose=0, n_jobs=-1, scoring='neg_mean_squared_error', cv=cv,
                                 error_score=np.nan)

            model.fit(ind_train, y_train)
            best_scores_train[n] = -model.best_score_
            # print(model_.best_params_)
            y_pred = model.predict(ind_test)
        
            results_tmp[n]={'pred':y_pred,'true':y_test}
            MSE_test[n] = mean_squared_error(y_pred, y_test)
            models.append(model)
        # pick the model with the best performances on the train set
        best_score = 100000
        index = None
        for n,(scale1, scale2, lambda_) in enumerate(hyperparams):
            if (best_scores_train[n] < best_score):
                best_score = best_scores_train[n]
                index = n

        scores[i] = MSE_test[index]
        results[i] = results_tmp[index]
       
        print('best scaling parameter (cv on the train set): ', hyperparams[index])
        print('best mse score (cv on the train set): ', best_scores_train[index])
    return scores.mean(), scores.std(), results


def model_higher_rank_sketch(X, y, depths1=[2], ncompos1=[20], rbf1=True, alphas1=[1], lambdas_=[10], depths2=[2], ncompos2=[20], rbf2=True, alphas2=[1], ll=None, at=False,  NUM_TRIALS=1, cv=3, grid={}):
    """Performs a kernel based distribution classification on ensembles (of possibly unequal cardinality)
       of univariate or multivariate time-series (of possibly unequal lengths)
       We use the RBF embedding throughout. 
       Input:
              X (list): list of lists such that
                        - len(X) = n_samples
                        - for any i, X[i] is a list of arrays of shape (length, dim)
                        - for any j, X[i][j] is an array of shape (length, dim)
              y (np.array): array of shape (n_samples,)
              rank (int): order of the DR kernel 
              alphas0 (list of floats): RBF kernel scaling parameter to cross-validate for order 0
              alphas1 (list of floats): RBF kernel scaling parameter to cross-validate for order 1
              lambdas (list of floats): conditional signature mean embedding regularizer to cross-validate for order 1 
              dyadic_order0 (int): dyadic order of PDE solver for order 0
              dyadic_order1 (int): dyadic order of PDE solver for order 1
              ll (list of ints): dimensions to lag (set to None by default)
              at (bool): if True pre-process the input path with add-time
              mode (str): "krr" -> Kernel Ridge Regression, 'svr' -> Support Vector Regresion
              NUM_TRIALS, cv : parameters for cross-validation
              grid (dict): a dictionary to specify the hyperparameter grid for the gridsearch. Unspecified entries will be set by default
       Output: mean MSE (and std) (both scalars) of regression performance on a cv-folds cross-validation (NUM_TRIALS times) as well results (a dictionary containing the predicted labels and true labels)
    """
    
    use_gpu = torch.cuda.is_available()

    # possibly augment the state space of the time series
    if ll is not None:
        X = LeadLag(ll).fit_transform(X)
    if at:
        X = AddTime().fit_transform(X)
    
    # default grid
    parameters = {'clf__kernel': ['precomputed'],
                    'rbf_mmd__gamma':[1e3, 1e2, 1e1, 1, 1e-1,1e-2,1e-3],
                    'clf__alpha': [1e3, 1e2, 1e1, 1e-1, 1e-2, 1e-3]}

    # check if the user has not given an irrelevant entry
    assert len(list(set(parameters.keys()) & set(grid.keys()))) == len(
        list(grid.keys())), "keys should be in " + ' '.join([str(e) for e in parameters.keys()])

    # merge the user grid with the default one
    parameters.update(grid)

    clf = KernelRidge

    list_kernels = []
    hyperparams = list(itertools.product(depths1,ncompos1, alphas1, lambdas_, depths2, ncompos2, alphas2))
    # Precompute the Gram matrices for the different scaling parameters, to avoid recomputing them for each grid search step
    for (depth1,ncompo1, alpha1, lambda_, depth2, ncompo2, alpha2) in hyperparams:

        pwCKME = SketchpwCKMETransform(order=depth1, ncompo=ncompo1, rbf=rbf1, lengthscale=alpha1, lambda_=lambda_).fit_transform(X) 
        pwCKME = AddTime().fit_transform(pwCKME)
        ES = SketchExpectedSignatureTransform(order=depth2, ncompo=ncompo2, rbf=rbf2, lengthscale=alpha2).fit_transform(np.array(pwCKME))  #(M,D)
        if k2=='rbf':
            mmd = -2*ES@ES.T
            mmd += np.diag(mmd)[:,None] + np.diag(mmd)[None,:]
        elif k2=='lin':
            mmd = ES@ES.T
        if np.isnan(mmd).any():
            list_kernels.append(np.eye(len(X)))
        else:
            list_kernels.append(mmd)


    
    scores = np.zeros(NUM_TRIALS)
    results = {}
    # Loop for each trial
    for i in tqdm(range(NUM_TRIALS)):

        best_scores_train = np.zeros(len(hyperparams))

        # will only retain the MSE (mean + std) corresponding to the model achieving the best score (on the train set)
        # i.e. the test set is not used to decide the hyperparameters.
   
        MSE_test = np.zeros(len(hyperparams))
        results_tmp = {}
        models = []
        for n,(depth1,ncompo1, alpha1, lambda_, depth2, ncompo2, alpha2) in enumerate(hyperparams):
            
            ind_train, ind_test, y_train, y_test = train_test_split(np.arange(len(y)), np.array(y), test_size=0.2,
                                                                random_state=i)
            
            # building the estimator
            pipe = Pipeline([('rbf_mmd', RBF_Sig_MMD_Kernel(K_full=list_kernels[n])),
                    ('clf', clf())
                    ])
            # parameter search
            model = GridSearchCV(pipe, parameters, refit=True,verbose=0, n_jobs=-1, scoring='neg_mean_squared_error', cv=cv,
                                 error_score=np.nan)

            model.fit(ind_train, y_train)
            best_scores_train[n] = -model.best_score_
            # print(model_.best_params_)
            y_pred = model.predict(ind_test)
        
            results_tmp[n]={'pred':y_pred,'true':y_test}
            MSE_test[n] = mean_squared_error(y_pred, y_test)
            models.append(model)
        # pick the model with the best performances on the train set
        best_score = 100000
        index = None
        for n,(depth1,ncompo1, alpha1, lambda_, depth2, ncompo2, alpha2) in enumerate(hyperparams):
            if (best_scores_train[n] < best_score):
                best_score = best_scores_train[n]
                index = n

        scores[i] = MSE_test[index]
        results[i] = results_tmp[index]
       
        print('best scaling parameter (cv on the train set): ', hyperparams[index])
        print('best mse score (cv on the train set): ', best_scores_train[index])
    return scores.mean(), scores.std(), results




class RBF_Sig_MMD_Kernel(BaseEstimator, TransformerMixin):
    def __init__(self, K_full=None,gamma=1.0,k2='rbf'):
        super(RBF_Sig_MMD_Kernel, self).__init__()
        self.gamma = gamma
        self.K_full = K_full
        self.k2 = k2

    def transform(self, X):
        K = self.K_full[X][:,self.ind_train].copy()
        if self.k2 =='rbf':
            return np.exp(-self.gamma*K) 
        elif self.k2=='lin':
            return self.gamma*K

    def fit(self, X, y=None, **fit_params):
        self.ind_train = X
        return self