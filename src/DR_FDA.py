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
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import pairwise_distances
from sklearn.metrics import pairwise_kernels
from sklearn.decomposition import PCA

def model(X, y, name ='sqr', alphas=[0.5], ll=None, at=False, mode='krr', NUM_TRIALS=1, cv=3, grid={}):
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

        for i in trange(len(X)):
            mu = np.array(X[i])
            mu = mu.reshape(mu.shape[0],-1)
            for j in range(i,len(X)):
                nu = np.array(X[j])
                nu = nu.reshape(mu.shape[0],-1)
                if name =='id':
                    gram = K_ID(mu,nu,gamma=alpha)
                if name =='sqr':
                    gram = K_SQR(mu,nu,gamma=alpha)
                elif name=='cov':
                    gram = K_COV(mu,nu,gamma=alpha)
                elif name=='fpca':
                    gram = K_FPCA(mu,nu,gamma=alpha)
                elif name=='cexp':
                    xx = CEXP(mu,n_freqs=20,l=np.sqrt(10)) 
                    yy = CEXP(nu,n_freqs=20,l=np.sqrt(10)) 
                    gram = K_ID(xx,yy,gamma=alpha)
                N = mu.shape[0]
                Kxx = gram[:N,:N]
                Kyy = gram[N:,N:]
                Kxy = gram[:N,N:]
                mmd[i,j] = (np.mean(Kxx)-2*np.mean(Kxy)+np.mean(Kyy))
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
            pipe = Pipeline([('rbf_mmd', RBF_MMD_Kernel(K_full=list_kernels[n])),
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

def K_SQR(X,Y,gamma = 1):
    """
    Forms the kernel matrix K for the two sample test using the SE-T kernel with bandwidth gamma
    where T is the map which sends x -> (x,x^{2}) in the Cartesian product of L^{2} with itself.
    
    Parameters:
    X - (n_samples,n_obs) array of samples from the first distribution 
    Y - (n_samples,n_obs) array of samples from the second distribution 
    gamma - bandwidth for the kernel to be used on the two norms, if -1 then median heuristic 
            is used to pick a different gamma for each norm, if gamma = 0 then median heuristic
            is used to pick a single gamma for each norm.
            
    Returns:
    K - matrix formed from the kernel values of all pairs of samples from the two distributions
    """
    n_obs = X.shape[1]
    XY = np.vstack((X,Y))
    dist_mat_1 = (1/np.sqrt(n_obs))*pairwise_distances(XY, metric='euclidean')
    dist_mat_2 = (1/np.sqrt(n_obs))*pairwise_distances(XY**2, metric='euclidean')
    dist_mat = dist_mat_1 + dist_mat_2
    if gamma == 0:
        gamma = np.median(dist_mat[dist_mat > 0])
        K = np.exp(-0.5*(1/gamma**2)*dist_mat**2)
        return K
    if gamma == -1:
        gamma_1 = np.median(dist_mat_1[dist_mat_1 > 0])
        gamma_2 = np.median(dist_mat_2[dist_mat_2 > 0])
        K = np.exp(-0.5*((1/gamma_1**2)*dist_mat_1**2 + (1/gamma_2**2)*dist_mat_2**2))
        return K
    K = np.exp(-0.5*((1/gamma**2)*(dist_mat**2)))
    return K

def K_ID(X,Y,gamma=1):
    """
    Forms the kernel matrix K for the two sample test using the SE-T kernel with bandwidth gamma
    where T is the identity operator
    
    Parameters:
    X - (n_samples,n_obs) array of samples from the first distribution 
    Y - (n_samples,n_obs) array of samples from the second distribution 
    gamma - bandwidth for the kernel, if -1 then median heuristic is used to pick gamma
    
    Returns:
    K - matrix formed from the kernel values of all pairs of samples from the two distributions
    """
    n_obs = X.shape[1]
    XY = np.vstack((X,Y))
    dist_mat = (1/np.sqrt(n_obs))*pairwise_distances(XY, metric='euclidean')
    if gamma == -1:
        gamma = np.median(dist_mat[dist_mat > 0])
   
    K = np.exp(-0.5*(1/gamma**2)*(dist_mat**2))
    return K

def K_COV(X,Y,gamma=1):
    """
    Forms the kernel matrix K for the two sample test using the COV kernel
    
    Parameters:
    X - (n_samples,n_obs) array of samples from the first distribution 
    Y - (n_samples,n_obs) array of samples from the second distribution 
    gamma - dummy variable noot used in function, is an input for ease of compatability with other kernels
    
    Returns:
    K - matrix formed from the kernel values of all pairs of samples from the two distributions
    """    
    n_obs = X.shape[1]
    XY = np.vstack((X,Y))
    return ((1/n_obs)*np.dot(XY,XY.T))**2

def K_FPCA(X,Y,gamma = 1,n_comp = 0.95):
    """
    Forms the kernel matrix K for the two sample test using the SE-T kernel with bandwidth gamma
    where T is the FPCA decomposition operator
    
    Parameters:
    X - (n_samples,n_obs) array of samples from the first distribution 
    Y - (n_samples,n_obs) array of samples from the second distribution 
    gamma - bandwidth for the kernel, if -1 then median heuristic is used to pick gamma
    n_comp - number of principal components to compute. If in (0,1) then it is the explained variance level
    
    Returns:
    K - matrix formed from the kernel values of all pairs of samples from the two distributions
    """
    n_obs = X.shape[1]
    XY = np.vstack((X,Y))
    e_vals,e_funcs = FPCA(XY,n_comp = n_comp)
    scaled_e_funcs = e_funcs*np.sqrt(e_vals[:,np.newaxis])
    XY_e = (1/n_obs)*np.dot(XY,scaled_e_funcs.T)
    dist_mat = pairwise_distances(XY_e,metric='euclidean')
    if gamma == -1:
        gamma = np.median(dist_mat[dist_mat > 0])
    K = np.exp(-0.5*(1/gamma**2)*(dist_mat**2))
    return K

class RBF_MMD_Kernel(BaseEstimator, TransformerMixin):
    def __init__(self, K_full=None,gamma=1.0):
        super(RBF_MMD_Kernel, self).__init__()
        self.gamma = gamma
        self.K_full = K_full

    def transform(self, X):
        K = self.K_full[X][:,self.ind_train].copy()
        return np.exp(-self.gamma*K) 

    def fit(self, X, y=None, **fit_params):
        self.ind_train = X
        return self


def cos_exp_kernel(x,y,n_freqs = 5,l=1):
    """
    The c-exp kernel
    
    Parameters:
    x,y - inputs 
    n_freqs - number of frequencies to include in the sum
    l- bandwidth of the kernel
    
    Returns:
    Kernel values given x,y
    """
    
    cos_term = np.sum([np.cos(2*np.pi*n*(x-y)) for n in range(n_freqs)])
    return cos_term*np.exp(-(0.5/(l**2))*(x-y)**2)

def CEXP(X,n_freqs = 20,l=np.sqrt(10)):
    """
    Transforms an array of function values using the integral operator induced by the cos-exp kernel. 
    The function values are assumed to be on [0,1]
    
    Parameters:
    X - (n_samples,n_obs) array of function values
    n_freqs - number of frequencies to include in the sum
    l- bandwidth of the kernel
    
    Returns:
    cos_exp_X - (n_samples,n_obs) array of function values where each function has been passed
                through the integral operator induced by the cos-exp kernel
    """
    n_obs = X.shape[1]
    obs_grid = np.linspace(0,1,n_obs)
    T_mat = pairwise_kernels(obs_grid.reshape(-1,1), metric = cos_exp_kernel, n_freqs = n_freqs,l=l)
    cos_exp_X = (1./n_obs)*np.dot(X,T_mat)
    return cos_exp_X


def FPCA(X,n_comp = 0.95):
    """
    Computes principal components of given data up to a specified explained variance level
    
    Parameters:
    X - (n_samples,n_obs) array of function values
    n_comp - number of principal components to compute. If in (0,1) then it is the explained variance level
    
    Returns:
    Normalised eigenvalues and eigenfunctions
    """
    n_points = np.shape(X)[1]
    pca = PCA(n_components = n_comp)
    pca.fit(X)
    return (1/n_points)*pca.explained_variance_,pca.components_