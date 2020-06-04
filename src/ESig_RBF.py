import numpy as np
from tqdm import tqdm as tqdm

import warnings
warnings.filterwarnings('ignore')

from utils import AddTime, LeadLag, bags_to_2D
import iisignature
from SigKerPDESolver import SigKerMMD

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


def model(X, y, depth=3, ll=[0], at=True, ss=False, mode='krr', NUM_TRIALS=5,  cv=3):
    
    """Performs a ESig-RBF Kernel based distribution regression on ensembles (of possibly unequal size)
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
    
    length = X[0][0].shape[0]
    dim = X[0][0].shape[1]  
    flag_no_trick = dim<100
       
    if flag_no_trick: # take Esig feature map
        X_esig = []
        for bag in tqdm(X):
            if ll is not None:
                bag = LeadLag(ll).fit_transform(bag)
            if at:
                bag = AddTime().fit_transform(bag)
            try:
                bag = iisignature.sig(bag, depth)
            except:
                bag = np.array([iisignature.sig(p, depth) for p in bag])
            if bag.shape[0]>0:
                X_esig.append(bag.mean(0))
        X = np.array(X_esig)
                       
    else: # Kernel trick
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
        if flag_no_trick:
            parameters = [{'clf__kernel': ['rbf'],
                           'clf__gamma': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3], 
                           'clf__alpha': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]}]
        else:
            parameters = [{'clf__kernel': ['precomputed'], 'clf__alpha':[1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3],
                           'esig_rbf__gamma':[1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]}]
        
        clf = KernelRidge
       
    else:     
        
        if flag_no_trick:
            parameters = [{'clf__kernel': ['rbf'],
                           'clf__gamma': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3], 
                           'clf__C': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]}]
            
        else:
            parameters = [{'clf__kernel': ['precomputed'], 'clf__C':[1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3],
                           'esig_rbf__gamma':[1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]}]
        
        clf = SVR

    # building RBF-RBF estimator
    if ss:
        if flag_no_trick:
            pipe = Pipeline([('std_scaler', StandardScaler()), ('clf', clf())])
        else:
            pipe = Pipeline([('std_scaler', StandardScaler()), 
                             ('esig_rbf', ESig_RBF_Kernel(dim_path = dim, max_items = max_items, size_item=dim_path*common_T)),
                             ('clf', clf())])
    else:
        if flag_no_trick:
            pipe = Pipeline([('clf', clf())])
        else:
            pipe = Pipeline([('esig_rbf', ESig_RBF_Kernel(dim_path = dim,max_items = max_items, size_item=dim_path*common_T)),
                             ('clf', clf())])
      
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


    

# The ESig_RBF kernel
class ESig_RBF_Kernel(BaseEstimator, TransformerMixin):
    def __init__(self, dim_path=None, max_items=None, size_item=None, gamma=1.0):
        super(ESig_RBF_Kernel, self).__init__()
        self.gamma = gamma
        self.size_item = size_item # L*dim_path
        self.max_items = max_items
        self.dim_path = dim_path
        

    def transform(self, X):
        '''
        X is a 2D array of shape (N_bags,n_max_items*L*dim_path)
        1) reshape X into (Nbags*max_items,L*dim_path)
        2) extract bags in the form (max_items,L,dim_path)
        3) remove nanned items
        '''
        
        alpha = 1. / (2 * self.gamma ** 2)
        x = X.reshape(-1, self.max_items,self.size_item)
   
        MMD_mat = np.zeros((X.shape[0],self.X_train_.shape[0]))    
        for i in range(X.shape[0]):
            bag_i = x[i][~np.isnan(x[i]).all(axis=1)]
            bag_i = np.split(bag_i,self.dim_path,axis=1)  # turn (N_items,dim_path*common_T) into  (N_items,common_T,dim_path)
            bag_i = np.concatenate([item[:,:,None] for item in bag_i],axis=2)
            for j in range(self.X_train_.shape[0]):
                bag_j = self.x_train[j][~np.isnan(self.x_train[j]).all(axis=1)]
                bag_j = np.split(bag_j,self.dim_path,axis=1)  # turn (N_items,dim_path*common_T) into  (N_items,common_T,dim_path)
                bag_j = np.concatenate([item[:,:,None] for item in bag_j],axis=2)
                MMD_mat[i][j] = SigKerMMD(bag_i,bag_j)
                
        return np.exp(-alpha * MMD_mat)

    def fit(self, X, y=None, **fit_params):
        self.X_train_ = X
        x_train = X.reshape(-1,self.max_items, self.size_item)  # x_train is [bag1_item1,bag1_item2,....bagN_itemN] some items are nans
        self.x_train = x_train

        return self