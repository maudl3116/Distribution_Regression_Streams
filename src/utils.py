import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class AddTime(BaseEstimator, TransformerMixin):
    # sklearn-type estimator to add time as an extra dimension of a D-dimensional path.
    # Note that the input must be a list of arrays (i.e. a list of D-dimensional paths)

    def __init__(self, init_time=0., total_time=1.):
        self.init_time = init_time
        self.total_time = total_time

    def fit(self, X, y=None):
        return self

    def transform_instance(self, X):
        t = np.linspace(self.init_time, self.init_time + 1, len(X))
        return np.c_[t, X]

    def transform(self, X, y=None):
        return [self.transform_instance(x) for x in X]



class LeadLag(BaseEstimator, TransformerMixin):
    # sklearn-type estimator to compute the Lead-Lag transform of a D-dimensional path.
    # Note that the input must be a list of arrays (i.e. a list of D-dimensional paths)

    def __init__(self, dimensions_to_lag):
        if not isinstance(dimensions_to_lag, list):
            raise NameError('dimensions_to_lag must be a list')
        self.dimensions_to_lag = dimensions_to_lag

    def fit(self, X, y=None):
        return self

    def transform_instance_1D(self, x):

        lag = []
        lead = []

        for val_lag, val_lead in zip(x[:-1], x[1:]):
            lag.append(val_lag)
            lead.append(val_lag)
            lag.append(val_lag)
            lead.append(val_lead)

        lag.append(x[-1])
        lead.append(x[-1])

        return lead, lag

    def transform_instance_multiD(self, X):
        if not all(i < X.shape[1] and isinstance(i, int) for i in self.dimensions_to_lag):
            error_message = 'the input list "dimensions_to_lag" must contain integers which must be' \
                            ' < than the number of dimensions of the original feature space'
            raise NameError(error_message)

        lead_components = []
        lag_components = []

        for dim in range(X.shape[1]):
            lead, lag = self.transform_instance_1D(X[:, dim])
            lead_components.append(lead)
            if dim in self.dimensions_to_lag:
                lag_components.append(lag)

        return np.c_[lead_components + lag_components].T

    def transform(self, X, y=None):
        return [self.transform_instance_multiD(x) for x in X]


def bags_to_2D(input_):

    '''
    This function transforms input data in the form of bags of items, where each item is a D-dimensional time series
    (represented as a list of list of (T,D) matrices, where T is the length of the time series) into a 2D array to be
    compatible with what sklearn pipeline takes in input, i.e. a 2D array (n_samples, n_features). 
    
    (1) Pad each time series with its last value such that all time series have the same length.
    -> This yields lists of lists of 2D arrays (max_length,dim)
    (2) Stack the dimensions of the time series for each item
    -> This yields lists of lists of 1D arrays (max_length*dim)
    (3) Stack the items in a bag
    -> This yields lists of 1D arrays (n_item*max_length*dim)
    (4) Create "dummy items" which are time series of NaNs, such that they can be retrieved and removed at inference time
    -> This yields a 2D array (n_bags,n_max_items*max_length*dim)

       Input:
              input_ (list): list of lists of (length,dim) arrays

                        - len(X) = n_bags

                        - for any i, input_[i] is a list of n_items arrays of shape (length, dim)

                        - for any j, input_[i][j] is an array of shape (length, dim)

       Output: a 2D array of shape (n_bags,n_max_items x max_length x dim)

    '''
    
    # dimension of the state space of the time series (D)
    dim_path = input_[0][0].shape[1]

    # Find the maximum length to be able to pad the smaller time-series
    T = [e[0].shape[0] for e in input_]
    common_T = max(T)  

    new_list = []
    for bag in input_:
        new_bag = []
        for item in bag:
            # (1) if the time series is smaller than the longest one, pad it with its last value 
            if item.shape[0]<common_T:
                new_item = np.concatenate([item,np.repeat(item[-1,:][None,:],common_T - item.shape[0],axis=0)])
            else:
                new_item = item
            new_bag.append(new_item) 
        new_bag = np.array(new_bag)
        # (2) stack the dimensions for all time series in a bag
        new_bag = np.concatenate([new_bag[:, :, k] for k in range(dim_path)], axis=1) 
        new_list.append(new_bag)
    
    # (3) stack the items in each bag 
    items_stacked = [bag.flatten() for bag in new_list]

    # retrieve the maximum number of items
    max_ = [bag.shape for bag in items_stacked]
    max_ = np.max(max_)
    max_items = int(max_ / (dim_path * common_T))

    # (4) pad the vectors with nan items such that we can obtain a 2d array.
    items_naned = [np.append(bag, (max_ - len(bag)) * [np.nan]) for bag in items_stacked]  # np.nan

    X = np.array(items_naned)

    return X, max_items, common_T, dim_path


def mse(results):
    mse_vec = np.zeros(len(results))
 
    for i in range(len(results)):
        pred = results[i]['pred']
        true = results[i]['true']
        mse_vec[i]=np.mean((pred-true)**2)
    return np.mean(mse_vec), np.std(mse_vec)

def mape(results):
    mape_vec = np.zeros(len(results))
 
    for i in range(len(results)):
        pred = results[i]['pred']
        true = results[i]['true']
        mape_vec[i]=np.mean(np.abs((true - pred) / true))*100 
    return np.mean(mape_vec), np.std(mape_vec)