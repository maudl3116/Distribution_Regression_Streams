import numpy as np
import copy
import random
import doctest

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    import imp

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import as_float_array

import tjl_dense_numpy_tensor as te
import tjl_hall_numpy_lie as li
from tosig import *

class AddTime(BaseEstimator, TransformerMixin):
    # sklearn-type estimator to add time as an extra dimension of a D-dimensional path.
    # Note that the input must be a list of arrays (i.e. a list of D-dimensional paths)

    """
    >>> total_observations = 10
    >>> path_dim = 3
    >>> brownian_paths = []
    >>> for k in range(total_observations): brownian_paths.append(te.brownian(100, path_dim))
    >>> ind = np.random.randint(0, path_dim-1)
    >>> add_time = AddTime()
    >>> X_addtime = add_time.fit_transform(brownian_paths)
    >>> d = 2
    >>> stream2sig(X_addtime[ind], d).shape[0] == 1+(path_dim+1)+(path_dim+1)**2
    True
    """
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

class Reversion(BaseEstimator, TransformerMixin):
    # sklearn-type estimator to compute the Reverse transform of a D-dimensional path.
    # Note that the input must be a list of arrays (i.e. a list of D-dimensional paths)

    """
    >>> total_observations = 10
    >>> path_dim = 3
    >>> brownian_paths = []
    >>> for k in range(total_observations): brownian_paths.append(te.brownian(100, path_dim))
    >>> reverse = Reversion()
    >>> X_reversed = reverse.fit_transform(brownian_paths)
    >>> np.sum([np.sum(np.subtract(i,j[::-1])) for i, j in zip(brownian_paths, X_reversed)]) < 1e-30
    True
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [as_float_array(x[::-1]) for x in X]

class LeadLag(BaseEstimator, TransformerMixin):
    # sklearn-type estimator to compute the Lead-Lag transform of a D-dimensional path.
    # Note that the input must be a list of arrays (i.e. a list of D-dimensional paths)

    """
    >>> total_observations = 10
    >>> path_dim = 3
    >>> brownian_paths = []
    >>> for k in range(total_observations): brownian_paths.append(te.brownian(100, path_dim))
    >>> ind = np.random.randint(0, path_dim-1)
    >>> dimensions_to_lag = [0, 2]
    >>> lead_lag = LeadLag(dimensions_to_lag)
    >>> X_leadlag = lead_lag.fit_transform(brownian_paths)
    >>> d = 2
    >>> stream2sig(X_leadlag[ind], d).shape[0] == 1+(path_dim+len(dimensions_to_lag))+(path_dim+len(dimensions_to_lag))**2
    True
    """
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

class SignatureTransform(BaseEstimator, TransformerMixin):
    """
    sklearn-type estimator to compute the signature from a list of multi-dimensional streams.
    With this interface it is now possible to easily write scikit-learn pipelines as follows:

    from sklearn.pipeline import Pipeline
    from sklearn import linear_model

    # Create list of 10 Brownian paths of 3-dimensions
    total_observations = 10
    path_dim = 3
    brownian_paths = []
    for k in range(total_observations):
        brownian_paths.append(te.brownian(100, path_dim))

    # Random list of integers for classification 
    y = np.random.randint(0, 10, total_observations)

    # Build ML Pipeline
    pipeline = Pipeline([('lead_lag_transform', LeadLag(dimensions_to_lag=[1, 2])),
                         ('signature_transform', SignatureTransform(order=2)),
                         ('clf', linear_model.SGDClassifier(max_iter=1000)),
                        ])

    # Fit the classification model
    clf_results = pipeline.fit(brownian_paths, y)

    # Perform GridSearch to search for the best parameters
    param_grid = {'lead_lag_transform__dimensions_to_lag': [[0], [1], [2], [0,1], [0,2], [1,2]],
                  'signature_transform__order': [2, 3, 4],
                  'clf__max_iter': [1000, 5000, 10000]
                 }
    search = GridSearchCV(pipeline, param_grid, iid=False, cv=5, return_train_score=False)
    search.fit(brownian_paths, y)
    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    print(search.best_params_)
    """
    def __init__(self, order):
        if not isinstance(order, int) or order<1:
            raise NameError('The order must be a positive integer.')
        self.order = order
        
    def fit(self, X, y=None):
        return self
                            
    def transform(self, X, y=None):
        return np.array([stream2sig(item, self.order) for item in X])

class LogSignatureTransform(BaseEstimator, TransformerMixin):
    # sklearn-type estimator to compute the log-signature from a list of multi-dimensional streams.

    def __init__(self, order):
        if not isinstance(order, int) or order<1:
            raise NameError('The order must be a positive integer.')
        self.order = order
        
    def fit(self, X, y=None):
        return self
                            
    def transform(self, X, y=None):
        return np.array([stream2logsig(item, self.order) for item in X])

if __name__ == "__main__":
    doctest.testmod()