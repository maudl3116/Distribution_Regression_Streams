import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import QuantileTransformer
from addtime import AddTime, LeadLag
from numpy import math
import torch
import gpytorch
from gpytorch.kernels import RBFKernel
from torch.autograd import Variable
from gpytorch.kernels import Kernel, RBFKernel
from gpytorch.kernels.kernel import Kernel
import pandas as pd

def split_standardize(y,data,standardized_input=True,standardized_label=True,method = 'standard',random_state=0):
    # 3. GET STRATIFIED SPLITS

    y = np.array(y)

    if method=='extrapolation':
        indices = np.argsort(y[:,0])
        y = np.sort(y[:,0])[:,None]
        #y  = np.concatenate([y[:100],y[150:],y[100:150]])
        #indices = list(indices[:100])+list(indices[150:])+list(indices[100:150])
        #y = np.concatenate([y[:50],  y[150:], y[50:150]])
        #indices = list(indices[:50]) + list(indices[150:]) + list(indices[50:150])
        data = np.array([data[i].copy() for i in indices])

        train, test, _, _ = train_test_split(np.arange(len(y)), np.array(y), shuffle=False, test_size=1./3)
    elif method == 'standard':
        train, test, _, _ = train_test_split(np.arange(len(y)), np.array(y), shuffle=True, test_size=1. /5,
                                             random_state=random_state)
    elif method == 'stratify':
        #[0., 232.7, 465.4, 698.1, 930.8, 1163.5, 1396.2, 1628.9,
         #1861.6, 2094.3, 2327.])

        bins = np.linspace(-10+min(y[:,0]), max(y[:,0])+10, 3)

        y_binned = np.digitize(y[:,0], bins)



        train, test, _, _ = train_test_split(np.arange(len(y)), np.array(y), shuffle=True, test_size=1./3,
                                         stratify=y_binned, random_state=0)

    # 3. STANDARDIZE
    if standardized_input:


        scaler = StandardScaler()



        to_fit = np.concatenate([np.concatenate([data[j][i] for i in range(len(data[j]))]) for j in train ])

        scaler.fit(to_fit)

        #to_transform = np.concatenate([np.concatenate([data[j][i] for i in range(len(data[j]))]) for j in range(len(y)) ])
        #data_scaled = scaler.transform(to_transform)

        data_scaled = [data[i].copy() for i in range(len(data))]
        for i in range(len(data)):
            for j in range(len(data[i])):
                to_transform = data[i][j]
                data_scaled[i][j] = scaler.transform(to_transform)
    else:
        data_scaled = data

    if standardized_label:
        #scaler = StandardScaler()
        scaler = QuantileTransformer(n_quantiles=10, random_state=0)
        to_fit = y[train]
        scaler.fit(to_fit)

        y_scaled = scaler.transform(y.copy())

    else:
        y_scaled = y

    return data_scaled, y_scaled, train, test


def split_standardize_label(y,standardized=True,method = 'standard'):
    # 3. GET STRATIFIED SPLITS

    y = np.array(y)

    if method=='extrapolation':
        indices = np.argsort(y[:,0])
        y = np.sort(y[:,0])[:,None]
        #y  = np.concatenate([y[:100],y[150:],y[100:150]])
        #indices = list(indices[:100])+list(indices[150:])+list(indices[100:150])
        #y = np.concatenate([y[:50],  y[150:], y[50:150]])
        #indices = list(indices[:50]) + list(indices[150:]) + list(indices[50:150])
        data = np.array([data[i].copy() for i in indices])

        train, test, _, _ = train_test_split(np.arange(len(y)), np.array(y), shuffle=False, test_size=1./4)
    elif method == 'standard':
        train, test, _, _ = train_test_split(np.arange(len(y)), np.array(y), shuffle=True, test_size=1. /4,
                                             random_state=0)
    elif method == 'stratify':

        bins = np.linspace(-0.01+min(y[:,0]), max(y[:,0])+0.01, 5)

        y_binned = np.digitize(y[:,0], bins)


        train, test, _, _ = train_test_split(np.arange(len(y)), np.array(y), shuffle=True, test_size=1./4,
                                         stratify=y_binned)

    # 3. STANDARDIZE
    if standardized:

        #scaler = QuantileTransformer(n_quantiles=10, random_state=0)
        scaler = StandardScaler()
        to_fit = y[train]
        scaler.fit(to_fit)

        y_scaled = scaler.transform(y.copy())

    else:
        y_scaled = y

    return y_scaled, train, test

def standardize_real_data(data,train,test,clim_col):

    # 3. STANDARDIZE
    if True:

        indices = [e.index for e in data]
        scaler = StandardScaler()
        to_fit = np.concatenate([data[i].copy() for i in train],axis=0)
        print(to_fit.shape)

        scaler.fit(to_fit)

        to_transform = np.concatenate([data[i].copy() for i in range(len(data))],axis=0)
        #print(to_transform.shape)
        data_scaled = [scaler.transform(data[i].copy()) for i in range(len(data))]#.reshape(np.array(data).shape)
        data_scaled = [pd.DataFrame(e, index=indices[i], columns=clim_col) for i,e in enumerate(data_scaled) ]
    return data_scaled


def add_dimension(samples,add_time,lead_lag=None):

    new_samples = []

    add_time_tf = AddTime()


    for i in range(len(samples)):

        if lead_lag is not None:
            lead_lag_tf = LeadLag(lead_lag)
            augmented = lead_lag_tf.fit_transform([samples[i]])[0]
            if add_time:
                augmented = add_time_tf.fit_transform([augmented])[0]
                new_samples.append(augmented)
            else:
                new_samples.append(augmented)

        if add_time and lead_lag is None:
            new_samples.append(add_time_tf.fit_transform([samples[i]])[0])

    return np.array(new_samples)


def add_dimension2(samples,add_time,lead_lag=None):

    new_samples = []

    add_time_tf = AddTime()


    for i in range(len(samples)):

        if add_time:

            augmented = add_time_tf.fit_transform([samples[i]])[0]
            if lead_lag is not None:
                lead_lag_tf = LeadLag(lead_lag)
                augmented = lead_lag_tf.fit_transform([augmented])[0]
                new_samples.append(augmented)
            else:
                new_samples.append(augmented)

    return np.array(new_samples)
