import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import QuantileTransformer

def split_standardize(y,data,standardized=True,method = 'standard'):
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
        train, test, _, _ = train_test_split(np.arange(len(y)), np.array(y), shuffle=True, test_size=1. / 4,
                                             random_state=0)
    elif method == 'stratify':
        bins = np.linspace(-0.01+min(y[:,0]), max(y[:,0])+0.01, 5)

        y_binned = np.digitize(y[:,0], bins)



        train, test, _, _ = train_test_split(np.arange(len(y)), np.array(y), shuffle=True, test_size=1./4,
                                         stratify=y_binned, random_state=0)

    # 3. STANDARDIZE
    if standardized:


        scaler = StandardScaler()
        to_fit = np.array([data[i].copy() for i in train]).reshape(-1, data[0][0].shape[1])
        scaler.fit(to_fit)

        to_transform = np.array([data[i].copy() for i in range(len(y))]).reshape(-1, data[0][0].shape[1])
        data_scaled = scaler.transform(to_transform).reshape(np.array(data).shape)

        scaler = QuantileTransformer(n_quantiles=10, random_state=0)
        to_fit = y[train]
        scaler.fit(to_fit)

        y_scaled = scaler.transform(y.copy())

    else:
        data_scaled = data
        y_scaled = y

    return data_scaled, y_scaled, train, test