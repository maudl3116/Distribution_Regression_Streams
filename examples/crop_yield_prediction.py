import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
sys.path.append("../")
from utils.addtime import AddTime, LeadLag
import utils.experiments as experiments
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import iisignature
import utils.signature_features as signature_features

def get_e_sig(bags):

    # signature spec
    level_sig = 4
    add_time_tf = AddTime()
    lead_lag_tf = LeadLag([1])

    # to store the input/output pairs
    expected_sigs = []

    for bag in bags:
        a = add_time_tf.fit_transform(bag)
        a = np.array(a)
        a = lead_lag_tf.fit_transform(a)
        a = np.array(a)
        expected_sig = np.mean(iisignature.sig(a, level_sig), axis=0)
        expected_sigs.append(expected_sig)

    return expected_sigs


def get_sig_e_sig(bags):
    # signature spec
    add_time_tf = AddTime()
    lead_lag_tf = LeadLag([1])
    sig_level1 = 3
    sig_level2 = 2

    # to store the input/output pairs
    sig_expected_sigs = []

    for bag in bags:
        a = add_time_tf.fit_transform(bag)
        a = np.array(a)  # a is of shape (N_fields,T,2)
        a = lead_lag_tf.fit_transform(a)
        a = np.array(a)

        expected_pathwise_sig = signature_features.scaled_pathwise_expected_iisignature([a],sig_level1)
        features = expected_pathwise_sig[0]
        signatures = iisignature.sig(features,sig_level2)
        sig_expected_sigs.append(signatures)

    return sig_expected_sigs

def rbf_e_sig(train_indices_list,test_indices_list,input_,y_):

    expected_sigs = get_e_sig(input_)

    r2_test_list = []
    r2_train_list = []
    rmse_train_list = []
    rmse_test_list = []
    #lengthscales = []

    for fold in range(len(train_indices_list)):

        train_indices = train_indices_list[fold]
        test_indices = test_indices_list[fold]

        features = np.array(expected_sigs)[:, :]

        # # Scale the (expected) signature features
        scaler = StandardScaler()
        to_fit = [features[i] for i in train_indices]
        scaler.fit(to_fit)
        features = scaler.transform(features)

        # # Precompute the Gram matrix
        K_precomputed = experiments.precompute_K(features)

        rmse_train, r2_train, rmse_test, r2_test = experiments.experiment_precomputed(K_precomputed, y_, train_indices, test_indices,
                                                        param_init=[0, 30, 0], RBF=True, plot=False)
        # a,b,c,d,l = experiments.experiment_ARD(features,np.array([labels[key] for key in labels.keys()])[:,None],3,level_sig,train_indices,test_indices,param_init=[0,0,0],RBF=True,plot=True,full=True)
        # a,b,c,d = experiments.experiment_ARD(features,np.array([labels[key] for key in labels.keys()])[:,None],2,level_sig,train_indices,test_indices,param_init=[-2,10,0],RBF=True,plot=True)
        # lengthscales.append(l)

        rmse_train_list.append(rmse_train)
        r2_train_list.append(r2_train)
        rmse_test_list.append(rmse_test)
        r2_test_list.append(r2_test)

    return rmse_train_list, r2_train_list,rmse_test_list,r2_test_list




def pathwise_e_sig(train_indices_list,test_indices_list,input_,y_):

    sig_expected_sigs = get_sig_e_sig(input_)

    r2_test_list = []
    r2_train_list = []
    rmse_train_list = []
    rmse_test_list = []


    for fold in range(len(train_indices_list)):

        train_indices = train_indices_list[fold]
        test_indices = test_indices_list[fold]

        features = np.array(sig_expected_sigs)[:, :]

        # # Scale the (expected) signature features
        # scaler = StandardScaler()
        # to_fit = [features[i] for i in train_indices]
        # scaler.fit(to_fit)
        # features = scaler.transform(features)

        # # Precompute the Gram matrix
        K_precomputed = experiments.precompute_K(features)

        rmse_train, r2_train, rmse_test, r2_test = experiments.experiment_precomputed(K_precomputed, y_, train_indices, test_indices,
                                                        param_init=[0, 30, 0], RBF=False, plot=False)

        rmse_train_list.append(rmse_train)
        r2_train_list.append(r2_train)
        rmse_test_list.append(rmse_test)
        r2_test_list.append(r2_test)

    return rmse_train_list, r2_train_list,rmse_test_list,r2_test_list

def rbf_rbf(train_indices_list,test_indices_list,input_,y_):

    r2_test_list = []
    r2_train_list = []
    rmse_train_list = []
    rmse_test_list = []


    for fold in range(len(train_indices_list)):

        train_indices = train_indices_list[fold]
        test_indices = test_indices_list[fold]

        input_ordered = []
        labels_ordered = []
        for i in train_indices:
            input_ordered.append(input_[i])
            labels_ordered.append(y_[i, 0])
        for i in test_indices:
            input_ordered.append(input_[i])
            labels_ordered.append(y_[i, 0])

        rmse_train, r2_train, rmse_test, r2_test = experiments.naive_experiment_arbitrary(input_ordered, np.array(labels_ordered)[:, None],
                                                                len(train_indices), ARD=False, RBF_top=True,
                                                                param_init=[10, 0, 0, 10], plot=False,
                                                                device=torch.device("cuda"))
        rmse_train_list.append(rmse_train)
        r2_train_list.append(r2_train)
        rmse_test_list.append(rmse_test)
        r2_test_list.append(r2_test)

    return rmse_train_list, r2_train_list, rmse_test_list, r2_test_list

def lin_rbf(train_indices_list,test_indices_list,input_,y_):

    r2_test_list = []
    r2_train_list = []
    rmse_train_list = []
    rmse_test_list = []


    for fold in range(len(train_indices_list)):

        train_indices = train_indices_list[fold]
        test_indices = test_indices_list[fold]

        input_ordered = []
        labels_ordered = []
        for i in train_indices:
            input_ordered.append(input_[i])
            labels_ordered.append(y_[i, 0])
        for i in test_indices:
            input_ordered.append(input_[i])
            labels_ordered.append(y_[i, 0])

        rmse_train, r2_train, rmse_test, r2_test = experiments.naive_experiment_arbitrary(input_ordered, np.array(labels_ordered)[:, None],
                                                                len(train_indices), ARD=False, RBF_top=False,
                                                                param_init=[10, 0, 0, 10], plot=False,
                                                                device=torch.device("cuda"))
        rmse_train_list.append(rmse_train)
        r2_train_list.append(r2_train)
        rmse_test_list.append(rmse_test)
        r2_test_list.append(r2_test)

    return rmse_train_list, r2_train_list, rmse_test_list, r2_test_list



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--nb_folds', type=int, default=3)
    parser.add_argument('--max_nb_items', type=int, default=30)
    parser.add_argument('--rbf_e_sig', default=True, action='store_true')  # False by default
    parser.add_argument('--pathwise_e_sig', default=True, action='store_true')  # False by default
    parser.add_argument('--rbf_rbf', default=True, action='store_true')  # False by default
    parser.add_argument('--lin_rbf', default=True, action='store_true')  # False by default
    parser.add_argument('--rbf_e_path', default=True, action='store_true')  # False by default
    args = parser.parse_args()

    max_nb_pixels = args.max_nb_pixels
    nb_folds = args.nb_folds

    # load the bags of time series
    input_list = pickle.load(open('../data/crops/bags.obj', 'rb'))
    y_ = pickle.load(open('../data/crops/labels.obj', 'rb'))

    # cap the number of items in each bag (for memory constraints of baselines methods)
    input_ = []
    for i in range(len(input_list)):
        bag = input_list[i]
        nb = min(max_nb_pixels, len(bag))
        input_.append(bag[:nb])

    # create folds
    y = y_[:, 0]
    bins = np.linspace(-0.1 + min(y), max(y) + 0.1, 20)
    y_binned = np.digitize(y, bins)

    train_indices_list = []
    test_indices_list = []
    skf = StratifiedKFold(n_splits=nb_folds, shuffle=True, random_state=0)
    skf.get_n_splits(np.arange(len(y)), y_binned)
    for train_index, test_index in skf.split(np.arange(len(y)), y_binned):
        train_indices_list.append(train_index)
        test_indices_list.append(test_index)

    # dictionary to save the results
    dico = {}
    # evaluate methods

    if args.rbf_e_sig:
        rmse_train, r2_train, rmse_test, r2_test = rbf_e_sig(train_indices_list,test_indices_list,input_,y_)
        dico['rbf_e_sig']={'rmse_train':rmse_train,'r2_train':r2_train,'rmse_test':rmse_test,'r2_test':r2_test}

    if args.pathwise_e_sig:
        rmse_train, r2_train, rmse_test, r2_test = pathwise_e_sig(train_indices_list, test_indices_list, input_, y_)
        dico['pathwise_e_sig'] = {'rmse_train': rmse_train, 'r2_train': r2_train, 'rmse_test': rmse_test, 'r2_test': r2_test}

    if args.rbf_rbf:
        rmse_train, r2_train, rmse_test, r2_test = rbf_rbf(train_indices_list, test_indices_list, input_, y_)
        dico['rbf_rbf'] = {'rmse_train': rmse_train, 'r2_train': r2_train, 'rmse_test': rmse_test, 'r2_test': r2_test}

    if args.lin_rbf:
        rmse_train, r2_train, rmse_test, r2_test = lin_rbf(train_indices_list, test_indices_list, input_, y_)
        dico['lin_rbf'] = {'rmse_train': rmse_train, 'r2_train': r2_train, 'rmse_test': rmse_test, 'r2_test': r2_test}

    pickle.dump(dico,open('results.obj','wb'))



