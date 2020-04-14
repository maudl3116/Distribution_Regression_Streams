import GP_models.GP_sig_precomputed as GP_sig
import GP_models.GP_classic as GP_naive
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import brentq as brentq
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import iisignature
#import torch
import numpy as np
#import signatory

def expected_sig(X,level_sig):
    expected_sig = []
    for bag in range(len(X)):
        sig = signatory.signature(torch.tensor(X[bag]),level_sig)
        #np.array(sig) is of shape N_items x sig_shape
        expected_sig.append(np.mean(np.array(sig),axis=0))
    return np.array(expected_sig)

def csaba_rescale(sig,d,level_sig):

    for k in range(1,level_sig+1):
        start = int(((1 - d ** k) / (1 - d)) - 1)
        end = int((1 - d ** (k + 1)) / (1 - d) - 1)
        sig[start:end] = sig[start:end]/np.sqrt(np.sum(sig[start:end]**2))

    return sig

def psi_tilde(x, M, a):
    if x <= M:
        return x
    else:
        return M + pow(M, 1. + a) * (pow(M, -a) - pow(x, -a)) / a


def get_norm_level_sig(sig, d, level_sig):

    norms = [1.]

    for k in range(1,level_sig+1):

        start = int(((1 - d ** k) / (1 - d)) - 1)
        end = int((1 - d ** (k + 1)) / (1 - d) - 1)

        norms.append(np.sum(sig[start:end]**2))

    return norms

def poly(x,psi,coef,level_sig):
    return np.sum([coef[i]*x**(2*i) for i in range(level_sig+1)])-psi



def scale_path_ilya(X,level_sig,M,a, return_norms=False):

    maxi = M*(1.+1./a)
    #print('maxi',maxi)


    bags_before = []
    bags_after = []

    for bag in range(len(X)):

        #go through each element of the bag to find its scaling:

        sig = iisignature.sig(X[bag], level_sig)

        norms = get_norm_level_sig(sig, X[bag].shape[1], level_sig)
        psi = psi_tilde(np.sum(norms),M,a)

        lambda_ = brentq(poly,0, 10000, args=(psi, norms,level_sig))

        bags_before.append(np.sum(norms))

        X[bag]=X[bag]*lambda_

        if return_norms:
            sig_after = iisignature.sig(X[bag], level_sig)
            norms_after = get_norm_level_sig(sig_after, X[bag].shape[1], level_sig)
            bags_after.append(np.sum(norms_after))

    expected_sig = iisignature.sig(X, level_sig)
    if return_norms:
        return np.array(expected_sig), bags_before, bags_after
    else:
        return np.array(expected_sig)

def scaled_expected_sig(X,level_sig,M=1,a=1, ilya_rescale=False, return_norms=False):
    expected_sig = []
    maxi = M*(1.+1./a)
    #print('maxi',maxi)


    bags_before = []
    bags_after = []

    for bag in range(len(X)):


        items_before = []
        items_after = []

        #go through each element of the bag to find its scaling:
        for i,item in enumerate(X[bag]):
            if ilya_rescale:
                sig = iisignature.sig(item, level_sig)

                norms = get_norm_level_sig(sig, item.shape[1], level_sig)
                psi = psi_tilde(np.sum(norms),M,a)

                lambda_ = brentq(poly,0, 10000, args=(psi, norms,level_sig))

                items_before.append(np.sum(norms))

                X[bag][i]=X[bag][i]*lambda_

                if return_norms:
                    sig_after = iisignature.sig(X[bag][i], level_sig)

                    norms_after = get_norm_level_sig(sig_after, item.shape[1], level_sig)
                    items_after.append(np.sum(norms_after))

        if return_norms:
            bags_before.append(np.mean(items_before))
            bags_after.append(np.mean(items_after))

        sig = iisignature.sig(X[bag],level_sig)

        expected_sig.append(np.mean(np.array(sig),axis=0))
    if return_norms:
        return np.array(expected_sig), bags_before, bags_after
    else:
        return np.array(expected_sig)




def scaled_pathwise_expected_iisignature(X,level_sig, M=1,a=1, ilya_rescale=False, return_norms=False):
    expected_sig = []
    maxi = M*(1.+1./a)


    for bag in range(len(X)):

        pathwise_items = [[] for i in range(len(X[bag]))]

        N_items = len(X[bag])


        #go through each element of the bag to find its scaling:
        for i,item in enumerate(X[bag]):

            if ilya_rescale:
                sig = iisignature.sig(item, level_sig, 2)

                norms = get_norm_level_sig(sig, item.shape[1], level_sig)
                psi = psi_tilde(np.sum(norms), M, a)

                lambda_ = brentq(poly, 0, 10000, args=(psi, norms, level_sig))

                pathwise_items[i] = iisignature.sig(item*lambda_, level_sig, 2)
            else:
                pathwise_items[i] = iisignature.sig(item,level_sig,2)


        pathwise_sig = np.array(pathwise_items)
        expected_sig.append(np.mean(pathwise_sig,axis=0))



    return np.array(expected_sig)


def csaba_scaled_expected_sig(X,level_sig):
    expected_sig = []


    for bag in range(len(X)):
        sig_items = []
        #go through each element of the bag to find its scaling:
        for i,item in enumerate(X[bag]):

            sig = signatory.signature(torch.tensor(item[None,:]), level_sig)

            sig_items.append(csaba_rescale(sig[0].numpy(), item.shape[1], level_sig))

        expected_sig.append(np.mean(np.array(sig_items),axis=0))

    return np.array(expected_sig)



def get_scaled_sig(bag, level_sig, M,a):

    new_bag = []
    maxi = M * (1. + 1. / a)

    for item in bag:
        #print(item[None,:].shape)
        sig = signatory.signature(torch.tensor(item[None, :]), level_sig)
        norms = get_norm_level_sig(sig[0].numpy(), item.shape[1], level_sig)
        psi = psi_tilde(np.sum(norms), M, a)

        lambda_ = brentq(poly, 0, 10000, args=(psi, norms, level_sig))
        print(norms.sum)
        #print(np.sum(norms))
        new_bag.append(item * lambda_)

    return new_bag

def get_Csaba_scaled_expected_sig(bag, level_sig):
    sig_items = []
    for item in bag:
        sig = signatory.signature(torch.tensor(item[None, :]), level_sig)
        sig_items.append(csaba_rescale(sig[0].numpy(), item.shape[1], level_sig))
    return np.mean(np.array(sig_items),axis=0)