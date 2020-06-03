from sklearn.cluster import KMeans
import numpy as np
import pickle
from esig import tosig as esig


def get_sig_keys(dico,sig_level):
    
    ndims = len(dico)
    
    keys = esig.sigkeys(ndims,sig_level).split("(")
    sig_keys = []

    for i in range(2,len(keys)-1):
        sig_keys.append(np.array(keys[i][:-2].split(',')))
    sig_keys.append(np.array(keys[len(keys)-1][:-1].split(',')))
    
    features_names = {}
    for i in range(len(sig_keys)):
        separator = '-'
        name = separator.join([dico[int(e)] for e in sig_keys[i]])
        features_names[1+i]= name

    return features_names

def subsample(input_,p):
    '''
    
    This function applies random subsampling of bags of items of D-dimensional time-series. All dimensions of the time-series 
    are observed/dropped at the same time. 

       Input:
              input_ (list): list of lists of (length,dim) arrays

                        - len(X) = n_bags

                        - for any i, input_[i] is a list of n_items arrays of shape (length, dim)

                        - for any j, input_[i][j] is an array of shape (length, dim)
               p   (float): the percentage of observations to drop. Should be between 0 and 1 

       Output: input_ (list): list of lists of (length*(1-p),dim) arrays

    '''
    
    assert p>=0 and p<1
    
    new_input_ = []
    
    for i in range(len(input_)): # loop through bags
        new_bag = []
        for j in range(len(input_[i])): # loop through items
            L = len(input_[i][j])
            # Number of observations to select
            N = (1.-p)*L
            time_stamps_kept = np.sort(np.random.choice(np.arange(L),round(N),replace=False))
            new_bag.append(input_[i][j][time_stamps_kept,:])
        new_input_.append(new_bag)
            
    return new_input_


def spatial_CV(nb_folds, targets_dico):
    '''
    The crops data consists in regional annual yield between 2006 and 2017 for the 22 (NUTS 2) regions in France. As nearby regions tend to
    be similar, we propose to create test/train splits which attempt to break spatial correlation. We cluster the 22 regions into nb_folds
    groups. Subsequently, we create train/test splits corresponding to a leave-one-out scheme. Each cluster is removed from the training set
    in turn.


       Input:
              nb_folds (int): how many train/test splits the user wants

              targets_dico (dictionary): this dictionary contains the targets (yield in tonnes/ha). Each key is a tuple (year,nut region)

       Output: a list of tuples (train_indices,test_indices)

    '''

    # Create spatial train/test splits. For this we need to get geographical information about the nut regions.
    # These are stored in dico_geom.obj
    dico_geom = pickle.load(open('../data/dico_geom.obj', 'rb'))

    clusters = KMeans(n_clusters=nb_folds, n_jobs=-1, random_state=2)
    clusters.fit(list(dico_geom.values()))

    v_lookup = {}
    for i in range(len(clusters.labels_)):
        v_lookup[list(dico_geom.keys())[i]] = clusters.labels_[i]

    splits = []
    for i in range(nb_folds):

        # train indices : all but cluster i
        train_indices = []
        # test indices : cluster i
        test_indices = []

        for k in range(len(targets_dico)):
            # get region for the point
            region = list(targets_dico.keys())[k][1]
            if v_lookup[region] != i:  # gives in which cluster the region is
                train_indices.append(k)
            else:
                test_indices.append(k)

        splits.append((train_indices, test_indices))

    return splits


def temporal_CV(targets_dico):
    '''
    The crops data consists in regional annual yield between 2006 and 2017 for the 22 (NUTS 2) regions in France. We create train/test splits
    corresponding to an expanding window scheme. Each data within a year is considered in turn as a test set, with associated training set
    being the data for the previous year. The number of folds is determined automatically, and corresponds to the number of years
    minus one.


       Input:
              targets_dico (dictionary): this dictionary contains the targets (yield in tonnes/ha). Each key is a tuple (year,nut region)

       Output: a list of tuples (train_indices,test_indices)

    '''
    years = [int(key[0]) for key in targets_dico.keys()]
    years = np.sort(np.unique(np.array(years)))

    splits = []
    for i in range(len(years) - 1):

        # train indices : all previous years
        train_indices = []
        # test indices : year to predict
        test_indices = []

        for k in range(len(targets_dico)):
            # get region for the point
            year = int(list(targets_dico.keys())[k][0])
            if year < years[i + 1]:  # gives in which cluster the region is
                train_indices.append(k)
            else:
                test_indices.append(k)

        splits.append((train_indices, test_indices))

    return splits

def filter_ndvi_data(input_list,labels):
    '''
        This function applies to a specific dataset, which consists in bags of items (lists of lists) of multi-spectral time-series
        (2D arrays). The dimensions of the array correspond to the near infra-red and the red spectral bands. We remove multi-spectral
        time-series for which the average of the first coordinate does not exceed 0.2. Subsequently, we remove bags which have less than
        10 items. The labels are filtered accordingly.

       Input:
              input_ (list): list of lists of (length,dim) arrays

                        - len(X) = n_bags

                        - for any i, input_[i] is a list of n_items arrays of shape (length, dim)

                        - for any j, input_[i][j] is an array of shape (length, dim)
              labels (dictionary): this dictionary contains the targets (yield in tonnes/ha). Each key is a tuple (year,nut region)

       Output: filtered inputs, filtered labels.

    '''
    new_input_list = []
    new_labels = {}
    for region in range(22*11):
        bag = []
        for i in range(input_list[region].shape[0]):
            if input_list[region][i,:,0].mean()>0.2 or input_list[region].shape[0]==1:
                bag.append(input_list[region][i,:,:][None,:,:])
        if len(bag)>10:
            new_input_list.append(np.concatenate(bag,axis=0))
            key = list(labels.keys())[region]
            new_labels[key]=labels[key]

    return new_input_list, new_labels