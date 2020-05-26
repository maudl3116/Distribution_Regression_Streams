from sklearn.cluster import KMeans
import numpy as np
import pickle

def bags_to_2D(input_):

    '''
    For the RBF-RBF model, we have defined a customed kernel in sklearn. An sklearn pipeline takes in input
    a 2D array (n_samples, n_features). Whilst we have data in the form of bags of items, where each item is a D-dimensional time series
    represented as a list of list of (LxD) matrices, where L is the length of the time series.

    This function transforms lists of lists of D-dimensional time series into a 2D array.

       Input:
              input_ (list): list of lists of (length,dim) arrays

                        - len(X) = n_bags

                        - for any i, input_[i] is a list of n_items arrays of shape (length, dim)

                        - for any j, input_[i][j] is an array of shape (length, dim)

       Output: a 2D array of shape (n_bags,n_items x length x dim)

    '''

    dim_path = input_[0][0].shape[1]

    # adjust time and concatenate the dimensions
    T = [e[0].shape[0] for e in input_]
    common_T = min(T)  # for other datasets would consider padding. Here we are just missing 10 values from time to time

    unlist_items = [np.concatenate([item[None, :common_T, :] for item in bag], axis=0) for bag in input_]

    # stack dimensions (we get a list of bags, where each bag is N_items x 3T)
    dim_stacked = [np.concatenate([bag[:, :common_T, k] for k in range(dim_path)], axis=1) for bag in unlist_items]

    # stack the items, we have item1(temp-hum-rain) - items2(temp-hum-rain) ...
    items_stacked = [bag.flatten() for bag in dim_stacked]

    # retrieve the maximum number of items
    max_ = [bag.shape for bag in items_stacked]
    max_ = np.max(max_)
    max_items = int(max_ / (dim_path * common_T))

    # pad the vectors with nan items such that we can obtain a 2d array.
    items_naned = [np.append(bag, (max_ - len(bag)) * [np.nan]) for bag in items_stacked]  # np.nan

    X = np.array(items_naned)

    return X, max_items, common_T, dim_path

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
    dico_geom = pickle.load(open('crops/dico_geom.obj', 'rb'))

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