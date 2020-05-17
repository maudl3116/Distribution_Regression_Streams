import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle


class NdviDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_file,label_file,train=True,nb_train=200):
        """
        Args:
                TBA
        """
        self.train = train
        self.ndvi_data = pickle.load(open(data_file,'rb'))
        self.labels = pickle.load(open(label_file, 'rb'))

        if self.train:
            self.train_bags_list, self.train_labels_list = [self.ndvi_data[i] for i in range(nb_train)], [self.labels[i][0] for i in range(nb_train)]
        else:
            self.test_bags_list, self.test_labels_list = [self.ndvi_data[i] for i in range(nb_train,len(self.labels))], [self.labels[i][0] for i in range(nb_train,len(self.labels))]

    def __len__(self):
        if self.train:
            return len(self.train_bags_list)
        else:
            return len(self.test_bags_list)

    def __getitem__(self, index):
        if self.train:
            bag = self.train_bags_list[index]
            label = self.train_labels_list[index]
        else:
            bag = self.test_bags_list[index]
            label = self.test_labels_list[index]
        #tuples = [[item,label,index] for item in bag]
        return bag,label