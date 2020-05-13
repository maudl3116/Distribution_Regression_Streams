import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle

class PadSequence:
    def __call__(self, batches):
        # Let's assume that each element in "batch" is a tuple (data, label).
        # Sort the batch in the descending order
        batch = []
        for b in batches:
            batch = batch+b
        sorted_batch = sorted(batch, key=lambda item: item[0].shape[0], reverse=True)
        # Get each sequence and pad it
        sequences = [torch.tensor(item[0]) for item in sorted_batch]
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        # Also need to store the length of each sequence
        # This is later needed in order to unpad the sequences
        lengths = torch.LongTensor([len(item[0]) for item in sequences])
        labels = torch.tensor(list(map(lambda item: item[1], sorted_batch)),dtype=torch.float32)
        bag_id = list(map(lambda item: item[2], sorted_batch))

        unique_ids = np.unique(bag_id)
        proj_vec = []
        for id in unique_ids:
            l = np.zeros(len(bag_id))
            l[bag_id == id] = 1.
            proj_vec.append(np.array(l)[:,None])
        proj_vec = torch.tensor(proj_vec,dtype=torch.float32)
        # Don't forget to grab the labels of the *sorted* batch
        return sequences_padded, lengths, proj_vec, labels

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
            self.train_bags_list, self.train_labels_list = [[e for e in self.ndvi_data[i]] for i in range(nb_train)], [self.labels[i][0] for i in range(nb_train)]
        else:
            self.test_bags_list, self.test_labels_list = [[e for e in self.ndvi_data[i]] for i in range(nb_train,len(self.labels))], [self.labels[i][0] for i in range(nb_train,len(self.labels))]

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
        tuples = [[item,label,index] for item in bag]
        return tuples