import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm as tqdm

class DataIterator(object):
    def __init__(self, X, y, batch_size, shuffle=False):

        self.X = X # X is of shape (n_bags, n_items, length x dim)
        self.y = y # y is of shape (n_bags, )
        self.batch_size = batch_size # batch of bags
        
        self.shuffle = shuffle
        self.L = self.X.shape[0]
        self.d = self.X.shape[2]
            
    def __len__(self):
        return len(self.y)//self.batch_size

    def get_iterator(self, loss=0.0):
        if self.shuffle:
            rng_state = np.random.get_state()
            np.random.shuffle(self.X)
            np.random.set_state(rng_state)
            np.random.shuffle(self.y)
            np.random.set_state(rng_state)
        return self.next_batch()
                    
    def next_batch(self):
        start = 0
        end = self.batch_size
        while end <= self.L:
            yield self.X[start:end], self.y[start:end]
            start = end
            end += self.batch_size
            
            
class Trainer(object):

    def __init__(self, in_dims, model, num_epochs=1000):
                
        self.model = model(in_dims).cuda()
        
#         self.l = nn.L1Loss()
        self.l = nn.MSELoss()
        
        self.optim = optim.Adam(self.model.parameters(), lr=0.001)
#         self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optim, factor=0.5, patience=50, verbose=True)
        
        self.num_epochs = num_epochs
        
    def fit(self, train):
        
        train_loss = 0.0
        best_mae = 1.0e3
        best_mse = 1.0e6

        for j in range(self.num_epochs):
        
            train_iterator = train.get_iterator(train_loss)
            
            for X, y in train_iterator:
                            
                self.optim.zero_grad()
                
                y_pred = self.model(Variable(X).cuda()).reshape(-1)
                
                loss = self.l(y_pred, Variable(y).cuda())
                
                loss.backward()
                
                self.optim.step()
            
            if j%200==0:
                print('Train loss: {0:.6f}'.format(loss.data.cpu().numpy()))
                
    def evaluate(self, test):
        counts = 0
        sum_mae = 0.
        sum_mse = 0.
        test_iterator = test.get_iterator()
        for X, y in test_iterator:
            counts += 1
            y_pred = self.model(Variable(X).cuda())
            sum_mse += self.l(y_pred, Variable(y).cuda()).data.cpu().numpy()
        return np.asscalar(sum_mse/counts)
        
    def predict(self, test):
        y_preds = []
        for X, y in test.next_batch():
            y_pred = self.model(Variable(X).cuda())
            y_preds.append(y_pred.data.cpu().numpy())
        return np.concatenate(y_preds)
    
    
class DeepSet(nn.Module):

    def __init__(self, in_features, set_features=200):
        super(DeepSet, self).__init__()
        self.in_features = in_features
        self.out_features = set_features
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, set_features)
        )

        self.regressor = nn.Sequential(
            nn.Linear(set_features, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 1),
        )
        
        self.add_module('0', self.feature_extractor)
        self.add_module('1', self.regressor)
        
        
    def reset_parameters(self):
        for module in self.children():
            reset_op = getattr(module, "reset_parameters", None)
            if callable(reset_op):
                reset_op()
            
    def forward(self, inp):
        x = self.feature_extractor(inp)
        x = x.sum(dim=1)
        x = self.regressor(x)
        return x
    
    
class DeepSetRNN(nn.Module):

    def __init__(self, in_features, set_features=100):
        super(DeepSetRNN, self).__init__()
        
        self.in_features = in_features
        self.out_features = set_features
        self.feature_extractor = nn.RNN(in_features, set_features, 1, batch_first=True)

        self.regressor = nn.Sequential(
            nn.Linear(set_features, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 1),
        )
        
        self.add_module('0', self.feature_extractor)
        self.add_module('1', self.regressor)
        
    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.out_features).cuda()
        
    def reset_parameters(self):
        for module in self.children():
            reset_op = getattr(module, "reset_parameters", None)
            if callable(reset_op):
                reset_op()
            
    def forward(self, inp):
        hidden = self.init_hidden(inp.size(0))
        rnn_out, _ = self.feature_extractor(inp, hidden)
        x = rnn_out.sum(dim=1)
        x = self.regressor(x)
        return x
    
    
    
class DeepSetGRU(nn.Module):

    def __init__(self, in_features, set_features=200):
        super(DeepSetGRU, self).__init__()
        
        self.in_features = in_features
        self.out_features = set_features
        self.feature_extractor = nn.GRU(in_features, set_features, 1, batch_first=True)

        self.regressor = nn.Sequential(
            nn.Linear(set_features, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
        )
        
        self.add_module('0', self.feature_extractor)
        self.add_module('1', self.regressor)
        
    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.out_features).cuda()
        
    def reset_parameters(self):
        for module in self.children():
            reset_op = getattr(module, "reset_parameters", None)
            if callable(reset_op):
                reset_op()
            
    def forward(self, inp):
        hidden = self.init_hidden(inp.size(0))
        rnn_out, _ = self.feature_extractor(inp, hidden)
        x = rnn_out.sum(dim=1)
        x = self.regressor(x)
        return x
    
    
    
class DeepSetLSTM(nn.Module):

    def __init__(self, in_features, set_features=200):
        super(DeepSetLSTM, self).__init__()
        
        self.in_features = in_features
        self.out_features = set_features
        self.feature_extractor = nn.LSTM(in_features, set_features, 1, batch_first=True)

        self.regressor = nn.Sequential(
            nn.Linear(set_features, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 1),
        )
        
        self.add_module('0', self.feature_extractor)
        self.add_module('1', self.regressor)
        
    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.out_features).cuda(), torch.zeros(1, batch_size, self.out_features).cuda()
        
    def reset_parameters(self):
        for module in self.children():
            reset_op = getattr(module, "reset_parameters", None)
            if callable(reset_op):
                reset_op()
            
    def forward(self, inp):
        hidden = self.init_hidden(inp.size(0))
        rnn_out, (_,_) = self.feature_extractor(inp, hidden)
        x = rnn_out.sum(dim=1)
        x = self.regressor(x)
        return x