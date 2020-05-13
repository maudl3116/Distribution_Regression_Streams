from __future__ import print_function

import numpy as np
import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable
from dataloader import PadSequence, NdviDataset
from model import MIL_LSTM
import matplotlib.pyplot as plt
import torch.nn as nn

# Training settings
parser = argparse.ArgumentParser(description='MIL with recurrent neural network')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                    help='weight decay')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--model', type=str, default='LSTM', help='Choose type of recurrent neural network')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is ON!')

print('Load Train and Test Set')
loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
root_dir = '/Users/maudlemercier/Desktop/Distribution_Regression_Streams/'

train_loader = data_utils.DataLoader(NdviDataset(data_file=root_dir+'input_list_RBF.obj',label_file=root_dir+'output_RBF.obj',train=True,nb_train=200),
                                     batch_size=1,
                                     shuffle=True,
                                     collate_fn=PadSequence(),
                                     **loader_kwargs)

test_loader = data_utils.DataLoader(NdviDataset(data_file=root_dir+'input_list_RBF.obj',label_file=root_dir+'output_RBF.obj',train=False,nb_train=200),
                                    batch_size=1,
                                    shuffle=False,
                                    collate_fn=PadSequence(),
                                    **loader_kwargs)

print('Init Model')
if args.model=='LSTM':
    model = MIL_LSTM(input_dim=2, hidden_dim=10, output_dim=1)

if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)

def Loss(yhat,y):
    return torch.sqrt((y-torch.mean(yhat))**2)



def train(epoch,ax=None):
    model.train()
    train_loss = 0.
    c = 0
    labels = []
    predictions = []

    for batch_idx, (data,len, bag_label) in enumerate(train_loader):
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)
        labels.append(bag_label.detach().cpu().item())
        # reset gradients
        optimizer.zero_grad()
        # calculate loss and metrics
        y_pred = model((data,len, bag_label))
        predictions.append(y_pred.mean().detach().cpu().item())
        loss = Loss(y_pred,bag_label)
        train_loss += loss.data
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        c+=1

    # calculate loss and error for epoch
    train_loss /= c

    print('Epoch: {}, Loss: {:.4f}, Number batches: {:.4f}'.format(epoch, train_loss.cpu().numpy(),c))
    if not ax is None:
        plot_fit(ax, np.array(labels),np.array(predictions))


def test(ax):
    model.eval()
    test_loss = 0.
    c = 0
    labels = []
    predictions = []
    for batch_idx, (data,len, bag_label) in enumerate(test_loader):
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)
        labels.append(bag_label.detach().cpu().item())
        # reset gradients
        optimizer.zero_grad()
        # calculate loss and metrics
        y_pred = model((data,len, bag_label))
        predictions.append(y_pred.mean().detach().cpu().item())
        loss = Loss(y_pred,bag_label)
        test_loss += loss.data
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        c+=1

    # calculate loss and error for epoch
    test_loss /= c

    print('Loss: {:.4f}, Number batches: {:.4f}'.format(test_loss.cpu().numpy(),c))
    plot_fit(ax, np.array(labels), np.array(predictions))


def plot_fit(ax, y_, pred):
    left, width = .25, .5
    bottom, height = .25, .5
    right = left + width + .2
    top = bottom + height

    ax.plot(np.linspace(min(y_), max(y_), 20), np.linspace(min(y_), max(y_), 20), linestyle='dashed', color='black')

    order = np.argsort(y_)

    ax.scatter(y_[order], pred[order])

    RMSE = np.sqrt(np.mean((y_[order] - pred[order]) ** 2))
    num = np.sum((pred[order] - y_[order]) ** 2)

    denum = np.sum((y_[order] - np.mean(y_[order])) ** 2)

    R_squared = 1. - num / denum


    MAPE = np.mean(np.abs(y_[order] - pred[order])/y_[order])

    ax.set_ylabel('predicted target', fontsize=18)
    ax.set_xlabel('true target', fontsize=18)
    plt.text(right, bottom, 'RMSE=%.3f' % RMSE,
             horizontalalignment='right',
             verticalalignment='bottom',
             transform=ax.transAxes, fontsize=18)
    plt.text(right, bottom - 0.1, 'R-squared=%.3f' % R_squared,
             horizontalalignment='right',
             verticalalignment='bottom',
             transform=ax.transAxes, fontsize=18)

    plt.text(right, bottom - 0.2, 'MAPE=%.3f' % MAPE,
             horizontalalignment='right',
             verticalalignment='bottom',
             transform=ax.transAxes, fontsize=18)

if __name__ == "__main__":
    print('Start Training')
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs = axs.ravel()
    for epoch in range(1, args.epochs):
        train(epoch)
    train(args.epochs, axs[0])

    print('Start Testing')
    test(axs[1])
    plt.show()