from __future__ import print_function

import numpy as np
import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable
from dataloader import NdviDataset
from model import MIL_LSTM, MIL_RNN
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
root_dir = '../'

train_loader = data_utils.DataLoader(NdviDataset(data_file=root_dir+'input_list_RBF.obj',label_file=root_dir+'output_RBF.obj',train=True,nb_train=200),
                                     batch_size=1,
                                     shuffle=True,
                                     **loader_kwargs)

test_loader = data_utils.DataLoader(NdviDataset(data_file=root_dir+'input_list_RBF.obj',label_file=root_dir+'output_RBF.obj',train=False,nb_train=200),
                                    batch_size=1,
                                    shuffle=False,
                                    **loader_kwargs)

print('Init Model')
if args.model=='LSTM':
    #model = MIL_LSTM(input_size=2, output_size=1, hidden_dim=10,n_layers=1)
    model = MIL_LSTM(input_dim=2, output_dim=1, hidden_dim=10,hidden_dim2=10, layer_dim=1)
elif args.model=='RNN':
    model = MIL_RNN(input_dim=2, hidden_dim=10, output_dim=1)

if args.cuda:
    model.cuda()

#optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)

optimizer = torch.optim.Adadelta(model.parameters(),weight_decay=args.reg)


def Loss(y_pred,y_true):

    return (y_true - y_pred.mean())**2



def train(epoch,ax=None):

    model.train()
    train_loss = 0.
    nb_bags = 0
    labels_plot = []
    preds_plot = []



    for batch_idx, (data, bag_label) in enumerate(train_loader):
        data = data[0]
        if args.cuda:
            data, bag_label= data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)

        # reset gradients
        optimizer.zero_grad()

        # calculate loss
        y_pred = model(data)
        loss = Loss(y_pred,bag_label)

        preds_plot.append(y_pred.mean().item())
        labels_plot.append(bag_label.detach().cpu())

        train_loss += loss.data

        # backward pass
        loss.backward()
        optimizer.step()

        nb_bags+=1

    # calculate loss for epoch
    train_loss /= nb_bags

    if epoch%100==0:
        print('Epoch: {}, Loss: {:.4f}'.format(epoch, train_loss.item()))

    if not ax is None:
        plot_fit(ax, np.array(labels_plot),np.array(preds_plot))
    return train_loss

def test(ax=None):
    model.eval()
    test_loss = 0.
    c = 0
    labels = []
    predictions = []

    for batch_idx, (data,bag_label) in enumerate(test_loader):
        data = data[0]

        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)

        # calculate loss and metrics
        y_pred = model(data)
        predictions.append(y_pred.mean().item())
        loss = Loss(y_pred,bag_label)
        labels.append(bag_label.detach().cpu())
        test_loss += loss.data

        c+=1

    # calculate loss and error for epoch
    test_loss /= c

    print('Loss: {:.4f}'.format(test_loss.item()))
    if not ax is None:
        plot_fit(ax, np.array(labels), np.array(predictions))
    return test_loss


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
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = axs.ravel()
    loss_train = []
    loss_test = []
    for epoch in range(1, args.epochs):
        loss = train(epoch)
        loss_train.append(loss.data)
        if epoch%50==0:
            loss = test()
            loss_test.append(loss.data)

    train(args.epochs, axs[0])
    axs[2].plot(loss_train)
    axs[3].plot(loss_test)
    print('Start Testing')
    test(axs[1])
    plt.savefig('nn.png')