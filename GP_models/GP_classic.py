import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from scipy.integrate import odeint
import time
import scipy
import math
from tqdm import tqdm
#from tqdm.notebook import trange, tqdm
import math
from matplotlib.pyplot import imshow, show, colorbar


def train(model, training_iter,RBF_top=False, plot=False,ax=None):

    optimizer = torch.optim.Adam(model.params, lr=0.1)
    losses = []
    already_plot = False

    for i in np.arange(training_iter):
        if RBF_top:
            loss = model.obj_RBF()
        else:
            loss = model.obj()
        optimizer.zero_grad()
        loss.backward()
        losses.append(loss)

        if i > 100 and np.abs(losses[i].cpu().detach().numpy() - losses[i - 1].cpu().detach().numpy()) < 1e-5:
            if plot:
                already_plot = True
                ax.plot(losses)
                ax.set_xlabel('epoch')
                ax.set_ylabel('negative marginal log likelihood')

            break
        optimizer.step()
    if plot and not already_plot:
        ax.plot([e[0].cpu().detach().numpy() for e in losses],color='blue')
        ax.set_xlabel('epoch')
        ax.set_ylabel('negative marginal log likelihood')




class GP():

    def __init__(self, X, Y, l_init, var_init, noise_init,l_init_top=None, param_list=['lengthscale', 'variance', 'noise'],ARD=False,dtype=torch.float64,
                 device=torch.device("cpu")):

        self.device = device
        self.dtype = dtype

        self.Y = Y

        self.training_data = X

        if device==torch.device('cuda'):
            self.training_data = self.training_data.cuda()
            self.Y = self.Y.cuda()

        self.n, self.n_items = X.shape[0], X.shape[2]

        self.jitter = 1e-6 * torch.ones(1, dtype=self.dtype, device=self.device)

        self.params = []

        self.mean_constant = torch.nn.Parameter(0. * torch.ones(1, dtype=self.dtype, device=self.device))
        self.params.append(self.mean_constant)

        if 'lengthscale' in param_list:
            if ARD:
                self.lengthscale = torch.nn.Parameter(l_init * torch.ones(X.shape[1], dtype=self.dtype, device=self.device))
            else:
                self.lengthscale = torch.nn.Parameter(l_init * torch.ones(1, dtype=self.dtype, device=self.device))
                self.lengthscale_top = torch.nn.Parameter(l_init_top * torch.ones(1, dtype=self.dtype, device=self.device))
            self.params.append(self.lengthscale)
            self.params.append(self.lengthscale_top)
        else:
            self.lengthscale = l_init * torch.ones(1, dtype=self.dtype, device=self.device)

        if 'variance' in param_list:
            self.variance = torch.nn.Parameter(var_init * torch.ones(1, dtype=self.dtype, device=self.device))
            self.variance_top = torch.nn.Parameter(var_init * torch.ones(1, dtype=self.dtype, device=self.device))
            self.params.append(self.variance)
            self.params.append(self.variance_top)
        else:
            self.variance = var_init * torch.ones(1, dtype=self.dtype, device=self.device)

        if 'noise' in param_list:
            self.noise_obs = torch.nn.Parameter(noise_init * torch.ones(1, dtype=self.dtype, device=self.device))
            self.params.append(self.noise_obs)
        else:
            self.noise_obs = noise_init * torch.ones(1, dtype=self.dtype, device=self.device)


    def transform_softplus(self, input, min=0.):
        return torch.log(1. + torch.exp(input)) + min

    def K_eval(self, x, y):


        tf_lengthscales = self.transform_softplus(self.lengthscale)

        # x is of shape [N_bags x T x N_items]

        x = x.div(tf_lengthscales[None,:,None])
        y = y.div(tf_lengthscales[None,:,None])


        yy = y.repeat(1, 1, self.n_items)
        xx = x.reshape(-1, 1).repeat(1, self.n_items).reshape(x.shape[0], x.shape[1], self.n_items**2)

        Xs = torch.sum(xx ** 2, axis=-2)
        X2s = torch.sum(yy ** 2, axis=-2)
        dist = -2 * torch.tensordot(x, y, [[-2], [-2]]).transpose(1, 2).reshape(x.shape[0], y.shape[0], self.n_items**2)

        dist += Xs[:, None, :] + X2s[None, :, :]

        return torch.mean( torch.exp(-dist / 2.), axis=2)


    def K_eval_full(self, X, Y=None):
        if (Y is None):
            return self.K_eval(X,X)
        else:
            return self.K_eval(X,Y)

    def get_K_top_RBF(self, K,shape=None):


        diag = torch.diag(K)

        dist = -2. * K + diag.repeat(K.shape[0], 1) + diag[:, None].repeat(1, K.shape[1])

        K_RBF = torch.exp(-0.5 * dist / (self.transform_softplus(self.lengthscale_top) ** 2))

        if shape == None:
            return K_RBF
        else:
            # train vs test
            K_sliced = torch.zeros((shape,K_RBF.shape[0]-shape),device=self.device,dtype=self.dtype)
            for i in range(shape):
                for j in range(K_RBF.shape[0]-shape):

                    K_sliced[i][j] = K_RBF[i, j+shape]
            return K_sliced

    def obj_RBF(self):

        self.K = self.K_eval_full(self.training_data)

        K = self.transform_softplus(self.variance_top)*self.get_K_top_RBF(self.K)

        K0 = K + self.transform_softplus(self.noise_obs, 1e-4) * torch.eye(K.shape[0], dtype=self.dtype,
                                                                           device=self.device)

        L = torch.cholesky(K0)

        logdetK0 = 2. * torch.sum(torch.log(torch.diag(L)))
        Lk = torch.triangular_solve(self.Y - self.mean_constant * torch.ones_like(self.Y,dtype=self.dtype,device=self.device), L, upper=False)[0]
        ytKy = torch.mm(Lk.t(), Lk)

        ml = -0.5 * logdetK0 - 0.5 * ytKy - 0.5 * math.log(2. * math.pi) * self.n

        return -ml.div_(self.n)

    def obj(self):

        K = self.transform_softplus(self.variance) * self.K_eval_full(self.training_data)

        K0 = K + self.transform_softplus(self.noise_obs, 1e-4) * torch.eye(K.shape[0], dtype=self.dtype,
                                                                           device=self.device)

        L = torch.cholesky(K0)

        logdetK0 = 2. * torch.sum(torch.log(torch.diag(L)))
        Lk = torch.triangular_solve(self.Y - self.mean_constant * torch.ones_like(self.Y,dtype=self.dtype,device=self.device), L, upper=False)[0]
        ytKy = torch.mm(Lk.t(), Lk)

        ml = -0.5 * logdetK0 - 0.5 * ytKy - 0.5 * math.log(2. * math.pi) * self.n

        return -ml.div_(self.n)


    def predict(self, X_train, X_test=None,RBF_top=False):

        if RBF_top:
            K = self.K_eval_full(X_train)
            K = self.transform_softplus(self.variance) * self.get_K_top_RBF(K)
        else:
            K = self.transform_softplus(self.variance) * self.K_eval_full(X_train)

        K0 = K + self.transform_softplus(self.noise_obs, 1e-4) * torch.eye(K.shape[0], dtype=self.dtype,device=self.device)

        L = torch.cholesky(K0)

        # Compute the mean at our test points
        if RBF_top:
            if X_test is None:
                K_s = K
                K_ss = K
            else:
                X_full = torch.cat((X_train,X_test))
                K_full = self.K_eval_full(X_full,X_full)
                K_ss = self.K_eval_full(X_test,X_test)
                K_ss = self.transform_softplus(self.variance) * self.get_K_top_RBF(K_ss)
                K_s = self.transform_softplus(self.variance) * self.get_K_top_RBF(K_full,X_train.shape[0])

        else:
            if X_test is None:
                X_test = X_train
            K_s = self.transform_softplus(self.variance) * self.K_eval_full(X_train,X_test)
            K_ss = self.transform_softplus(self.variance) * self.K_eval_full(X_test,X_test)

        Lk = torch.triangular_solve(K_s, L, upper=False)[0]
        Ly = torch.triangular_solve(self.Y - self.mean_constant * torch.ones_like(self.Y, dtype=self.dtype,device=self.device), L, upper=False)[0]

        mu_test = self.mean_constant * torch.ones((K_ss.shape[0], 1),dtype=self.dtype,device=self.device) + torch.mm(Lk.t(), Ly)

        # Comoute the standard devaitoin so we can plot it

        s2 = torch.diag(K_ss) - torch.sum(Lk ** 2, axis=0)
        stdv_test = 2 * torch.sqrt(s2 + self.transform_softplus(self.noise_obs, 1e-4))

        return mu_test.cpu().detach().numpy(), stdv_test.cpu().detach().numpy()
