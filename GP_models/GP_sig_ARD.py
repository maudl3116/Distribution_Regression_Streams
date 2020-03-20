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
from tqdm import tqdm_notebook as tqdm
# from tqdm.notebook import trange, tqdm
import math


def get_K_RBF_Sig_dummy(K, sigma_2, l):
    K = torch.tensor(K, dtype=torch.float64)

    diag = torch.diag(K)
    dist = -2. * K + diag.repeat(K.shape[0], 1) + diag[:, None].repeat(1, K.shape[1])

    K_RBF = sigma_2 * torch.exp(-0.5 * dist / (l ** 2))
    return K_RBF.detach().numpy()


def get_K_RBF_Sig(K, sigma_2, l):
    K_RBF = np.zeros((K.shape[0], K.shape[1]))

    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            dist = (K[i][i] - 2 * K[i][j] + K[j][j])
            K_RBF[i][j] = sigma_2 * np.exp(-0.5 * dist / (l ** 2))
            K_RBF[j][i] = K_RBF[i][j]

    return K_RBF


def get_K(K_input, indices1, indices2=None):
    if indices2 is None:
        indices2 = indices1

    K = np.zeros((len(indices1), len(indices2)))

    for i in range(len(indices1)):
        for j in range(len(indices2)):
            K[i][j] = K_input[indices1[i], indices2[j]]
    return K


def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:" + fmt + "}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:" + str(col_maxes[i]) + fmt + "}").format(y), end="  ")
        print("")


def train(model, training_iter, RBF=False, plot=False):
    optimizer = torch.optim.Adam(model.params, lr=0.1)
    losses = []
    already_plot = False
    for i in tqdm(np.arange(training_iter)):
        # Zero gradients from previous iteration
        # Output from model
        # Calc loss and backprop gradients
        if RBF:
            loss = model.obj_RBF()
        else:
            loss = model.obj()
        optimizer.zero_grad()
        loss.backward()
        losses.append(loss)
        if i > 300 and np.abs(losses[i].cpu().detach().numpy() - losses[i - 1].cpu().detach().numpy()) < 1e-5:

            # print(np.abs(losses[i].detach().numpy()-losses[i-1].detach().numpy()))
            if plot:
                already_plot = True
                plt.plot(losses)
                plt.xlabel('epoch')
                plt.ylabel('negative marginal log likelihood')
                plt.show()

            break
        optimizer.step()
    if plot and not already_plot:
        plt.plot([e[0].cpu().detach().numpy() for e in losses])
        plt.show()


def plot_marginal_log_lik(model):
    lengthscales = torch.linspace(-25, 25, 50)
    obs_noises = torch.linspace(-15, 10, 50)

    # print(model.transform_softplus(lengthscales)**2)
    # print(model.transform_softplus(obs_noises,1e-4))

    loss = np.zeros((len(obs_noises), len(lengthscales)))

    for i, l in enumerate(lengthscales):

        model.lengthscale.data = l * torch.ones(1, dtype=model.dtype, device=model.device)
        for j, obs_noise in enumerate(obs_noises):
            model.noise_obs.data = obs_noise * torch.ones(1, dtype=model.dtype, device=model.device)
            loss[j][i] = model.obj_RBF()

    X, Y = np.meshgrid(model.transform_softplus(lengthscales) ** 2, model.transform_softplus(obs_noises, 1e-4))

    cp = plt.contourf(X, Y, loss)
    plt.colorbar(cp)

    plt.title('negative marginal log likelihood')
    plt.xlabel(r'$l^2$')
    plt.ylabel(r'$\sigma_n^2$')
    plt.show()


class GP():

    def __init__(self, X, Y, d, level_sig, l_init, var_init, noise_init, param_list, dtype=torch.float64,
                 device=torch.device("cpu")):

        self.device = device
        self.dtype = dtype
        # constants (except K)
        self.Y = Y
        self.training_data = X
        self.n = len(X)

        # d is the dimension of the paths before taking their signature
        self.d = d
        self.level_sig = level_sig

        if device == torch.device('cuda'):
            self.Y = self.Y.cuda()
            self.training_data = self.training_data.cuda()
        self.jitter = 1e-6 * torch.ones(1, dtype=self.dtype, device=self.device)

        self.params = []

        self.mean_constant = torch.nn.Parameter(0. * torch.ones(1, dtype=self.dtype, device=self.device))
        self.params.append(self.mean_constant)

        if 'lengthscale' in param_list:
            self.lengthscale = torch.nn.Parameter(l_init * torch.ones(self.level_sig, dtype=self.dtype, device=self.device))
            self.params.append(self.lengthscale)
        else:
            self.lengthscale = l_init * torch.ones(1, dtype=self.dtype, device=self.device)

        if 'variance' in param_list:
            self.variance = torch.nn.Parameter(var_init * torch.ones(1, dtype=self.dtype, device=self.device))
            self.params.append(self.variance)
        else:
            self.variance = var_init * torch.ones(1, dtype=self.dtype, device=self.device)

        if 'noise' in param_list:
            self.noise_obs = torch.nn.Parameter(noise_init * torch.ones(1, dtype=self.dtype, device=self.device))
            self.params.append(self.noise_obs)
        else:
            self.noise_obs = noise_init * torch.ones(1, dtype=self.dtype, device=self.device)

        self.post_logger = list()
        self.MLpost_logger = list()
        self.lengthscale_logger = list()
        self.variance_logger = list()
        self.noiseobs_logger = list()
        self.B_shape_logger = list()
        self.eig_logger = list()

        # initialize the loggers
        self.lengthscale_logger.append(self.lengthscale.data)
        self.variance_logger.append(self.variance.data)
        self.noiseobs_logger.append(self.noise_obs.data)

    def transform_softplus(self, input, min=0.):
        return torch.log(1. + torch.exp(input)) + min

    def get_K_RBF_Sig(self, K):

        diag = torch.diag(K)
        dist = -2. * K + diag.repeat(K.shape[0], 1) + diag[:, None].repeat(1, K.shape[1])

        K_RBF = torch.exp(-0.5 * dist / (self.transform_softplus(self.lengthscale) ** 2))

        return K_RBF

    def get_K_RBF_Sig_dummy(self, indices1, indices2=None):

        if indices2 is None:
            indices2 = indices1

        K = torch.zeros((len(indices1), len(indices2)), dtype=self.dtype, device=self.device)

        for i in range(len(indices1)):
            for j in range(len(indices2)):
                K[i][j] = self.K_full[indices1[i], indices2[j]]
        return K


    def get_lengthscales(self):

        tf_lengthscales = self.transform_softplus(self.lengthscale)
        l = torch.cat([e.repeat(self.d**(i)) for i,e in enumerate(tf_lengthscales)])
       
        return torch.cat([e.repeat(self.d**(i)) for i,e in enumerate(tf_lengthscales)])

    def K_RBF_eval(self,x1,x2=None):

        if x2 is None:
            x2=x1

        tf_lengthscales = self.get_lengthscales()

        # x is of shape [N_bags x T x N_items]

        x1 = x1.div(tf_lengthscales)
        x2 = x2.div(tf_lengthscales)

        Xs = torch.sum(x1 ** 2, axis=-1)

        X2s = torch.sum(x2 ** 2, axis=-1)
        dist = -2 * torch.tensordot(x1, x2, [[-1], [-1]])

        dist += Xs[:, None] + X2s[None, :]

        return torch.exp(-dist / 2.)



    def K_eval(self,x1,x2=None):
        if x2 is None:
            x2=x1

        tf_lengthscales = self.transform_softplus(self.lengthscale)

        # x is of shape [N_bags x T x N_items]
        x1 = x1.div(tf_lengthscales)
        x2 = x2.div(tf_lengthscales)

        dot = torch.tensordot(x1, x2, [[-1], [-1]])

        return dot


    def obj_RBF(self):

        self.K = self.K_RBF_eval(self.training_data)

        K = self.transform_softplus(self.variance) * self.K

        K0 = K + self.transform_softplus(self.noise_obs, 1e-4) * torch.eye(K.shape[0], dtype=self.dtype,
                                                                           device=self.device)

        L = torch.cholesky(K0)
        logdetK0 = 2. * torch.sum(torch.log(torch.diag(L)))
        Lk = torch.triangular_solve(self.Y - self.mean_constant * torch.ones_like(self.Y), L, upper=False)[0]
        ytKy = torch.mm(Lk.t(), Lk)

        ml = -0.5 * logdetK0 - 0.5 * ytKy - 0.5 * math.log(2. * math.pi) * self.n

        return -ml.div_(self.n)

    def obj(self):
        self.K = self.K_eval(self.training_data)
        K = self.K

        K0 = K + self.transform_softplus(self.noise_obs, 1e-4) * torch.eye(K.shape[0], dtype=self.dtype,
                                                                           device=self.device)

        L = torch.cholesky(K0)

        logdetK0 = 2. * torch.sum(torch.log(torch.diag(L)))
        Lk = torch.triangular_solve(self.Y - self.mean_constant * torch.ones_like(self.Y), L, upper=False)[0]
        ytKy = torch.mm(Lk.t(), Lk)

        ml = -0.5 * logdetK0 - 0.5 * ytKy - 0.5 * math.log(2. * math.pi) * self.n

        return -ml.div_(self.n)



    def predict(self, x_new, RBF=False):


        K0 = self.K + self.transform_softplus(self.noise_obs, 1e-4) * torch.eye(self.K.shape[0], dtype=self.dtype,
                                                                           device=self.device)

        L = torch.cholesky(K0)

        # Compute the mean at our test points
        if RBF:
            K_ss = self.transform_softplus(self.variance) * self.K_RBF_eval(x_new)
            K_s = self.transform_softplus(self.variance) * self.K_RBF_eval(self.training_data, x_new)

        else:
            K_ss = self.K_eval(x_new)
            K_s = self.K_eval(self.training_data, x_new)
        Lk = torch.triangular_solve(K_s, L, upper=False)[0]
        Ly = torch.triangular_solve(self.Y - self.mean_constant * torch.ones_like(self.Y, dtype=self.dtype), L, upper=False)[
            0]

        mu_test = self.mean_constant * torch.ones((K_ss.shape[0], 1), dtype=self.dtype, device=self.device) + torch.mm(
            Lk.t(), Ly)

        # Comoute the standard devaitoin so we can plot it

        s2 = torch.diag(K_ss) - torch.sum(Lk ** 2, axis=0)
        stdv_test = 2 * torch.sqrt(s2 + self.transform_softplus(self.noise_obs, 1e-4))

        return mu_test.cpu().detach().numpy(), stdv_test.cpu().detach().numpy()

    def predict_on_training(self, RBF=False):

        K0 = self.K + self.transform_softplus(self.noise_obs, 1e-4) * torch.eye(self.K.shape[0], dtype=self.dtype,
                                                                                device=self.device)

        L = torch.cholesky(K0)

        # Compute the mean at our test points
        if RBF:
            K_ss = self.transform_softplus(self.variance) * self.K
            K_s = self.transform_softplus(self.variance) * self.K

        else:
            K_s = self.K
            K_ss = self.K
        Lk = torch.triangular_solve(K_s, L, upper=False)[0]
        Ly = \
        torch.triangular_solve(self.Y - self.mean_constant * torch.ones_like(self.Y, dtype=self.dtype), L, upper=False)[
            0]

        mu_test = self.mean_constant * torch.ones((K_ss.shape[0], 1), dtype=self.dtype, device=self.device) + torch.mm(
            Lk.t(), Ly)

        # Comoute the standard devaitoin so we can plot it

        s2 = torch.diag(K_ss) - torch.sum(Lk ** 2, axis=0)
        stdv_test = 2 * torch.sqrt(s2 + self.transform_softplus(self.noise_obs, 1e-4))

        return mu_test.cpu().detach().numpy(), stdv_test.cpu().detach().numpy()