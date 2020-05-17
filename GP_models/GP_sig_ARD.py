import torch
import numpy as np
from tqdm import tqdm_notebook as tqdm
import math


def train(model, training_iter, RBF=False, plot=False,ax=None):
    optimizer = torch.optim.Adam(model.params, lr=0.01)
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
                ax.plot(losses)
                ax.set_xlabel('epoch')
                ax.set_ylabel('negative marginal log likelihood')


            break
        optimizer.step()
    if plot and not already_plot:
        ax.plot([e[0].cpu().detach().numpy() for e in losses])
        #plt.show()


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
            #self.lengthscale = torch.nn.Parameter(torch.randn(self.level_sig, dtype=self.dtype, device=self.device))
            self.lengthscale = torch.nn.Parameter(torch.ones(self.level_sig, dtype=self.dtype, device=self.device))
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


    def get_lengthscales(self):

        tf_lengthscales = self.transform_softplus(self.lengthscale)

        return torch.cat([e.repeat(self.d**(i+1)) for i,e in enumerate(tf_lengthscales)])

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

        tf_lengthscales = self.get_lengthscales()

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

        if RBF:
            K0 = self.transform_softplus(self.variance) *self.K + self.transform_softplus(self.noise_obs, 1e-4) * torch.eye(self.K.shape[0], dtype=self.dtype,
                                                                                    device=self.device)
        else:
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
        if RBF:
            K0 = self.transform_softplus(self.variance)*self.K + self.transform_softplus(self.noise_obs, 1e-4) * torch.eye(self.K.shape[0], dtype=self.dtype,
                                                                                    device=self.device)

        else:
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