import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sys
sys.path.append('../')
from utils import *
import iisignature
from fbm import FBM
import gpytorch
import torch

class Area():

    def __init__(self):
        self.path = None
        self.labels = None

    def set_parameters(self):
        pass

    def generate_data_Matern(self,t_span,nu=0.5,variance=1.):

        self.t_span = t_span
        self.nu = nu

        kern = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu))
        kern._set_outputscale(variance)
        mean = gpytorch.means.ZeroMean()
        model = ExactGPModel(torch.tensor(t_span), None, gpytorch.likelihoods.GaussianLikelihood(), mean, kern)
        dist = model(torch.tensor(t_span))
        K = dist._covar.evaluate().detach().numpy()
        mean = dist.loc.detach().numpy()

        sample = np.random.multivariate_normal(mean, K, 2).T

        self.path =  sample[:,0][:,None] #brownian(int(len(t_span) - 1),width=1, time=t_span[-1])

        self.new_dim  = sample[:,1][:,None]


    def generate_data_fBm(self,t_span,hurst=0.5):

        self.t_span = t_span
        self.hurst = hurst

        f = FBM(n=(len(t_span) - 1), hurst=hurst, length=t_span[-1], method='daviesharte')
        t_values = f.times()
        self.t_values = t_values

        fbm_sample = f.fbm()
        self.path =  fbm_sample[:,None] #brownian(int(len(t_span) - 1),width=1, time=t_span[-1])

        fbm_sample2 = f.fbm()
        self.new_dim = fbm_sample2[:,None]

    def get_area(self,path_aug='0',level=5,chunks=5):

        if path_aug == '0':
            self.add_uniform_time()

        elif path_aug == '1':
            self.add_time()

        elif path_aug == '2':
            self.lead_lag()
            print(self.path_aug_time.shape)

        elif path_aug == '3':
            self.add_dim_process()


        self.paths_seq = np.array_split(self.path_aug_time, chunks)

        pathwise_sig = np.concatenate([iisignature.sig(self.paths_seq,level)] ,axis=1)
        # 12-21 <-> (for a 2d path) 3-4   [1,2|11,12,21,22]
        areas = pathwise_sig[:, 3] - pathwise_sig[:, 4]

        self.labels = areas[:,None]
        self.pathwise_sig = pathwise_sig
        self.paths_seq = np.array(self.paths_seq)



    def add_dim_process(self):

        self.path_aug_time = np.concatenate((self.new_dim, self.path), axis=1)

    def add_time(self):
        self.path_aug_time = np.concatenate((self.t_span[:,None],self.path),axis=1)
        #self.paths_seq = [self.path_aug_time[:i,:] for i in range(2,self.path_aug_time.shape[0]+1)]

    def lead_lag(self):
        tf = LeadLag([0])

        self.path_aug_time = tf.fit_transform([self.path])[0]
        self.path_aug_time = self.path_aug_time[:self.path_aug_time.shape[0]-1,:]

    def add_uniform_time(self):
        tf = AddTime()

        self.path_aug_time = tf.fit_transform([self.path])[0]

        #[self.path_aug_time[:i,:] for i in range(2,self.path_aug_time.shape[0]+1)]


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, mean, kernel):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = mean

        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)