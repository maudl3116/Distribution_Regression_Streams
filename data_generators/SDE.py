import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sys
sys.path.append('../')
import utils
# import iisignature
# from fbm import FBM
# import gpytorch
import torch

class sde():

    def __init__(self,N_bags=100, N_items=15, t_span=np.linspace(0,1,500),spec_param={'theta_1':[1,3], 'theta_2':[1,3],'Y0':[1,1.5 ]}):
        self.path = None
        self.labels = None

        self.N_bags = N_bags
        self.N_items = N_items
        self.t_span = t_span
        self.spec_param = spec_param


    def set_parameters(self):
        pass

    def generate_data(self):

        ''' dY_t = f(Y_t;\theta_1)dW^{1}_t + g(Y_t;\theta_2)dW^2_t  '''

        # GENERATE RANDOM NUMBERS
        theta_1 = (self.spec_param['theta_1'][1] - self.spec_param['theta_1'][0]) * np.random.rand(self.N_bags) + self.spec_param['theta_1'][0]*np.ones(self.N_bags)
        #theta_2 = (self.spec_param['theta_2'][1]-self.spec_param['theta_2'][0])*np.random.rand(self.N_bags) + self.spec_param['theta_2'][0]*np.ones(self.N_bags)
        Y0 = (self.spec_param['Y0'][1] - self.spec_param['Y0'][0]) * np.random.rand(self.N_bags) + \
                  self.spec_param['Y0'][0] * np.ones(self.N_bags)
        self.theta_1 = theta_1
        #self.theta_2 = theta_2
        self.Y0 = Y0


        # EULER SCHEME
        # 1. Generate brownian motion samples
        BM_samples = utils.brownian(steps=len(self.t_span)-1, width=2*self.N_bags*self.N_items, time=self.t_span[-1])
        #self.samples = BM_samples.reshape(len(self.t_span),self.N_bags,self.N_items,2)
        self.samples = BM_samples.reshape(len(self.t_span),self.N_bags,self.N_items,2)
        self.increments = np.diff(self.samples,axis=0)

        # 2. EULER
        paths = np.zeros((len(self.t_span),self.N_bags,self.N_items))
        # set the initial point
        paths[0,:,:] = np.repeat(self.Y0[:,None],self.N_items, axis=1)

        for t in range(1,len(self.t_span)):
            paths[t,:,:] = paths[t-1,:,:] + self.increments[t-1,:,:,0]  + (1.+self.theta_1[:,None]**(0.5))*self.increments[t-1,:,:,1]

        self.paths_tmp = paths # time, N_bags, N_items

        self.paths =  torch.tensor(paths).transpose(1,0).transpose(2,1)[:,:,:,None].numpy()   # N_bags, N_items, time, dim


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

    def get_param(self):
        self.labels = self.theta_1[:,None]#/self.theta_2[:,None]
        #self.labels = [:,None]


    def plot(self,N_bags=4,N_items=5):
        #sns.set_style("whitegrid")
        #sns.set(font_scale=1.5)
        sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
        fig, ax = plt.subplots(1, N_bags, figsize=(N_bags*5, 5))
        colors = sns.color_palette("hls", N_bags+3)[3:]
        for i in range(N_bags):
            for j in range(N_items):
                ax[i].plot(self.t_span,self.paths[i, j, :, 0],color=colors[i],label=r'$y(\omega_%i$'%j)
            ax[i].set_title(r'$\theta=$%.2f'%self.labels[i])
            ax[i].set_xlabel('$t$')
            ax[i].set_ylabel(r'$y_t$')
        #plt.legend()
        plt.tight_layout()

        plt.show()

