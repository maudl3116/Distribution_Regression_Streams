import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import iisignature
from fbm import FBM
import sys
sys.path.append('../')
import utils
import torch

class Rough_Volatility():

    # equation: x^2/a^2 + y^2/b^2 = 1
    # parametrix equation: (x,y) = (acos(t),bsin(t))  t in [0,2pi]

    def __init__(self,N_bags=50, N_items=15, t_span=np.linspace(0,1,300), hurst=0.3,spec_param={'alpha':[2,9],'m':[0.000025,0.000025],'nu':[1.,1.],'mu':[0.0003,0.0003],'Y0_1':[600,600],'Y0_2':[0.001,0.001]}):

        self.paths = None
        self.labels = None
        self.N_bags = N_bags
        self.N_items = N_items
        self.t_span = t_span
        self.L = len(t_span)
        self.spec_param = spec_param
        self.hurst = hurst


    def generate_data(self):

        alpha = (self.spec_param['alpha'][1] - self.spec_param['alpha'][0]) * np.random.rand(self.N_bags) + self.spec_param['alpha'][
            0] * np.ones(self.N_bags)

        m = (self.spec_param['m'][1] - self.spec_param['m'][0]) * np.random.rand(self.N_bags) + self.spec_param['m'][
            0] * np.ones(self.N_bags)

        nu = (self.spec_param['nu'][1] - self.spec_param['nu'][0]) * np.random.rand(self.N_bags) + self.spec_param['nu'][
            0] * np.ones(self.N_bags)

        mu = (self.spec_param['mu'][1] - self.spec_param['mu'][0]) * np.random.rand(self.N_bags) + self.spec_param['mu'][
            0] * np.ones(self.N_bags)

        Y0_1 = (self.spec_param['Y0_1'][1] - self.spec_param['Y0_1'][0]) * np.random.rand(self.N_bags) + \
                  self.spec_param['Y0_1'][0] * np.ones(self.N_bags)
        Y0_2 = (self.spec_param['Y0_2'][1] - self.spec_param['Y0_2'][0]) * np.random.rand(self.N_bags) + \
                  self.spec_param['Y0_2'][0] * np.ones(self.N_bags)

        # 1.a. GENERATE FRACTIONAL BROWNIAN MOTIONS FOR VOLATILITY
        f = FBM(len(self.t_span)-1, hurst=self.hurst,length=self.t_span[-1])
        fBM_samples = []
        for bag in range(self.N_bags):
            fBM_bag = []
            for item in range(self.N_items):
                fbm_sample = f.fbm()
                fBM_bag.append(fbm_sample)
            fBM_samples.append(fBM_bag)
        self.fBM_samples = np.array(fBM_samples).T
        self.fBM_samples = self.fBM_samples.reshape(len(self.t_span),self.N_bags,self.N_items)
        increments_fBM = np.diff(self.fBM_samples, axis=0)

        # 1.b. GENERATE BROWNIAN MOTION FOR PRICE
        BM_samples = utils.brownian(steps=len(self.t_span) - 1, width= self.N_bags * self.N_items,
                                    time=self.t_span[-1])

        self.BM_samples = BM_samples.reshape(len(self.t_span),self.N_bags,self.N_items)
        increments_BM = np.diff(self.BM_samples,axis=0)

        # 2. EULER
        paths = np.zeros((len(self.t_span), self.N_bags, self.N_items,2))
        # set the initial point
        paths[0, :, :,0] = np.repeat(Y0_1[:, None], self.N_items, axis=1)
        paths[0, :, :, 1] = np.repeat(Y0_2[:, None], self.N_items, axis=1)


        dt = self.t_span[1]-self.t_span[0]
        for t in range(1, len(self.t_span)):
            paths[t, :, :,0] = paths[t-1, :, :,0]+mu[:,None]*paths[t - 1, :, :,0]*dt + (0.01)*paths[t - 1, :, :,0]*increments_BM[t - 1, :, :]
            paths[t, :, :,1] = paths[t-1, :, :,1] -alpha[:, None]*(paths[t-1, :, :,1]-m[:,None])*dt + nu[:,None]*increments_fBM[t - 1, :, :]
#paths[t - 1, :, :,1]

        self.paths =  torch.tensor(paths).transpose(1,0).transpose(2,1).numpy()   # N_bags, N_items, time, dim

        self.alpha = alpha
        self.m = m
        self.nu = nu

    def subsample(self,N_obs):
        paths_sub = np.zeros((self.N_bags,self.N_items,N_obs,2))
        for i in range(self.N_bags):
            for j in range(self.N_items):
                choice_obs = np.random.choice(np.arange(self.L), size=N_obs, replace=False)
                choice_obs = np.sort(choice_obs)
                paths_sub[i,j] = self.paths[i,j,choice_obs,:]

        self.paths_sub = paths_sub

    def get_alpha(self):
        self.labels = self.alpha[:,None]

    def get_m(self):
        self.labels = self.m[:,None]

    def get_nu(self):

        self.labels = self.nu[:,None]

    def compute_plot_naive_norms(self):

        dim_1 = np.array(np.array(self.paths)[:, :, :, 0])
        dim_2 = np.array(np.array(self.paths)[:, :, :, 1])
        input_ = np.concatenate([dim_1, dim_2], axis=2)

        avg_norms = []

        for i in range(self.N_bags):
            avg_norm = np.mean(np.sum(np.array(input_[i][:, :]) ** 2, axis=1))
            avg_norms.append(avg_norm)

        sns.set(font_scale=1)
        fig = plt.figure(figsize=(5, 5))

        order = np.argsort(self.labels[:, 0])
        scaler = MinMaxScaler()
        plt.plot(scaler.fit_transform(np.array(avg_norms)[order][:, None]),
                 scaler.fit_transform(np.array(self.labels)[order]), color='blue')
        plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), ls='--', color='grey')
        plt.xlabel(r'$\mathbb{E}[||x||^2]$')
        plt.ylabel('eccentricity')
        plt.title('labels against naive norm (min-max-scaled for visualization)')
        plt.show()

    def subsample_paths(self, N, same_grid_items=True):

        paths_sub = []

        subs = []
        for bag in range(self.N_bags):
            if same_grid_items:
                sub = np.sort(np.random.choice(np.arange(self.L), N, replace=False))
                paths_sub.append(self.paths[bag][:, sub])
                subs.append(sub)
            else:
                items = []
                sub_items = []
                for item in range(self.N_items):
                    sub = np.sort(np.random.choice(np.arange(self.L), N, replace=False))
                    items.append(self.paths[bag][item, sub])
                    sub_items.append(sub)
                subs.append(sub_items)
                paths_sub.append(items)

        self.paths_sub = np.array(paths_sub)
        self.subs = subs

    def plot(self, N=3, N_items=5):

        sns.set(font_scale=1)
        fig, ax = plt.subplots(2, N, figsize=(N * 5, 5))
        #ax = ax.ravel()
        colors = sns.color_palette("husl", N_items)

        order = np.argsort(self.labels[:, 0])
        samples = np.arange(0, len(order), int(len(order) / N))
        samples = samples[:N]


        for i in range(len(samples)):
            for item in range(N_items):
                ax[0][i].plot(self.t_span, self.paths[order][samples[i]][item, :, 0].T,
                           color=colors[item])
                ax[1][i].plot(self.t_span, self.paths[order][samples[i]][item, :, 1].T,
                           color=colors[item])
            ax[0][i].set_title('label: %.2f' % self.labels[order][samples[i]],fontsize=18)
            ax[0][i].set_xlabel('$t$')
            ax[0][i].set_ylabel('price')
            ax[1][i].set_xlabel('$t$')
            ax[1][i].set_ylabel('volatility')


            #ax[i].legend(loc='upper right')
        plt.tight_layout()
        plt.show()

    def plot2(self, N=3, N_items=5):
        sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
        # sns.set(font_scale=1)
        fig, ax = plt.subplots(1, N, figsize=(N * 5, 5))
        ax = ax.ravel()
        colors = ['dodgerblue', 'tomato']  # N*['blue']#sns.color_palette("RdYlBu", N)
        colors2 = sns.color_palette("RdYlBu", N_items)

        order = np.argsort(self.labels[:, 0])
        samples = np.arange(0, len(order), int(len(order) / N) + 1)
        samples = samples[:N]

        for i in range(len(samples)):

            for item in range(N_items):
                ax[i].plot(self.paths[order][samples[i]][item, :, 0].T, self.paths[order][samples[i]][item, :, 1].T,
                           color=colors[item], linewidth=1)
                marker_style = dict(color=colors[item], linestyle=':', marker='o',
                                    markersize=10, markerfacecoloralt='tab:red', markeredgecolor='black')
                ax[i].plot(self.paths[order][samples[i]][item, 0, 0], self.paths[order][samples[i]][item, 0, 1],
                           **marker_style)
                ax[i].set_title('label: %.2f' % self.labels[order][samples[i]])
            ax[i].axhline(0, ls='--', color='black', linewidth=0.5)
            ax[i].axvline(0, ls='--', color='black', linewidth=0.5)
            ax[i].set_xlabel('$x^1(t)$')
            ax[i].set_ylabel('$x^2(t)$')
            ax[i].set_xlim([0 - 3.5, 3.5])
            ax[i].set_ylim([0 - 3.5, 3.5])

            # ax[i].scatter(self.pos_x[order][samples[i]],self.pos_y[order][samples[i]],marker='+',color='red',s=100,label=r'$(\alpha,\beta)$')
            # ax[i].legend(loc='upper right')
        plt.tight_layout()
        plt.savefig('ellipsis.pdf')
        plt.show()

    def plot_subsampled_paths(self, N=3, N_items=5):

        sns.set(font_scale=1)
        fig, ax = plt.subplots(N, N, figsize=(10, 10))
        ax = ax.ravel()
        colors = sns.color_palette("husl", N * N)

        order = np.argsort(self.labels[:, 0])
        samples = np.arange(0, len(order), int(self.N_bags / N ** 2))

        for i in range(N * N):

            for item in range(N_items):
                # ax[i].plot(self.paths[order][samples[i]][item, :, 0].T, self.paths[order][samples[i]][item, :, 1].T, color=colors[i])
                ax[i].scatter(self.paths_sub[order][samples[i]][item, :, 0].T,
                              self.paths_sub[order][samples[i]][item, :, 1].T,
                              color=colors[i])
                ax[i].set_title('label: %.2f' % self.labels[order][samples[i]])
            ax[i].axhline(0, ls='--', color='black')
            ax[i].axvline(0, ls='--', color='black')
        plt.show()

    def one_plot(self, N=3):

        sns.set(font_scale=1)
        fig = plt.figure(figsize=(5, 5))
        colors = sns.color_palette("husl", N)

        order = np.argsort(self.labels[:, 0])
        samples = np.arange(0, len(order), int(self.N_bags / N))

        for i in range(N):
            plt.plot(self.paths[order][samples[i], 0, :, 0].T - self.pos_x[order][samples[i]],
                     self.paths[order][samples[i], 0, :, 1].T - self.pos_y[order][samples[i]], color=colors[i],
                     label='label: %.2f' % self.labels[:, 0][order][samples[i]])

        m = np.maximum(self.paths[order][samples, 0, :, 0] - self.pos_x[order][samples][:, None],
                       self.paths[order][samples, 0, :, 1] - self.pos_y[order][samples][:, None])
        m = np.max(m)
        plt.xlim([-m - m / 10, m + m / 10])
        plt.ylim([-m - m / 10, m + m / 10])
        plt.plot(m * np.cos(self.t_span), m * np.sin(self.t_span), ls='--', color='grey')
        plt.xlabel('$x^1(t)$')
        plt.ylabel('$x^2(t)$')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()


def rbf_kernel(x1, x2, variance=1, lengthscale=0.0001):
    return variance * np.exp(-1 * ((x1 - x2) ** 2) / (2 * lengthscale))


def gram_matrix(xs, variance):
    return [[rbf_kernel(x1, x2, variance=variance) for x2 in xs] for x1 in xs]