import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import iisignature
import sys
sys.path.append('../')
import utils
from fbm import FBM
import torch

class gbm():

    # equation: x^2/a^2 + y^2/b^2 = 1
    # parametrix equation: (x,y) = (acos(t),bsin(t))  t in [0,2pi]

    def __init__(self, N_bags=100, N_items=15, t_span=np.linspace(0, 10, 500), spec_param={'a': [1., 2.], 'b': [1., 2.]},nb_sub=200):
        self.N_bags = N_bags
        self.N_items = N_items

        self.t_span = t_span
        self.L = len(t_span)
        self.spec_param = spec_param
        self.nb_obs = nb_obs
        self.paths = None
        self.labels = None

    def generate_data_GBM(self):


        a = (self.spec_param['a'][1] - self.spec_param['a'][0]) * np.random.rand(self.N_bags) + self.spec_param['a'][0] * np.ones(self.N_bags)
        b = (self.spec_param['b'][1] - self.spec_param['b'][0]) * np.random.rand(self.N_bags) + self.spec_param['b'][0] * np.ones(self.N_bags)

        # sample 2d brownian motions
        BM_samples = utils.brownian(steps=len(self.t_span) - 1, width=self.N_bags * self.N_items,
                                    time=self.t_span[-1])
        # self.samples = BM_samples.reshape(len(self.t_span),self.N_bags,self.N_items,2)
        self.samples = BM_samples.reshape(len(self.t_span), self.N_bags, self.N_items, 1)


        paths = np.exp((a[None,:,None,None]-0.5*b[None,:,None,None]**2)*self.t_span[:,None,None,None]+b[None,:,None,None]*self.samples)

        paths = torch.tensor(paths).transpose(1, 0).transpose(2, 1).numpy()

        paths_sub = np.zeros((self.N_bags,self.N_items,self.nb_obs,1))
        for i in range(self.N_bags):
            for j in range(self.N_items):
                choice_obs = np.random.choice(np.arange(self.L), size=self.nb_obs, replace=False)
                choice_obs = np.sort(choice_obs)
                paths_sub[i, j] = paths[i, j, choice_obs, :]

        self.paths = paths_sub
        self.paths_complete = paths


        self.a = a[:, None]
        self.b = b[:, None]


    def get_a(self):
        self.labels = self.a
    def get_b(self):
        self.labels = self.b

    def get_sig_area(self):

        sig = iisignature.sig(self.paths, 2)
        expected_sig = np.mean(sig,axis=1)

        # 12-21 <-> (for a 2d path) 3-4   [1,2|11,12,21,22]

        areas = 0.5 * (expected_sig[:, 3] - expected_sig[:, 4])

        self.labels = areas[:, None]


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
        fig, ax = plt.subplots(1, N, figsize=(N * 5, 5))
        ax = ax.ravel()
        colors = N * ['blue']  # sns.color_palette("RdYlBu", N)

        order = np.argsort(self.labels[:, 0])
        samples = np.arange(0, len(order), int(len(order) / N) + 1)
        samples = samples[:N]

        for i in range(len(samples)):

            for item in range(N_items):
                ax[i].plot(self.t_span, self.paths_complete[order][samples[i]][item, :, 0],
                           color=colors[i])
                ax[i].set_title('label: %.2f' % self.labels[order][samples[i]])
            ax[i].axhline(0, ls='--', color='black')
            ax[i].axvline(0, ls='--', color='black')
            ax[i].set_xlabel('$t$')
            ax[i].set_ylabel('$x(t)$')
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

