import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from fbm import FBM

class SDE_FBM():

    # equation: x^2/a^2 + y^2/b^2 = 1
    # parametrix equation: (x,y) = (acos(t),bsin(t))  t in [0,2pi]

    def __init__(self):
        self.paths = None
        self.labels = None

    def set_parameters(self, N_bags, N_items, t_span, spec_param):
        self.N_bags = N_bags
        self.N_items = N_items
        self.t_span = t_span
        self.L = len(t_span)
        self.spec_param = spec_param


    def generate_data(self, N_bags=100, N_items=15, t_span=np.linspace(0, 1, 100),spec_param={'sigma': [1., 2.],'hurst':[0.2,0.8]},y0_mean=1.,y0_stdv=0.):

        self.set_parameters(N_bags, N_items, t_span, spec_param)

        # generate the labels. If spec_param[param][1]= spec_param[param][0], then the parameter param is fixed
        sigmas = (spec_param['sigma'][1] - spec_param['sigma'][0]) * np.random.rand(N_bags) + spec_param['sigma'][0] * np.ones(N_bags)
        mus = (spec_param['mu'][1] - spec_param['mu'][0]) * np.random.rand(N_bags) + spec_param['mu'][0] * np.ones(N_bags)
        hursts = (spec_param['hurst'][1] - spec_param['hurst'][0]) * np.random.rand(N_bags) + spec_param['hurst'][0] * np.ones(N_bags)
        y0 = y0_mean + y0_stdv * np.random.randn(N_bags)

        # generate the paths
        times = []
        paths = []

        for i in range(N_bags):

            items = []


            f = FBM(n=(len(t_span)-1), hurst=hursts[i], length=t_span[-1], method='daviesharte')
            t_values = f.times()
            times.append(t_values)

            for j in range(N_items):
                fbm_sample = f.fbm()

                item = y0[i]*np.exp(mus[i]*t_span[:,None] + sigmas[i]*fbm_sample[:,None])

                items.append(item)

            paths.append(items)

        self.paths = np.array(paths)
        self.mus = mus[:,None]
        self.sigmas = sigmas[:,None]
        self.hursts = hursts[:,None]
        self.times = np.array(times)
        self.y0 = y0[:,None]

    def get_hurst(self):
        self.labels = self.hursts

    def get_sigma(self):
        self.labels = self.sigmas

    def get_mu(self):
        self.labels = self.mus

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

    def subsample_paths(self,N,same_grid_items=True):

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

    def subsample_items(self,N):

        sub = np.sort(np.random.choice(np.arange(self.N_items), N, replace=False))
        self.paths_sub = self.paths[:,sub,:,:]


    def plot(self, N=3, N_items=5):

        sns.set(font_scale=1)
        fig, ax = plt.subplots(1,N, figsize=(N*5, 5))
        ax = ax.ravel()
        colors = sns.color_palette("hls", int(self.N_bags / N))

        order = np.argsort(self.labels[:, 0])
        samples = np.arange(0, len(order), int(self.N_bags / N))

        for i in range(int(len(samples))):

            for item in range(N_items):

                ax[i].plot(self.times[order][samples[i]].T, self.paths[order][samples[i]][item,:,0].T,
                           color=colors[i])
                ax[i].set_title('label: %.2f' % self.labels[order][samples[i]])
        plt.tight_layout()
        plt.show()

    def plot_subsampled_paths(self, N=3, N_items=5):

        sns.set(font_scale=1)
        fig, ax = plt.subplots(N, N, figsize=(10, 10))
        ax = ax.ravel()
        colors = sns.color_palette("hls", N * N)

        order = np.argsort(self.labels[:, 0])
        samples = np.arange(0, len(order), int(self.N_bags / (N ** 2)))

        for i in range(len(samples)):

            for item in range(N_items):
                ax[i].plot(self.times[order][samples[i]][item, :].T, self.paths[order][samples[i]][item, :].T,
                           color=colors[i])
                ax[i].scatter(self.times_sub[order][samples[i]][item, :].T, self.paths_sub[order][samples[i]][item, :].T,
                           color=colors[i])
                ax[i].set_title('label: %.2f' % self.labels[order][samples[i]])

        plt.show()


class SDE_FBM_2D_Lagged():

    # equation: x^2/a^2 + y^2/b^2 = 1
    # parametrix equation: (x,y) = (acos(t),bsin(t))  t in [0,2pi]

    def __init__(self):
        self.paths = None
        self.labels = None

    def set_parameters(self, N_bags, N_items, t_span, spec_param):
        self.N_bags = N_bags
        self.N_items = N_items
        self.t_span = t_span
        self.L = len(t_span)
        self.spec_param = spec_param


    def generate_data(self, N_bags=100, N_items=15, t_span=np.linspace(0, 1, 100),spec_param={'sigma1': [1., 2.],'sigma2':[0.2,0.8],'hurst':[0.2,0.8]},lag=3):

        self.set_parameters(N_bags, N_items, t_span, spec_param)

        # generate the labels. If spec_param[param][1]= spec_param[param][0], then the parameter param is fixed
        sigma1 = (spec_param['sigma1'][1] - spec_param['sigma1'][0]) * np.random.rand(N_bags) + spec_param['sigma1'][0] * np.ones(N_bags)
        sigma2 = (spec_param['sigma2'][1] - spec_param['sigma2'][0]) * np.random.rand(N_bags) + spec_param['sigma2'][0] * np.ones(N_bags)
        hursts = (spec_param['hurst'][1] - spec_param['hurst'][0]) * np.random.rand(N_bags) + spec_param['hurst'][0] * np.ones(N_bags)

        # generate the paths
        times = []
        paths = []

        for i in range(N_bags):

            items = []


            f = FBM(n=(len(t_span)-1), hurst=hursts[i], length=t_span[-1], method='daviesharte')
            t_values = f.times()
            times.append(t_values)

            for j in range(N_items):
                fbm_sample = f.fbm()
                x1 = np.exp(sigma1[i]*fbm_sample[:-lag])
                x2 = np.exp(sigma2[i] * fbm_sample[lag:])

                items.append(np.hstack([x1[:,None],x2[:,None]]))

            paths.append(items)

        self.paths = np.array(paths)
        self.sigma1 = sigma1[:,None]
        self.sigma2 = sigma2[:, None]
        self.hursts = hursts[:,None]
        self.times = np.array(times)


    def get_hurst(self):
        self.labels = self.hursts

    def get_sigma(self):
        self.labels = self.sigmas

    def get_ratio(self):
        self.labels = self.sigma1/self.sigma2

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

    def subsample_paths(self,N,same_grid_items=True):

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

    def subsample_items(self,N):

        sub = np.sort(np.random.choice(np.arange(self.N_items), N, replace=False))
        self.paths_sub = self.paths[:,sub,:,:]


    def plot(self, N=3, N_items=5,method='2d'):

        sns.set(font_scale=1)
        fig, ax = plt.subplots(1,N, figsize=(N*5, 5))
        ax = ax.ravel()
        colors = sns.color_palette("hls", int(self.N_bags / N))

        order = np.argsort(self.labels[:, 0])
        samples = np.arange(0, len(order), int(self.N_bags / N))

        for i in range(int(len(samples))):

            for item in range(N_items):
                if method=='2d':
                    ax[i].plot(self.paths[order][samples[i]][item, :, 0].T,self.paths[order][samples[i]][item, :, 1].T,
                               color=colors[i])
                else:
                    ax[i].plot(self.paths[order][samples[i]][item,:,0].T,
                           color=colors[i])
                    ax[i].plot(self.paths[order][samples[i]][item,:,1].T,
                           color=colors[i])
                ax[i].set_title('label: %.2f' % self.labels[order][samples[i]])
        plt.tight_layout()
        plt.show()

    def plot_subsampled_paths(self, N=3, N_items=5):

        sns.set(font_scale=1)
        fig, ax = plt.subplots(N, N, figsize=(10, 10))
        ax = ax.ravel()
        colors = sns.color_palette("hls", N * N)

        order = np.argsort(self.labels[:, 0])
        samples = np.arange(0, len(order), int(self.N_bags / (N ** 2)))

        for i in range(len(samples)):

            for item in range(N_items):
                ax[i].plot(self.times[order][samples[i]][item, :].T, self.paths[order][samples[i]][item, :].T,
                           color=colors[i])
                ax[i].scatter(self.times_sub[order][samples[i]][item, :].T, self.paths_sub[order][samples[i]][item, :].T,
                           color=colors[i])
                ax[i].set_title('label: %.2f' % self.labels[order][samples[i]])

        plt.show()