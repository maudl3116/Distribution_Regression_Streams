import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import iisignature


class Circuit():

    # equation: x^2/a^2 + y^2/b^2 = 1
    # parametrix equation: (x,y) = (acos(t),bsin(t))  t in [0,2pi]

    def __init__(self,N_bags=50, N_items=15, t_span=np.linspace(0,10,10000), nb_obs=100, spec_param={'C':[0,5],'omega':[1,50]}):

        self.paths = None
        self.labels = None
        self.N_bags = N_bags
        self.N_items = N_items
        self.t_span = t_span
        self.L = len(t_span)
        self.spec_param = spec_param
        self.nb_obs = nb_obs


    def generate_data(self):


        paths = np.zeros((self.N_bags,self.N_items,self.nb_obs,2))

        omega = (self.spec_param['omega'][1] - self.spec_param['omega'][0]) * np.random.rand(self.N_bags) + self.spec_param['omega'][
            0] * np.ones(self.N_bags)
        C = (self.spec_param['C'][1] -self.spec_param['C'][0]) * np.random.rand(self.N_bags) + self.spec_param['C'][0] * np.ones(self.N_bags)
        phi = (self.spec_param['phi'][1] - self.spec_param['phi'][0]) * np.random.rand(self.N_bags) + self.spec_param['phi'][0] * np.ones(self.N_bags)

        # generate one path per bag
        paths_ref = np.zeros((self.N_bags,self.L,2))
        for i in range(self.N_bags):
            paths_ref[i,:,0] = np.cos(omega[i]*self.t_span)
            paths_ref[i, :, 1] = C[i]*omega[i]*np.cos(omega[i] * self.t_span + phi[i])

        for i in range(self.N_bags):
            for j in range(self.N_items):
                choice_obs = np.random.choice(np.arange(self.L), size=self.nb_obs, replace=False)
                choice_obs = np.sort(choice_obs)
                paths[i,j] = paths_ref[i,choice_obs,:]

        self.paths = paths
        self.C = C
        self.phi = phi
        self.P = (1/self.t_span[-1])*np.sum(paths_ref[:,:,0]*paths_ref[:,:,1],axis=1)

    def generate_data_RLC(self):

        paths = np.zeros((self.N_bags, self.N_items, self.nb_obs, 2))

        omega = (self.spec_param['omega'][1] - self.spec_param['omega'][0]) * np.random.rand(self.N_bags) + \
                self.spec_param['omega'][
                    0] * np.ones(self.N_bags)
        em = (self.spec_param['em'][1] - self.spec_param['em'][0]) * np.random.rand(self.N_bags) + \
                self.spec_param['em'][
                    0] * np.ones(self.N_bags)
        R = (self.spec_param['R'][1] - self.spec_param['R'][0]) * np.random.rand(self.N_bags) + self.spec_param['R'][
            0] * np.ones(self.N_bags)
        L = (self.spec_param['L'][1] - self.spec_param['L'][0]) * np.random.rand(self.N_bags) + \
              self.spec_param['L'][0] * np.ones(self.N_bags)

        C = (self.spec_param['C'][1] - self.spec_param['C'][0]) * np.random.rand(self.N_bags) + \
              self.spec_param['C'][0] * np.ones(self.N_bags)


        X_L = omega*L
        X_C = 1./(omega*C)
        Z = np.sqrt(R**2+(X_L-X_C)**2)
        I_m = em*(1./Z)
        tan_phi = (X_L-X_C)*(1./R)

        # generate one path per bag
        paths_ref = np.zeros((self.N_bags, self.L, 2))
        for i in range(self.N_bags):
            paths_ref[i, :, 0] = em[i] * np.sin(omega[i] * self.t_span)
            paths_ref[i, :, 1] = I_m[i]* np.sin(omega[i] * self.t_span - np.arctan(tan_phi[i]))
        figure = plt.figure(figsize=(30, 10))
        plt.plot(self.t_span, paths_ref[0, :, 0])
        plt.plot(self.t_span, paths_ref[0, :, 1])
        plt.title('checking the frequency')
        plt.show()

        for i in range(self.N_bags):
            for j in range(self.N_items):
                choice_obs = np.random.choice(np.arange(self.L), size=self.nb_obs, replace=False)
                choice_obs = np.sort(choice_obs)
                paths[i, j] = paths_ref[i, choice_obs, :]

        self.paths = paths
        self.C = C
        self.tan_phi = tan_phi
        self.P = (1 / self.t_span[-1]) * np.sum(paths_ref[:, :, 0] * paths_ref[:, :, 1], axis=1)

    def get_C(self):
        self.labels = self.C[:,None]

    def get_phi(self):
        self.labels = self.phi[:,None]

    def get_P(self):

        self.labels = self.P[:,None]

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
        colors = sns.color_palette("husl", N)

        order = np.argsort(self.labels[:, 0])
        samples = np.arange(0, len(order), int(len(order) / N) + 1)
        samples = samples[:N]

        for i in range(len(samples)):

            for item in range(N_items):
                ax[i].scatter(self.paths[order][samples[i]][item, :, 0].T, self.paths[order][samples[i]][item, :, 1].T,
                           color=colors[item],s=1)
                ax[i].set_title('label: %.2f' % self.labels[order][samples[i]])
            ax[i].axhline(0, ls='--', color='black')
            ax[i].axvline(0, ls='--', color='black')
            ax[i].set_xlabel('$x^1(t)$')
            ax[i].set_ylabel('$x^2(t)$')
            ax[i].set_xlim([-4, 4])
            ax[i].set_ylim([-4,4])

            #ax[i].legend(loc='upper right')
        plt.tight_layout()
        plt.savefig('ellipsis.pdf')
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