import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import iisignature

class Ellipsis():

    # equation: x^2/a^2 + y^2/b^2 = 1
    # parametrix equation: (x,y) = (acos(t),bsin(t))  t in [0,2pi]

    def __init__(self):
      
        self.paths = None
        self.labels = None

    def set_parameters(self,N_bags,N_items,t_span,spec_param,stdv_pos,stdv_noise):
        self.N_bags = N_bags
        self.N_items = N_items
        self.t_span = t_span
        self.L = len(t_span)
        self.spec_param = spec_param
        self.stdv_noise = stdv_noise
        self.stdv_pos = stdv_pos

    def generate_data(self,N_bags=100, N_items=15, t_span = np.linspace(0,2*np.pi,100), spec_param={'a':[1.,2.],'b':[1.,2.]}, stdv_pos=0, noise='gaussian', stdv_noise=0):

        self.set_parameters(N_bags,N_items,t_span,spec_param,stdv_pos,stdv_noise)

        paths = []



        a = (spec_param['a'][1] - spec_param['a'][0]) * np.random.rand(N_bags) + spec_param['a'][0]*np.ones(N_bags)
        b = (spec_param['b'][1]-spec_param['b'][0])*np.random.rand(N_bags) + spec_param['b'][0]*np.ones(N_bags)


        if stdv_pos!=0:
            pos_x = stdv_pos * np.random.randn(N_bags)
            pos_y = stdv_pos * np.random.randn(N_bags)
        else:
            pos_x = np.zeros(N_bags)
            pos_y = np.zeros(N_bags)

        # store the values of the positions of the ellipsis, to be able to visualize them.
        self.pos_x = pos_x
        self.pos_y = pos_y

        if stdv_noise not in spec_param.keys():
            stdv_noise = stdv_noise * np.ones(N_bags)
        else:
            stdv_noise = stdv_noise * np.random.randn(N_bags)


        for i in range(N_bags):

            items = []

            for j in range(N_items):

                x = pos_x[i] + a[i] * np.cos(t_span)
                y = pos_y[i] + b[i] * np.sin(t_span)

                if noise=='gaussian':
                    x = x +  stdv_noise[i]*np.random.randn(len(t_span) )
                    y = y + stdv_noise[i]*np.random.randn(len(t_span) )

                elif noise=='brownian':
                # Does not make much sense
                    x = x + stdv_noise[i]*brownian(len(t_span) - 1, t_span[-1])[:, 0]
                    y = y + stdv_noise[i]*brownian(len(t_span) - 1, t_span[-1])[:, 0]

                items.append(np.hstack([x[:,None],y[:,None]]))

            paths.append(items)


        self.paths = np.array(paths)
        self.a = a[:,None]
        self.b = b[:,None]
        self.stdv_noise = stdv_noise[:,None]

    def get_e1(self):
        # first excentricity
        self.e1 = np.sqrt(1.-np.minimum(self.b,self.a)**2/np.maximum(self.b,self.a)**2)
        self.labels = self.e1

    def get_e2(self):
        # second excentricity
        self.e2 = np.sqrt(np.maximum(self.b, self.a) ** 2 / np.minimum(self.b, self.a) ** 2 -1. )
        self.labels = self.e2

    def get_area(self):

        sig = iisignature.sig(self.paths,2)
        # 12-21 <-> (for a 2d path) 3-4   [1,2|11,12,21,22]

        areas = 0.5*(sig[:, :,3] - sig[:,:, 4])

        self.labels = areas[:,None]

    def get_e3(self):
        # second excentricity
        self.e3 = np.sqrt(np.maximum(self.b, self.a) ** 2 - np.minimum(self.b, self.a) ** 2)/np.sqrt(np.maximum(self.b, self.a) ** 2 + np.minimum(self.b, self.a) ** 2)
        self.labels = self.e3


    def get_e_ang(self):
        # angular excentricity
        self.e_ang = np.arccos(np.minimum(self.b, self.a)/np.maximum(self.b, self.a))
        self.labels = self.e_ang

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
                 scaler.fit_transform(np.array(self.labels)[order]),color='blue')
        plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), ls='--',color='grey')
        plt.xlabel(r'$\mathbb{E}[||x||^2]$' )
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


    def plot(self,N = 3, N_items=5):

        sns.set(font_scale=1)
        fig, ax = plt.subplots(1,N, figsize=(N*5, 5))
        ax = ax.ravel()
        colors = N*['blue']#sns.color_palette("RdYlBu", N)


        order = np.argsort(self.labels[:,0])
        samples = np.arange(0,len(order),int(self.N_bags/(N)))


        for i in range(len(samples)):

            for item in range(N_items):
                ax[i].plot(self.paths[order][samples[i]][item, :, 0].T, self.paths[order][samples[i]][item, :, 1].T, color=colors[i])
                ax[i].set_title('label: %.2f' % self.labels[order][samples[i]])
            ax[i].axhline(0,ls='--',color='black')
            ax[i].axvline(0,ls='--',color='black')
            ax[i].set_xlabel('$x^1(t)$')
            ax[i].set_ylabel('$x^2(t)$')
            ax[i].set_xlim([self.pos_x[order][samples[i]]-3.5,self.pos_x[order][samples[i]]+3.5])
            ax[i].set_ylim([self.pos_y[order][samples[i]]-3.5,self.pos_y[order][samples[i]]+3.5])

            ax[i].scatter(self.pos_x[order][samples[i]],self.pos_y[order][samples[i]],marker='+',color='red',s=100,label=r'$(\alpha,\beta)$')
            ax[i].legend(loc='upper right')
        plt.tight_layout()
        plt.savefig('ellipsis.pdf')
        plt.show()

    def plot_subsampled_paths(self,N = 3, N_items=5):
    
        sns.set(font_scale=1)
        fig, ax = plt.subplots(N, N, figsize=(10, 10))
        ax = ax.ravel()
        colors = sns.color_palette("husl", N*N)

        order = np.argsort(self.labels[:,0])
        samples = np.arange(0,len(order),int(self.N_bags/N**2))
       
        
        for i in range(N*N):

            for item in range(N_items):
                #ax[i].plot(self.paths[order][samples[i]][item, :, 0].T, self.paths[order][samples[i]][item, :, 1].T, color=colors[i])
                ax[i].scatter(self.paths_sub[order][samples[i]][item, :, 0].T, self.paths_sub[order][samples[i]][item, :, 1].T,
                           color=colors[i])
                ax[i].set_title('label: %.2f' % self.labels[order][samples[i]])
            ax[i].axhline(0,ls='--',color='black')
            ax[i].axvline(0,ls='--',color='black')
        plt.show()

    def one_plot(self,N = 3):

        sns.set(font_scale=1)
        fig = plt.figure(figsize=(5, 5))
        colors = sns.color_palette("husl", N)

        order = np.argsort(self.labels[:,0])
        samples = np.arange(0,len(order),int(self.N_bags/N))


        for i in range(N):

            plt.plot(self.paths[order][samples[i],0, :, 0].T-self.pos_x[order][samples[i]], self.paths[order][samples[i],0, :, 1].T-self.pos_y[order][samples[i]], color=colors[i], label = 'label: %.2f' % self.labels[:,0][order][samples[i]])

        m = np.maximum(self.paths[order][samples,0, :, 0]-self.pos_x[order][samples][:,None], self.paths[order][samples,0, :, 1]-self.pos_y[order][samples][:,None])
        m = np.max(m)
        plt.xlim([-m - m/10, m + m/10])
        plt.ylim([-m - m/10, m + m/10])
        plt.plot(m*np.cos(self.t_span), m*np.sin(self.t_span), ls='--', color='grey')
        plt.xlabel('$x^1(t)$')
        plt.ylabel('$x^2(t)$')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

