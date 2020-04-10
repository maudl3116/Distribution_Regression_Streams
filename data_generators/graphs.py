import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import iisignature


class Graph():

    # equation: x^2/a^2 + y^2/b^2 = 1
    # parametrix equation: (x,y) = (acos(t),bsin(t))  t in [0,2pi]

    def __init__(self):
        self.paths = None
        self.labels = None

    def set_parameters(self, N_bags, N_items,G, t_span, spec_param):
        self.N_bags = N_bags
        self.N_items = N_items
        self.t_span = t_span
        self.L = len(t_span)
        self.G = G
        self.spec_param = spec_param


    def generate_data(self, N_bags=100, N_items=15, G=5, t_span=np.linspace(0,10,50), spec_param={'a': [0., 1.], 'b': [0., 1.]}):
        self.set_parameters(N_bags, N_items,G, t_span, spec_param)

        paths = []



        a = (spec_param['a'][1] - spec_param['a'][0]) * np.random.rand(N_bags) + spec_param['a'][0]*np.ones(N_bags)
        b = (spec_param['b'][1]-spec_param['b'][0])*np.random.rand(N_bags) + spec_param['b'][0]*np.ones(N_bags)


        for bag in range(N_bags):
            bag_paths = []
            for item in range(N_items):



                # some factors
                m0 = b[bag] / (a[bag] + b[bag])
                m1 = a[bag] / (a[bag] + b[bag])
                s = a[bag]+b[bag]

                # initial adjacency matrix:
                A_0 = np.zeros((G,G))
                U = np.random.rand(G,G)
                p0 = m0
                A_0[U<p0]=1

                # item path
                item_path = np.zeros((len(t_span), len(A_0[np.triu_indices_from(A_0, k=1)])))

                item_path[0,:] = A_0[np.triu_indices_from(A_0, k=1)]

                for k,t in enumerate(t_span[1:]):
                    A_t = A_0.copy()

                    # number of connexions at the previous time step
                    nb_connexions = np.sum(A_0,axis=(0,1))

                    # with probability a/(a+b)[1-exp{-(a+b)t}] = m[1-exp{-st}]remove the edge
                    p1 = m1*(1-np.exp(-s*t))
                    u = np.random.rand(int(nb_connexions))
                    u = np.where(u < p1, 0, 1)
                    A_t[A_0 == 1] = u.reshape(A_t[A_0 == 1].shape)


                    # number of absence of connexions at the previous time step
                    nb_non_connexions = G*G-np.sum(A_0, axis=(0, 1))

                    # with probability b/(a+b)[1-exp{-(a+b)t}] = m[1-exp{-st}] add the edge
                    p0 = m0 * (1 - np.exp(-s * t))
                    u = np.random.rand(int(nb_non_connexions))
                    u = np.where(u < p0, 1, 0)
                    A_t[A_0 == 0] = u.reshape(A_t[A_0 == 0].shape)

                    item_path[k,:] = A_t[np.triu_indices_from(A_t, k=1)]
                bag_paths.append(item_path)
            paths.append(bag_paths)

            self.paths = np.array(paths)
            self.a = a[:, None]
            self.b = b[:, None]


    def get_a(self):
        self.labels = self.a

    def get_ratio(self):
        self.labels = self.a/(self.a+self.b)


