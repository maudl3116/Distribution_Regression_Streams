import warnings
warnings.filterwarnings('ignore')

import numpy as np
from fbm import FBM
from utils import cache_result
import hashlib
import json
import os
import pickle

def fOU_generator(a,n=0.3,h=0.2,length=300):
    
    fbm_increments = np.diff(FBM(length, h).fbm())
    # X(t+1) = X(t) - a(X(t)-m) + n(W(t+1)-W(t))
    x0 = np.random.normal(1,0.1)
    x0 = 0.5
    m = x0
    price = [x0]
    for i in range(length):
        p = price[i] - a*(price[i]-m) + n*fbm_increments[i]
        price.append(p)
    return np.array(price)

class DatasetRoughVol():
    """
    Class for computing the path signatures.
    """

    def __init__(self, M, N, L, ymin, ymax, *kwargs):
        """
        Parameters
        ----------
        dataset : string
            Name of the dataset
        dataset_type: string
            Whether it is the train, validation or the test set
        truncation_level : int
            Path signature trucation level.
        add_time: bool
            Whether add time was applied
        lead_lag_transformation : bool
            Whether lead-lag was applied.
        """
        self.name = 'RoughVol'
        self.M = M 
        self.N = N 
        self.L = L 
        self.ymin = ymin 
        self.ymax = ymax

    @cache_result
    def generate(self):
        X = []
        y = np.array((self.ymax-self.ymin)*np.random.rand(self.M)+self.ymin)
        for a in tqdm(y):
            intermediate = []
            for n in range(N):
                path = np.exp(fOU_generator(a, length=self.L)).reshape(-1,1)
                intermediate.append(path)
            X.append(intermediate)
        return [X, y]

    def __str__(self):
        """
        Convert object to string. Used in conjunction with cache_result().
        """
        return str((self.name,
                    self.M,
                    self.N,
                    self.L,
                    self.ymin,
                    self.ymax))

