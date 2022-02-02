import numpy as np
from numpy.random import randn


def get_dataset(seed, x_dim, freq, samples, scale=1, sep=1):
    np.random.seed(seed)
    mu_0 = randn(x_dim,1) # mean of class 0
    mu_1 = randn(x_dim,1) # mean of class 1
    # print(mu_0, mu_1)
    sig = randn(x_dim,x_dim)/np.sqrt(x_dim) # covariance of class 0
    Sigma = sig@sig.T
    class_0 = scale * mu_0 + sig@randn(x_dim,int(samples*(1-freq)))
    class_1 = scale * mu_1 + sig@randn(x_dim,int(samples*freq))
    X = np.concatenate((class_0,class_1),axis=1)
    Y = np.zeros((2,samples)); Y[0,:int(samples*(1-freq))] = 1; Y[1,-int(samples*freq):] = 1

    return (class_0, class_1), (X, Y), (mu_0, mu_1, Sigma)
