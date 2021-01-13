import numpy as np


def eta(t):
    """
    Learning rate for w
    Online: 3e-5
    Offline: 1/(t+5)
    Parameters:
    ====================
    t -- time at which learning rate is to be evaluated
    Output:
    ====================
    e_step -- learning rate at time t
    """

    return 5e-3

def gamma(t):
    """
    Learning rate for lambda
    Online: 3e-5
    Offline: 1/(t+10)
    Parameters:
    ====================
    t -- time at which learning rate is to be evaluated
    Output:
    ====================
    g_step -- learning rate at time t
    """

    return 7e-3

class Offline_LDA:
    """
    Parameters:
    ====================
    K             -- Dimension of PCA subspace to learn
    D             -- Dimensionality of data
    M0            -- Initial guess for the lateral weight matrix M, must be of size K-by-K
    W0            -- Initial guess for the forward weight matrix W, must be of size K-by-D
    learning_rate -- Learning rate as a function of t
    tau           -- Learning rate factor for M (multiplier of the W learning rate)
    Methods:
    ========
    fit_next()
    """

    def __init__(self, K, D, tau=0.5):

        

        self.eta = eta
        self.gamma = gamma
        self.t = 0
        
        
        self.w = np.random.normal(0, 1.0 / np.sqrt(D), size=(D, K))#np.random.normal(0, 1.0 / D, size=(D, K))
        self.l = np.random.normal(0, 1)#np.random.normal(0, 1.0/D)
        
        
        
        self.K = K
        self.D = D
        self.tau = tau

    def fit(self, mu1, mu2, SW):

        t, tau, w, l, K = self.t, self.tau, self.w, self.l, self.K

        if len(mu1.shape) != 2:
            mu1 = np.expand_dims(mu1, axis=-1)
        if len(mu2.shape) != 2:
            mu2 = np.expand_dims(mu2, axis=-1)
        
        assert mu1.shape[1] == 1
        assert mu2.shape[1] == 1
        
        e_step = self.eta(t)
        g_step = self.gamma(t)
        w = w + e_step * (mu1 - mu2 - l * (SW@w))
        l = l + g_step * (w.T@SW@w -1)
        
        self.w = w
        self.l = l
        self.t += 1
        
        
class Online_LDA:
    """
    Parameters:
    ====================
    K             -- Dimension of PCA subspace to learn
    D             -- Dimensionality of data
    M0            -- Initial guess for the lateral weight matrix M, must be of size K-by-K
    W0            -- Initial guess for the forward weight matrix W, must be of size K-by-D
    learning_rate -- Learning rate as a function of t
    tau           -- Learning rate factor for M (multiplier of the W learning rate)
    Methods:
    ========
    fit_next()
    """

    def __init__(self, K, D, tau=0.5):

        

        self.eta = eta
        self.gamma = gamma
        self.t = 1
        
        self.a = 1/2
        self.b = 1/2
        
        self.w = np.random.normal(0, 1.0/np.sqrt(D), size=(D,))
        self.l = np.random.normal(0, 1.0)

        self.mu1 = 0
        self.mu2 = 0
        
        self.v1 = 0
        self.v2 = 0
        
        self.K = K
        self.D = D
        self.tau = tau

    def fit_next(self, x, r, s, m1, m2, SW):

        assert x.shape == (self.D,)
        

        t, tau, w, l = self.t, self.tau, self.w, self.l
        mu1, mu2 = self.mu1, self.mu2
        v1, v2 = self.v1, self.v2
        a, b = self.a, self.b
        
        y = w.T@x
        a = a + 1/t * (r - a)
        b = b + 1/t * (s - b)
        
        r_a = r/a if a != 0 else 0
        s_b = s/b if b != 0 else 0
        
        mu1 = mu1 + r_a*(x-mu1)/t
        mu2 = mu2 + s_b*(x-mu2)/t
        v1 = v1 + r_a*(y-v1)/t
        v2 = v2 + s_b*(y-v2)/t
        
        e_step = self.eta(t)
        g_step = self.gamma(t)
        
        mu = r*mu1 + s*mu2
        v = r*v1 + s*v2
        w =  w + e_step*(r_a-s_b)*x + e_step*l*(y-v)*(x-mu)
        l = l + g_step*((y-v)**2-1)
        
        
        self.a = a
        self.b = b
        self.mu1 = mu1
        self.mu2 = mu2
        self.v1 = v1
        self.v2 = v2
        self.w = w
        self.l = l
        