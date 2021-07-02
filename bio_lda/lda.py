import numpy as np
from numba import int32, float32, float64, jit    # import the types
from numba.experimental import jitclass
import numba

spec = [
    ("t", int32), 
    ("a", float64), 
    ("b", float64),
    ("mu1", numba.float64[:]),
    ("mu2", numba.float64[:]),
    ("v1", float64),
    ("v2", float64),
    ("w", numba.float64[:]),
    ("l", float64),
    ("D", int32),
    ("eta", float64),
    ("gamma", float64),
    ("mu", numba.float64[:]),
    ("nu", numba.float64[:])  
]

spec_bio = [
    ("t", int32),  
    ("b", float64),
    ("mu1", numba.float64[:]),
    ("mu2", numba.float64[:]),
    ("v1", float64),
    ("v2", float64),
    ("w", numba.float64[:]),
    ("l", float64),
    ("K", int32),
    ("D", int32),
    ("tau", int32),
    ("eta", float64),
    ("gamma", float64),
    ("mu", numba.float64[:])
    
]

offline_spec = [
    ("t", numba.int32),
    ("w", numba.float64[:,:]),
    ("l", float64),
    ("K", int32),
    ("D", int32),
    ("tau", int32),
    ("eta", float64),
    ("gamma", float64)
]


@jit(nopython=True)
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
    return 5e-8

@jit(nopython=True)
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
    return 5e-8

#@jitclass(offline_spec)
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

    def __init__(self, D, eta, gamma):

        self.eta = eta
        self.gamma = gamma
        
        self.t = 0
        self.w = np.random.randn(D, 1)
        self.l = 1
        self.D = D

    def fit(self, mu1, mu2, SW):

        t, w, l = self.t, self.w, self.l
        eta = self.eta
        gamma = self.gamma
        
        w = w + eta * (mu1 - mu2 - l *SW@w)
        l = l + gamma * (w.T@SW@w -1)
        
        self.w = w
        self.l = l
        self.t += 1
        
@jitclass(spec)
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

    def __init__(self, D):
        
        self.t = 1
        
        
        self.a = 1/2
        self.b = 1/2
        
        self.w = np.random.randn(D)/np.sqrt(D)
        self.l = 1

        self.mu1 = np.zeros(D)
        self.mu2 = np.zeros(D)
        
        self.D = D
        
    def fit_next(self, x, r, s, eta, gamma):

        assert x.shape == (self.D,)
        

        t, w, l = self.t, self.w, self.l
        mu1, mu2 = self.mu1, self.mu2
        a, b = self.a, self.b
        
        z = w.T@x
        
        a = max(a + (r-a)/t, 1e-5)
        b = max(b + (s-b)/t, 1e-5)
        
        mu1 = mu1 + ((r/a)*x-mu1)/t
        mu2 = mu2 + ((s/b)*x-mu2)/t
        
        mu = mu1*r + mu2*s
        nu = w.T@mu
        
        e_step = eta
        g_step = gamma
        
        w =  w + e_step*(r/a-s/b)*x - e_step*l*(z-nu)*(x-mu)
        l = l + g_step*((z-nu)**2 - 1)
        
        self.a = a
        self.b = b
        self.mu1 = mu1
        self.mu2 = mu2
        self.w = w
        self.l = l
        
        
@jitclass(spec_bio)
class Online_BioLDA:
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
        
        self.t = 1
        
        self.eta =  5e-8
        self.gamma =  5e-8
        
        self.a = 1/2
        self.b = 1/2
        
        self.w = np.random.normal(0, 1.0/np.sqrt(D), size=(D,))
        self.l = np.random.normal(0, 1.0)

        self.mu1 = np.array([0.0])
        self.mu2 = np.array([0.0])
        
        self.v1 = 0.0
        self.v2 = 0.0
        
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
        
        mu1 = mu1 + (r_a*x-mu1)/t
        mu2 = mu2 + (s_b*x-mu2)/t
        v1 = v1 + (r_a*y-v1)/t
        v2 = v2 + (s_b*y-v2)/t
        
        e_step = self.eta
        g_step = self.gamma
        
        mu = r*mu1 + s*mu2
        v = r*v1 + s*v2
        w =  w + e_step*(r_a-s_b)*x + e_step*l*(y-v)*(x-mu)
        l = l + g_step*((y-v)**2-1)
        
        self.a = a
        self.b = b
        self.mu1 = mu1
        self.mu2 = mu2
        self.mu = mu
        self.v1 = v1
        self.v2 = v2
        self.w = w
        self.l = l
        