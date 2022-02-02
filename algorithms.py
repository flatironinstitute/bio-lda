import jax.numpy as np
import numpy as onp
from jax import random, lax, vmap, jit
from jax.random import normal as randn
import matplotlib.pyplot as plt
from tqdm import tqdm

def fit(mu_0, mu_1, Sigma, class_0, class_1, samples, eta, gam, iters=int(1e5)):
    w = np.array(onp.random.randn(mu_1.shape[0],1)/onp.sqrt(mu_1.shape[0]))
    l = np.ones((1,1))
    w_opt = np.linalg.inv(Sigma)@(mu_0-mu_1)
    w_opt = w_opt/np.sqrt(w_opt.T@(Sigma)@w_opt)
    err = np.zeros(iters)
    acc = np.zeros(iters)
    def update_wl(i, upd):
        print(f"iter: {i}")
        w, l, err, acc = upd
        w += eta*(mu_0 - mu_1 - l*Sigma@w)
        l += gam*((w.T@Sigma@w)- 1)
        err.at[i].set(np.linalg.norm(w - w_opt)**2)
        acc.at[i].set((np.sum(w.T@class_0 > 1/2 * w.T@(mu_0+mu_1))  + np.sum(w.T@class_1 < 1/2 * w.T@(mu_0+mu_1)))/samples)
        return (w,l, err, acc)
    print(iters, " iters")
    w,l, err, acc = lax.fori_loop(0, iters, update_wl, (w,l, err, acc))
    return w, l, err, acc