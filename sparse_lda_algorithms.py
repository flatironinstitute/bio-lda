from tqdm.auto import tqdm
# from jax import jit, vmap, grad, random, lax
# from functools import partial
# import jax.numpy as np
# from jax.experimental import host_callback
import numpy as np
import matplotlib.pyplot as plt
import time
from numba import njit, jit
# from jax import random
# from jax.random import normal as randn
import numpy.random as random
from numpy.random import randn
import time

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

class Timer:
    def __init__(self):
        self._start_time = None
    
    def __enter__(self):
        """Start a new timer as a context manager"""
        self.start()
        return self

    def __exit__(self, *exc_info):
        """Stop the context manager timer"""
        self.stop()


    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        print(f"Elapsed time: {elapsed_time:0.4f} seconds")


def eta(e, e2, t):
    return e/(1 + e2*min(t, int(25e6)))

@jit
def fit_bio(w, l, mu, x, y, zeta, l_, c,  t, step, gam):
    r = w.T@x
    
    if y[1] == 0:
        mu += (x - mu)/t
        zeta += (r-zeta)/t
        c += (r-c)/(2*t)
        w += step*(mu - l*(r-zeta)*(x-mu))
        l += gam*step*((r-zeta)**2 - 1)
        l_ += 1
        
    else:
        c += (l_*r - c)/(2*t)
        w -= step*l_*x
        l_ = 1
        
    z = r - c  
    return w, l, r, c, mu, zeta, l_




def run_bio(w, l, c, zeta, l_, mu, X, Y, err, acc, start_epoch, end_epoch, every, e, e2, gam, samples, w_opt):

    # @progress_bar_scan(samples)
    def _run_bio(w, l, c, zeta, l_, mu, X, Y, err, acc, start_epoch, end_epoch, every, e, e2, gam, samples, w_opt):
        for i_epoch in tqdm(range(start_epoch, end_epoch)):

            idx = random.permutation(samples)
            
            for i_sample in range(samples):

                i_iter = i_epoch*samples + i_sample
                t = i_iter + 1

                x = X[:,idx[i_sample]]
                y = Y[:,idx[i_sample]]
                step = eta(e, e2, t)
                
                w, l, r, c, mu, zeta, l_ = fit_bio(w, l, mu, x, y, zeta, l_, c, t, step, gam)
                err.at[i_iter].set(np.linalg.norm(w - w_opt[:,0])**2)
                acc.at[i_iter].set(acc[i_iter-1] * (t-1))
                if (y[1] == 0 and r > c) or (y[1] == 1 and r < c):  
                    acc.at[i_iter].set(acc[i_iter]+1)
                acc.at[i_iter].set(acc[i_iter]/t)
            if i_epoch % every == 0:
                # print(f"step: {step}")
                print(f"err: {str(err[i_iter])}, acc:{str(acc[i_iter])}")
        return w, l, c, zeta, l_, mu, err, acc, step

    return _run_bio(w, l, c, zeta, l_, mu, X, Y, err, acc, start_epoch, end_epoch, every, e, e2, gam, samples, w_opt)