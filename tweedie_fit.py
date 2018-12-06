# Given a numpy array of non-negative values, fit a Tweedie distribution
# by MLE.

import numpy as np
import scipy as sp
import tweedie_dist


def fit_tweedie(data):
    results = {}

    if len(data) <= 2:
        results['mu'] = data[0]
        results['p'] = 0
        if len(data) == 1:
            results['phi'] = 1
        else:
            results['phi'] = np.var(data)
        return(results)

    def quasi_loglikelihood(x):
        # print -np.sum(tweedie_dist(mu=x[0], p=x[1], phi=x[2]).logpdf(data))
        # print type(-np.sum(tweedie_dist(mu=x[0], p=x[1],
        # phi=x[2]).logpdf(data)))
        ll = - np.sum(tweedie_dist.tweedie(
            mu=x[0], p=x[1], phi=x[2]).logpdf(data))
        return(ll)

    x_initial = np.array([np.mean(data), 1.5,
                          np.var(data) / np.power(np.mean(data), 0.66)])

    # SQSLP seems to hang, and L-BFGS-B halts too early,
    # so we're not going to enforce bounds and just hope for the best.
    # mu_bounds = (None, None)
    # p_bounds = (1.02, None)
    # phi_bounds = (0, None)

    opt = sp.optimize.minimize(quasi_loglikelihood, x_initial,
                               method='Nelder-Mead'
                               )

    results['mu'] = opt.x[0]
    results['p'] = opt.x[1]
    results['phi'] = opt.x[2]
    results['loglikelihood'] = opt.fun
    return(results)
