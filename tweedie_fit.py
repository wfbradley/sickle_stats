# Given a numpy array of non-negative values, fit a Tweedie distribution
# by MLE.

import numpy as np
import scipy as sp
import tweedie_dist


def tweedie_fit(data):
    results = {}

    if len(data) <= 2:
        results['mu'] = data[0]
        results['p'] = 0
        if len(data) == 1:
            results['phi'] = 1
        else:
            results['phi'] = np.var(data)
        results['loglikelihood'] = np.inf
        return(results)

    def quasi_loglikelihood(x):
        ll = - np.sum(tweedie_dist.tweedie(
            mu=x[0], p=x[1], phi=x[2]).logpdf(data))
        return(ll)

    x_initial = np.array([np.mean(data), 1.5,
                          np.var(data) / np.power(np.mean(data), 0.66)])

    # SLSQP: sometimes a smidge better than L-BFGS, but sometimes
    # has less stable results.
    mu_bounds = (None, None)
    p_bounds = (1.02, 1.95)
    phi_bounds = (0.5, None)

    opt = sp.optimize.minimize(quasi_loglikelihood, x_initial,
                               method='L-BFGS-B',
                               # method='SLSQP',
                               bounds=(mu_bounds, p_bounds, phi_bounds),
                               # options={'maxiter': 200}
                               )

    results['mu'] = opt.x[0]
    results['p'] = opt.x[1]
    results['phi'] = opt.x[2]
    results['loglikelihood'] = opt.fun
    return(results)
