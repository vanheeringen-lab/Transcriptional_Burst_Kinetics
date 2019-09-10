import copy
import numpy as np

import operator as op
import scipy.optimize
import scipy.interpolate
import scipy.stats

from .bp import beta_poisson_log_likelihood


def _beta_poisson_log_likelihood_burst(params: np.array, vals: np.array) -> float:
    """
    Calculate the negative sum of the log likelihood of values for
    """
    assert len(params) == 3 and type(params) == np.ndarray, "params should be of length 3 and of" \
                                                            "type numpy.array"
    assert len(vals.shape) == 1, "vals should be a 1D array"
    lambd, burst_size, nu = params
    mu = nu / burst_size
    return beta_poisson_log_likelihood(np.array([lambd, mu, nu]), vals)


def bounds_params(_params, _vals, param_name):
    """

    """
    assert param_name in ['burst_freq', 'burst_size']
    # take copies of values and params not to change them
    original_params = np.copy(_params); vals = np.copy(_vals)

    # calculate the chi square cutoff
    alpha = 0.05
    cutoff = scipy.stats.chi2.ppf(1-alpha, 1) / 2

    # set our bounds
    original_bounds = ((1e-3, 1e2), (1e-3, 1e3), (1e-3, 1e10))

    # re-estimate and store
    res = scipy.optimize.minimize(beta_poisson_log_likelihood, original_params, args=vals,
                                  method='L-BFGS-B', bounds=original_bounds)
    original = copy.copy(res)

    # store our values
    subtract = (op.sub, [], [])
    addition = (op.add, [], [])

    # to estimate confidence interval we slide our param from our initial guess and see how that
    # affects the likelihood
    for (operator, vals, likelihoods) in [subtract, addition]:
        if param_name == 'burst_size':
            # set the bounds specific for the burst-size
            param = original_params[2] / original_params[1]
            params[1] = param
            bounds = (original_bounds[0],
                      (param, param),
                      original_bounds[2])

        elif param_name == 'burst_freq':
            # set the bounds specific for the burst_freq value
            param = original_params[0]
            bounds = ((param, param),
                      original_bounds[1],
                      original_bounds[2])

        initial_param = param
        # stepsize
        h = 0.05 * param
        # slide
        i = 0
        while i < 100:
            param = operator(param, h)

            if param <= 0:
                break

            try:
                if param_name == 'burst_freq':
                    res.x[0] = param
                    bounds = ((param, param), bounds[1], bounds[2])
                    res = scipy.optimize.minimize(beta_poisson_log_likelihood, res.x, args=_vals,
                                   method='L-BFGS-B', bounds=bounds)
                else:
                    res.x[1] = param
                    bounds = (bounds[0], (param, param), bounds[2])
                    res = scipy.optimize.minimize(_beta_poisson_log_likelihood_burst, res.x,
                                                  args=_vals, method='L-BFGS-B', bounds=bounds)
            except ValueError:
                # TODO: check if values are representative?!
                break

            if not res.success:
                param += param - operator(param, h)
                h *= 0.5
                continue

            vals.append(param)
            likelihoods.append(res.fun)

            if i != 0:
                if (2 * (likelihoods[i] - min(likelihoods)) > cutoff + 0.2) and (likelihoods[i] > likelihoods[i-1]):
                    break

            # increment our iterator
            i += 1

    likelihoods = np.concatenate((np.array(subtract[2][::-1]),  # add the reverse
                         np.array([original.fun]),              # add the initial
                         np.array(addition[2])))                # add the increase

    values = np.concatenate((np.array(subtract[1][::-1]),  # add the reverse
                             np.array([initial_param]),    # add the initial
                             np.array(addition[1])))       # add the increase

    ll_ratio = 2*(likelihoods - min(likelihoods)).squeeze()
    try:
        f_1 = scipy.interpolate.interp1d(ll_ratio[:np.argmin(ll_ratio)],
                       values[:np.argmin(ll_ratio)], kind='cubic', fill_value="extrapolate")
        f_2 = scipy.interpolate.interp1d(ll_ratio[np.argmin(ll_ratio):],
                       values[np.argmin(ll_ratio):], kind='cubic', fill_value="extrapolate")
        conf = np.array([initial_param, f_1(cutoff), f_2(cutoff)])
        return conf, values, ll_ratio
    except (ValueError,np.linalg.linalg.LinAlgError, TypeError):
        return np.array([initial_param, np.nan, np.nan]), values, ll_ratio


def confidence_intervals(param_estimate: tuple, vals: np.array) -> tuple:
    """
    Estimate the confidence intervals for the burst frequency (lambda) and burst size (nu / mu).

    See method bounds_params for an extensive explanation of how the interval is estimated.

    returns
    """
    confidence_freq, *_ = bounds_params(param_estimate, vals, 'burst_freq')
    confidence_size, *_ = bounds_params(param_estimate, vals, 'burst_size')

    return confidence_freq, confidence_size
