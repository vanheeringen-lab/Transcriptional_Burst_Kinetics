"""
Functions related to the estimation of confidence interval
"""
import copy
import operator as op
import numpy as np

import scipy.optimize
import scipy.interpolate
import scipy.stats

from .bp import beta_poisson_log_likelihood


def _beta_poisson_log_likelihood_burst(params: np.array, vals: np.array) -> float:
    """
    Calculate the negative sum of the log likelihood of values corrected for burst_size.
    """
    assert len(params) == 3 and isinstance(params, np.array), "params should be of length 3 and " \
                                                              "of type numpy.array"
    assert len(vals.shape) == 1, "vals should be a 1D array"
    lambd, burst_size, nu = params
    mu = nu / burst_size
    return beta_poisson_log_likelihood(np.array([lambd, mu, nu]), vals)


def get_param_bound(param_name, original_params, original_bounds):
    """
    Get params and bound depending on which parameter we are using.
    """
    if param_name == 'burst_size':
        # set the bounds specific for the burst-size
        param = original_params[2] / original_params[1]
        bounds = (original_bounds[0],
                  (param, param),
                  original_bounds[2])

    elif param_name == 'burst_freq':
        # set the bounds specific for the burst_freq value
        param = original_params[0]
        bounds = ((param, param),
                  original_bounds[1],
                  original_bounds[2])

    return param, bounds


def fit(res, param_name, param, _vals, bounds):
    """
    Fit depending on which parameter we are using.
    """
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

    return res


def bounds_params(_params, _vals, param_name, alpha=0.05):
    """
    Estimate the confidence interval of a parameter (either burst size or frequency).

    Burst frequency simply is parameter lambda, burst size is calculated as nu / mu.
    Estimation is done by first slowly decreasing the value of the parameter of interest, and see
    which effect this has on the likelihood of the best possible fit with that changed parameter.
    We keep on decreasing this value until the difference in fit is (quite arbitrarily?) a chi
    squared value + a bit more for better estimation later. We then repeat what we did but then
    slowly increase the value of our parameter.

    Finally we then have a range of parameter values, and the difference in maximum fit. We then
    estimate our confidence interval as where the difference in maximum fit equals our chi squared
    value.
    """
    assert param_name in ['burst_freq', 'burst_size']
    # take copies of values and params not to change them
    original_params = np.copy(_params); vals = np.copy(_vals)

    # calculate the chi square cutoff
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
        param, bound = get_param_bound(param_name, original_params, original_bounds)
        initial_param = param

        # stepsize
        stepsize = 0.05 * param

        # do a max of 100 iterations
        i = 0
        while i < 100:
            param = operator(param, stepsize)

            # don't let our parameter go to unreasonable values
            if param <= 0:
                break

            try:
                res = fit(res, param_name, param, _vals, bound)
            except ValueError:
                break

            # if the fit was unsuccessfull try again with a smaller stepsize
            if not res.success:
                param += param - operator(param, stepsize)
                stepsize *= 0.5
                continue

            vals.append(param)
            likelihoods.append(res.fun)

            if i != 0 and \
                    (2 * (likelihoods[i] - min(likelihoods)) > cutoff + 0.2) and \
                    (likelihoods[i] > likelihoods[i-1]):
                # if we reach our end then stop
                break
            else:
                # increment our iterator
                i += 1

    # store our likelihoods and values
    likelihoods = np.concatenate((np.array(subtract[2][::-1]),  # add the reverse
                                  np.array([original.fun]),     # add the initial
                                  np.array(addition[2])))       # add the increase

    values = np.concatenate((np.array(subtract[1][::-1]),  # add the reverse
                             np.array([initial_param]),    # add the initial
                             np.array(addition[1])))       # add the increase

    # set lowest likelihood to zero
    ll_ratio = 2*(likelihoods - min(likelihoods)).squeeze()

    try:
        # guess the confidence intervals by interpolating (sometimes extrapolating) our found values
        f_1 = scipy.interpolate.interp1d(ll_ratio[:np.argmin(ll_ratio)],
                                         values[:np.argmin(ll_ratio)], kind='cubic',
                                         fill_value="extrapolate")
        f_2 = scipy.interpolate.interp1d(ll_ratio[np.argmin(ll_ratio):],
                                         values[np.argmin(ll_ratio):], kind='cubic',
                                         fill_value="extrapolate")

        conf = np.array([initial_param, f_1(cutoff), f_2(cutoff)])
        return conf, values, ll_ratio

    except (ValueError, np.linalg.linalg.LinAlgError, TypeError):
        return np.array([initial_param, np.nan, np.nan]), values, ll_ratio


def confidence_intervals(param_estimate: tuple, vals: np.array) -> tuple:
    """
    Estimate the confidence intervals for the burst frequency (lambda) and burst size (nu / mu).

    See method bounds_params for an extensive explanation of how the interval is estimated.

    Returns np.array([most_likely, conf_low, conf_high]) for both burst frequency and burst size.
    """
    confidence_freq, *_ = bounds_params(param_estimate, vals, 'burst_freq')
    confidence_size, *_ = bounds_params(param_estimate, vals, 'burst_size')

    return confidence_freq, confidence_size
