"""
Estimate (infer) the parameters of the IAP and/or Beta Poisson models.
"""
from typing import Tuple

import numpy as np
import scipy.optimize
import scipy.stats
import scipy.special

from .bp import beta_poisson_log_likelihood


def moment_based(vals: np.array) -> np.array:
    """
    Estimate parameters lambda, mu, and nu based on the values' first three moments.
    Based on the paper: Markovian Modelling of Gene Product Synthesis
    """
    assert len(vals.shape) == 1, "vals should be an 1D array"

    # calculate the moments (27)
    m_1 = np.sum(vals) / len(vals)
    m_2 = np.sum(vals * (vals - 1)) / len(vals)
    m_3 = np.sum(vals * (vals - 1) * (vals - 2)) / len(vals)

    # if any of the moments equals zero, then we would attempt zero divisions, so we return nan
    if 0 in [m_1, m_2]:
        return np.array([np.nan, np.nan, np.nan])

    r_1 = m_1
    r_2 = m_2 / m_1
    r_3 = m_3 / m_2

    la_denom = (r_1 * r_2 - 2 * r_1 * r_3 + r_2 * r_3)
    nu_denom = (r_1 - 2 * r_2 + r_3)

    # if any of the denominators is zero we return nan
    if 0 in [la_denom, nu_denom]:
        return np.array([np.nan, np.nan, np.nan])

    # estimate the parameters (26)
    la_est = (2 * r_1 * (r_3 - r_2)) / la_denom
    mu_est = (2 * (r_3 - r_2) * (r_1 - r_3) * (r_2 - r_1)) / (la_denom * nu_denom)
    nu_est = (2 * r_1 * r_3 - r_1 * r_2 - r_2 * r_3) / nu_denom

    return np.array([la_est, mu_est, nu_est])


def get_bounds_params3(vals: np.array) -> Tuple[tuple, np.array]:
    """
    Estimate the initial parameters of the BP3 model, and its bounds.

    The parameters are estimated based on the first three moments of the values. If they are out of
    bounds, they are set to their closest value inside the bounds.
    """
    assert len(vals.shape) == 1, "vals should be an 1D array"

    # our parameter estimation bounds
    bounds = ((1e-6, 1e6), (1e-6, 1e6), (1e-6, 1e6))

    params = moment_based(vals)
    if np.isnan(params).any() or any(params < 0):
        params = np.array([10, 10, 10])

    # force estimated params between bounds
    for i, param in enumerate(params):
        params[i] = sorted([bounds[i][0], bounds[i][1], param])[1]

    return bounds, params


def get_bounds_params4() -> Tuple[tuple, np.array]:
    """
    Estimate the initial parameters of the BP4 model, and its bounds.

    The parameters are arbitrarily set to;
    lambda1: 10, mu: 10, nu: 10, lambda2: 0.5
    """
    # our parameter estimation bounds
    bounds = ((1e-6, 1e6), (1e-6, 1e6), (1e-6, 1e6), (1e-6, 1))
    params = np.array([10, 10, 10, 0.5])

    return bounds, params


def maximum_likelihood(_vals: np.array, model: str = 'BP3') -> np.array:
    """
    Get the most likely parameters of either the BP3 or the BP4 model.

    Parameters are estimated by scipy optimization.
    """
    # remove the missing value data
    vals = _vals[~np.isnan(_vals)]

    # when no gene is expressed or only 1 value, we shouldn't try to infer parameters
    if not np.any(vals) or not vals.size > 1:
        return np.array([np.nan, np.nan, np.nan])

    assert len(vals.shape) == 1, "vals should be an 1D array"

    if model == 'BP3':
        bounds, params = get_bounds_params3(vals)
    elif model == 'BP4':
        bounds, params = get_bounds_params4()
    else:
        raise NotImplementedError

    # convert our values to the unique values and their counts
    vals = np.unique(vals, return_counts=True)

    # let scipy do the complicated param estimation
    res = scipy.optimize.minimize(beta_poisson_log_likelihood,
                                  params,
                                  args=(vals, ),
                                  method='L-BFGS-B',
                                  bounds=bounds)

    # if not successful return nan, else the result
    if not res.success:
        return np.array([np.nan, np.nan, np.nan])

    # FIXME: BP4
    alpha, beta, lambd = res.x
    return np.array([alpha, beta, lambd])


def wald_test(_vals_1: np.array, _vals_2: np.array):
    """
    Do the wald test
    """
    vals_1, vals_2 = np.copy(_vals_1), np.copy(_vals_2)

    # calculate the most likely parameters (theta hat)
    theta_hat_1 = maximum_likelihood(vals_1)
    theta_hat_2 = maximum_likelihood(vals_2)

    if np.isnan(theta_hat_1).any() or np.isnan(theta_hat_2).any():
        return theta_hat_1, theta_hat_2, np.array([np.nan, np.nan, np.nan])

    # store the likelihood of the second model
    zero_hypothesis = beta_poisson_log_likelihood(theta_hat_2, vals_2)

    probabilities = np.zeros(3)
    for i, _ in enumerate(theta_hat_1):
        bounds, _ = get_bounds_params3(vals_2)

        # now fix one of the params of the second model to the value of model 1
        bounds = np.array(bounds)
        bounds[i] = theta_hat_1[i], theta_hat_1[i]
        bounds = tuple(tuple(i) for i in np.array(bounds))

        theta_hat_2_c = theta_hat_2.copy()
        theta_hat_2_c[i] = theta_hat_1[i]

        # now optimize
        res = scipy.optimize.minimize(beta_poisson_log_likelihood,
                                      theta_hat_2_c,
                                      args=vals_2,
                                      method='L-BFGS-B',
                                      bounds=bounds)

        theta_zero = beta_poisson_log_likelihood(res.x, vals_2)

        # calculate the
        probability = 1 - scipy.stats.chi2.cdf(2*(theta_zero - zero_hypothesis), 1)
        probabilities[i] = probability

    return theta_hat_1, theta_hat_2, probabilities
