"""
Estimate (infer) the parameters of the IAP and/or Beta Poisson models.
"""
from typing import Tuple

import numpy as np
import scipy.optimize
import scipy.stats
import scipy.special

from tbk.bp.bp import beta_poisson_log_likelihood


def moment_based(vals: np.array) -> np.array:
    """
    Estimate parameters lambda, mu, and nu based on the values' first three moments.
    Based on the paper: Markovian Modelling of Gene Product Synthesis
    """

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


def get_bounds_params(vals: np.array, model: str = 'BP3') -> Tuple[tuple, np.array]:
    """
    Estimate the initial parameters of the model, and its bounds.

    When the model is BP3 the parameters are estimated based on the first three moments of the
    values. If they are out of bounds, they are set to their closest value inside the bounds.

    The parameters for the BP4 model are arbitrarily set to;
    lambda1: 10, mu: 10, nu: 10, lambda2: 0.5
    """
    # our parameter estimation bounds
    bounds = ((1e-3, 1e3), (1e-3, 1e3), (1e-3, 1e4), (1e-3, 0.9999))

    if model == 'BP3':
        params = moment_based(vals)
        if any(params < 0):
            params = np.array([10, 10, 10])

        # force estimated params between bounds
        for i, param in enumerate(params):
            params[i] = sorted([bounds[i][0], bounds[i][1], param])[1]

        # keep only bounds for first three parameters
        bounds = bounds[:-1]
    elif model == 'BP4':
        params = np.array([10, 10, 10, 0.5])
    else:
        raise NotImplementedError

    return bounds, params


def maximum_likelihood(vals: np.array, model: str = 'BP3') -> np.array:
    """
    Get the most likely parameters of either the BP3 or the BP4 model.

    Parameters are estimated by scipy optimization
    """
    bounds, params = get_bounds_params(vals, model)

    # let scipy do the complicated param estimation
    res = scipy.optimize.minimize(beta_poisson_log_likelihood,
                                  params,
                                  args=vals[..., np.newaxis],
                                  method='L-BFGS-B',
                                  bounds=bounds)

    # if not successful return nan, else the result
    if not res.success:
        return np.array([np.nan, np.nan, np.nan])

    # FIXME: BP4
    lambda1, alpha, beta = res.x
    return np.array([alpha, beta, lambda1])
