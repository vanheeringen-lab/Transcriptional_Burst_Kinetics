import numpy as np
import scipy.optimize
import scipy.stats
import scipy.special
from TBK.BP.bp import beta_poisson_log_likelihood
from typing import Tuple


def moment_based(vals: np.array) -> np.array:
    """
    Estimate parameters lambda, mu, and nu based on the values' first three moments.
    Based on the paper: Markovian Modelling of Gene Product Synthesis
    """

    # calculate the moments (27)
    m1 = np.sum(vals) / len(vals)
    m2 = np.sum(vals * (vals - 1)) / len(vals)
    m3 = np.sum(vals * (vals - 1) * (vals - 2)) / len(vals)

    # if any of the moments equals zero, then we would attempt zero divisions, so we return nan
    if 0 in [m1, m2]:
        return np.array([np.nan, np.nan, np.nan])

    r1 = m1
    r2 = m2 / m1
    r3 = m3 / m2

    la_denom = (r1 * r2 - 2 * r1 * r3 + r2 * r3)
    nu_denom = (r1 - 2 * r2 + r3)

    # if any of the denominators is zero we return nan
    if 0 in [la_denom, nu_denom]:
        return np.array([np.nan, np.nan, np.nan])

    # estimate the parameters (26)
    la_est = (2 * r1 * (r3 - r2)) / la_denom
    mu_est = (2 * (r3 - r2) * (r1 - r3) * (r2 - r1)) / (la_denom * nu_denom)
    nu_est = (2 * r1 * r3 - r1 * r2 - r2 * r3) / nu_denom

    return np.array([la_est, mu_est, nu_est])


def estimate_bounds_params(vals: np.array, model: str = 'BP3') -> Tuple[tuple, np.array]:
    """

    """
    # our parameter estimation bounds
    bounds = ((1e-3, 1e3), (1e-3, 1e3), (1, 1e4), (1e-3, 0.9999))

    if model == 'BP3':
        params = moment_based(vals)
        if any(params < 0):
            params = np.array([10, 10, 10])

        # TODO: fix initial guess between bounds
        # keep only bounds for first three parameters
        bounds = bounds[:-1]
    elif model == 'BP4':
        params = np.array([10, 10, 10, 0.5])
    else:
        raise NotImplementedError

    return bounds, params


def maximum_likelihood(vals: np.array, model: str = 'BP3') -> np.array:
    """

    """
    bounds, params = estimate_bounds_params(vals, model)

    # let scipy do the complicated param estimation
    res = scipy.optimize.minimize(beta_poisson_log_likelihood,
                                  params,
                                  args=vals[..., np.newaxis],
                                  method='L-BFGS-B',
                                  bounds=bounds)

    # if not successful return nan, else the result
    if not res.success:
        return np.array([np.nan, np.nan, np.nan])
    return res.x
