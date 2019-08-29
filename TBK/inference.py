from numba import jit
import numpy as np
import scipy.optimize
import scipy.stats
import scipy.special
from TBK.BP.bp import beta_poisson_log_likelihood, beta_poisson3_log_likelihood


@jit()
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


def maximum_likelihood(vals: np.array, model: str ='BP3') -> np.array:
    """

    """
    if model == 'BP3':
        params = moment_based(vals)
        if any(params < 0):
            params = np.array([10, 10, 10])

        # TODO: fix initial guess between bounds
        bounds = ((1e-3, 1e3), (1e-3, 1e3), (1, 1e4))
    elif model == 'BP4':
        params = np.array([10, 10, 10, 0.5])
        bounds = ((1e-3, 1e3), (1e-3, 1e3), (1, 1e4), (1e-3, 0.9999))
    else:
        raise NotImplementedError

    # let scipy do the complicated param estimation
    print(params)
    print(bounds)
    res = scipy.optimize.minimize(beta_poisson_log_likelihood,
                                  params,
                                  args=vals[..., np.newaxis],
                                  method='L-BFGS-B',
                                  bounds=bounds)

    # if not successful return nan, else the result
    if not res.success:
        return np.array([np.nan, np.nan, np.nan])
    return res.x
