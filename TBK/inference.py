from numba import jit
import numpy as np
import scipy.optimize
import scipy.stats
import scipy.special


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

    if 0 in [m1, m2]:
        return np.array([np.nan, np.nan, np.nan])

    r1 = m1
    r2 = m2 / m1
    r3 = m3 / m2

    la_denom = (r1 * r2 - 2 * r1 * r3 + r2 * r3)
    nu_denom = (r1 - 2 * r2 + r3)

    if 0 in [la_denom, nu_denom]:
        return np.array([np.nan, np.nan, np.nan])

    # 26
    la_est = (2 * r1 * (r3 - r2)) / la_denom
    mu_est = (2 * (r3 - r2) * (r1 - r3) * (r2 - r1)) / (la_denom * nu_denom)
    nu_est = (2 * r1 * r3 - r1 * r2 - r2 * r3) / nu_denom

    return np.array([la_est, mu_est, nu_est])


def bp3_log_likelihood(vals: np.array, alpha: float, beta: float, lambd: float) -> float:
    """

    """
    x, w = scipy.special.j_roots(50, alpha=beta - 1, beta=alpha - 1)
    # estimate the
    gs = np.sum(w*scipy.stats.poisson.pmf(vals, m=lambd*(x + 1) / 2), axis=1)

    # calculate the probability
    prob = (1 / scipy.special.beta(alpha, beta)) * \
           (2**(-alpha-beta+1)) * \
           gs

    return -np.sum(np.log(prob + 1e-10))


def maximum_likelihood(vals: np.array) -> np.array:
    """

    """
    params = moment_based(vals)
    if any(params < 0):
        params = np.array([10, 10, 10])

    # TODO: fix initial guess between bounds
    bounds = ((1e-3, 1e3), (1e-3, 1e3), (1, 1e4))

    # let scipy do the complicated param estimation
    res = scipy.optimize.minimize(bp3_log_likelihood,
                                  params,
                                  args=vals[..., np.newaxis],
                                  method='L-BFGS-B',
                                  bounds=bounds)

    # if not successful return nan, else the result
    if not res.success:
        return np.array([np.nan, np.nan, np.nan])
    return res.x
