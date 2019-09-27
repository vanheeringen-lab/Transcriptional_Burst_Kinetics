"""
Collection of beta poisson (3 and 4) functions
"""
import numpy as np
import scipy.special
import scipy.stats

from typing import Tuple

def beta_poisson3(alpha: float, beta: float, lambd: float, size: int = 1) -> np.array:
    """
    Generate data sampled from the beta poisson 3 distribution.
    """
    return np.random.poisson(lambd * np.random.beta(alpha, beta, size))


def beta_poisson4(alpha: float, beta: float, lambda1: float, lambda2: float, size: int = 1) \
        -> np.array:
    """
    Generate data sampled from the beta poisson 4 distribution.
    """
    return lambda2 * beta_poisson3(alpha, beta, lambda1, size)


def beta_poisson4_log_likelihood(
        alpha: float,
        beta: float,
        lambda1: float,
        lambda2: float,
        vals: Tuple[np.ndarray, np.ndarray]  # tuple of values and their corresponding counts
) -> np.array:
    """
    Calculate the log likelihood for your values, based on the beta poisson 4 model.
    """
    # get the unique values and their counts
    uniques, counts = np.copy(vals[0])[..., np.newaxis] / lambda2, np.copy(vals[1])

    # if the optimizer tries to pull a fast one and give us nan values, also return nan
    if np.any(np.isnan([alpha, beta, lambda1, lambda2])):
        return np.nan

    # get the sample points and weights
    x, w = scipy.special.j_roots(50, alpha=beta - 1, beta=alpha - 1)

    # estimate the integral
    chances = np.sum(w*scipy.stats.poisson.pmf(uniques, lambda1 * (x + 1) / 2), axis=1)

    if np.any(np.isnan(chances)):
        return np.nan

    # calculate the probabilities for every unique value
    probs = (1.0 / scipy.special.beta(alpha, beta)) \
            * (2.0**(-alpha-beta+1)) \
            * chances

    # now calculate the log likelihood
    probs = -np.sum(np.log(probs + 1e-10) * counts)

    return probs


def beta_poisson3_log_likelihood(alpha: float, beta: float, lambd: float, vals: np.array) -> float:
    """
    Calculate the negative sum of the log likelihood of values for the beta poisson 3 model.
    """
    # assert len(vals.shape) == 1, "vals should be an 1D array"
    return beta_poisson4_log_likelihood(alpha, beta, lambd, 1.0, vals)


def beta_poisson_log_likelihood(params: np.array, vals: np.array) -> float:
    """
    Calculate the negative sum of the log likelihood of values for either the beta poisson 3 or beta
    poisson 4 model, dependent on the amount of parameters in params.
    """
    assert len(params) == 3 and type(params) == np.ndarray, "params should be of length 3 and of" \
                                                            "type numpy.array"
    # assert len(vals.shape) == 1, "vals should be a 1D array"

    if len(params) not in [3, 4]:
        raise NotImplementedError

    if len(params) == 3:
        return beta_poisson3_log_likelihood(*params, vals)
    return beta_poisson4_log_likelihood(*params, vals)
