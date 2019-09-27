"""
Collection of beta poisson (3 and 4) functions
"""
import numpy as np
import scipy.special
import scipy.stats


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


def beta_poisson4_likelihood(
        alpha: float,
        beta: float,
        lambda1: float,
        lambda2: float,
        _vals: np.array
) -> np.array:
    """
    Calculate the likelihood for each value in your array of values, based on the beta poisson 4
    model.
    """
    # scale to lambda2
    vals = np.copy(_vals)[..., np.newaxis] / lambda2

    # if the optimizer tries to pull a fast one and give us nan values, also return nan
    if np.any(np.isnan([alpha, beta, lambda1, lambda2])):
        return np.full(vals.size, np.nan)

    # get the sample points and weights
    x, w = scipy.special.j_roots(50, alpha=beta - 1, beta=alpha - 1)

    # estimate the integral
    chances = np.sum(w*scipy.stats.poisson.pmf(vals, lambda1 * (x + 1) / 2), axis=1)

    if np.any(np.isnan(chances)):
        return np.full(vals.size, np.nan)

    # calculate the probabilities
    probs = (1.0 / scipy.special.beta(alpha, beta)) \
            * (2.0**(-alpha-beta+1)) \
            * chances

    return probs


def beta_poisson3_likelihood(alpha: float, beta: float, lambd: float, vals: np.array) -> np.array:
    """
    Calculate the likelihood for each value in your array of values, based on the beta poisson 3
    model. Calls the beta_poisson4_likelihood with lambda2 as 1, effectively making it a bp3 model.
    """
    assert len(vals.shape) == 1, "vals should be an 1D array"
    return beta_poisson4_likelihood(alpha, beta, lambd, 1.0, vals)


def beta_poisson3_log_likelihood(alpha: float, beta: float, lambd: float,  vals: np.array) -> float:
    """
    Calculate the negative sum of the log likelihood of values for the beta poisson 3 model.
    """
    assert len(vals.shape) == 1, "vals should be an 1D array"
    return -np.sum(np.log(beta_poisson3_likelihood(alpha, beta, lambd, vals) + 1e-10))


def beta_poisson4_log_likelihood(
        alpha: float,
        beta: float,
        lambda1: float,
        lambda2: float,
        vals: np.array
) -> float:
    """
    Calculate the negative sum of the log likelihood of values for the beta poisson 4 model.
    """
    assert len(vals.shape) == 1, "vals should be an 1D array"
    return -np.sum(np.log(beta_poisson4_likelihood(alpha, beta, lambda1, lambda2, vals) + 1e-10))


def beta_poisson_log_likelihood(params: np.array, vals: np.array) -> float:
    """
    Calculate the negative sum of the log likelihood of values for either the beta poisson 3 or beta
    poisson 4 model, dependent on the amount of parameters in params.
    """
    assert len(params) == 3 and type(params) == np.ndarray, "params should be of length 3 and of" \
                                                            "type numpy.array"
    assert len(vals.shape) == 1, "vals should be a 1D array"

    if len(params) not in [3, 4]:
        raise NotImplementedError

    if len(params) == 3:
        return beta_poisson3_log_likelihood(*params, vals)
    return beta_poisson4_log_likelihood(*params, vals)
