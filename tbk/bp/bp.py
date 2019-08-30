"""
Collection of beta poisson (3 and 4) functions
"""
import numpy as np
import scipy.special
import scipy.stats


def beta_poisson3(alpha: float, beta: float, lambd: float, size: int = 1) -> np.array:
    """
    Generate 'random' data from the beta poisson 3 model.
    """
    return np.random.poisson(lambd * np.random.beta(alpha, beta, size))


def beta_poisson4(lambda1: float, lambda2: float, alpha: float, beta: float, size: int = 1) \
        -> np.array:
    """
    Generate 'random' data from the beta poisson 4 model.
    """
    return lambda2 * beta_poisson3(lambda1, alpha, beta, size)


def beta_poisson4_likelihood(
        lambda1: float,
        lambda2: float,
        alpha: float,
        beta: float,
        _vals: np.array
) -> np.array:
    """
    Calculate the likelihood for each value in your array of values, based on the beta poisson 4
    model.
    """
    # scale to lambda2
    vals = np.copy(_vals) / lambda2

    # get the sample points and weights
    x, w = scipy.special.j_roots(50, alpha=beta - 1, beta=alpha - 1)

    # estimate the integral
    gs = np.sum(w*scipy.stats.poisson.pmf(vals, lambda1 * (x + 1) / 2), axis=1)

    # calculate the probabilities
    probs = (1.0 / scipy.special.beta(alpha, beta)) * \
            (2.0**(-alpha-beta+1)) * \
            gs

    return probs


def beta_poisson3_likelihood(lambda1: float, alpha: float, beta: float, vals: np.array) -> np.array:
    """
    Calculate the likelihood for each value in your array of values, based on the beta poisson 3
    model. Calls the beta_poisson4_likelihood with lambda2 as 1, effectively making it a bp3 model.
    """
    return beta_poisson4_likelihood(lambda1, 1.0, alpha, beta, vals)


def beta_poisson3_log_likelihood(lambd: float, alpha: float, beta: float, vals: np.array) -> float:
    """
    Calculate the negative sum of the log likelihood of values for the beta poisson 3 model.
    """
    return -np.sum(np.log(beta_poisson3_likelihood(lambd, alpha, beta, vals) + 1e-10))


def beta_poisson4_log_likelihood(
        lambda1: float,
        lambda2: float,
        alpha: float,
        beta: float,
        vals: np.array
) -> float:
    """
    Calculate the negative sum of the log likelihood of values for the beta poisson 4 model.
    """
    return -np.sum(np.log(beta_poisson4_likelihood(lambda1, lambda2, alpha, beta, vals) + 1e-10))


def beta_poisson_log_likelihood(params: tuple, vals: np.array) -> float:
    """
    Calculate the negative sum of the log likelihood of values for either the beta poisson 3 or beta
    poisson 4 model, dependent on the amount of parameters in params.
    """
    if len(params) not in [3, 4]:
        raise NotImplementedError

    if len(params) == 3:
        return beta_poisson3_log_likelihood(*params, vals)
    return beta_poisson4_log_likelihood(*params, vals)
