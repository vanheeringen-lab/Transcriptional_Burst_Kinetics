import numpy as np
import scipy.special
import scipy.stats


def beta_poisson3(alpha: float, beta: float, lambd: float, size: int = 1) -> np.array:
    """

    """
    return np.random.poisson(lambd * np.random.beta(alpha, beta, size))


def beta_poisson4(lambda1: float, lambda2: float, alpha: float, beta: float, size: int = 1) -> np.array:
    """

    """
    return lambda2 * beta_poisson3(lambda1, alpha, beta, size)


def beta_poisson4_likelihood(lambda1: float, lambda2: float, alpha: float, beta: float, _vals: np.array) -> np.array:
    """

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

    """
    return beta_poisson4_likelihood(lambda1, 1.0, alpha, beta, vals)


def beta_poisson3_log_likelihood(lambd: float, alpha: float, beta: float, vals: np.array) -> float:
    """

    """
    return -np.sum(np.log(beta_poisson3_likelihood(lambd, alpha, beta, vals) + 1e-10))


def beta_poisson4_log_likelihood(lambda1: float, lambda2: float, alpha: float, beta: float, vals: np.array) -> float:
    """

    """
    return -np.sum(np.log(beta_poisson4_likelihood(lambda1, lambda2, alpha, beta, vals) + 1e-10))


def beta_poisson_log_likelihood(params: tuple, vals: np.array) -> float:
    """

    """
    if len(params) == 3:
        return beta_poisson3_log_likelihood(*params, vals)
    elif len(params) == 4:
        return beta_poisson4_log_likelihood(*params, vals)
    else:
        raise NotImplementedError
