from numba import jit
import numpy as np


@jit()
def moment_inference(vals: np.array):
    """
    Estimate parameters lambda, mu, and nu based on their first three moments.
    Based on the paper: Markovian Modelling of Gene Product Synthesis
    """

    # calculate the moments (27)
    m1 = np.sum(vals) / len(vals)
    m2 = np.sum(vals * (vals - 1)) / len(vals)
    m3 = np.sum(vals * (vals - 1) * (vals - 2)) / len(vals)

    if 0 in [m1, m2]:
        return np.array([np.nan, np.nan, np.nan])

    r1 = m1
    r2 = m2/m1
    r3 = m3/m2

    la_denom = (r1*r2 - 2*r1*r3 + r2*r3)
    nu_denom = (r1 - 2*r2 + r3)

    if 0 in [la_denom, nu_denom]:
        return np.array([np.nan, np.nan, np.nan])

    # 26
    la_est = (2 * r1 * (r3 - r2)) / la_denom
    mu_est = (2 * (r3 - r2) * (r1 - r3) * (r2 - r1)) / (la_denom * nu_denom)
    nu_est = (2 * r1 * r3 - r1 * r2 - r2 * r3) / nu_denom

    return np.array([la_est, mu_est, nu_est])