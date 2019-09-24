import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

sys.path.append(os.path.abspath(f"{os.getcwd()}/."))
from tbk.bp import beta_poisson3
from tbk.inference import maximum_likelihood
from tbk.confidence_interval import confidence_intervals


def get_estimation_and_confidence(lambd, mu, nu, repeats=5):
    """

    """
    reps = []
    for repeat in range(repeats):
        vals = beta_poisson3(lambd, mu, nu, size=100)

        estimation = maximum_likelihood(vals)
        confidence_freq, confidence_size = confidence_intervals(estimation, vals)

        freq = (confidence_freq[2] - confidence_freq[1]) / confidence_freq[0]
        size = (confidence_size[2] - confidence_size[1]) / confidence_size[0]

        mean = size
        reps.append(mean)

    return reps


def get_diffs_and_conf(freq_range, size_range):
    nu = 100
    lambdas, burst_sizes = np.mgrid[freq_range, size_range]
    lambdas = 10. ** lambdas
    burst_sizes = 10. ** burst_sizes

    mus = nu / burst_sizes

    with mp.Pool(processes=40) as pool:
        diffs = pool.starmap(get_estimation_and_confidence,
                             [(lambd, mu, nu) for lambd, mu in zip(lambdas.flatten(), mus.flatten())])


    for diff in diffs:
        print(diff, np.nanmean(diff) if 0 < np.nanmean(diff) < 1000 else np.nan)

    means = [np.nanmean(diff) if 0 < np.nanmean(diff) < 1000 else np.nan for diff in diffs]
    means = np.array(means).reshape(lambdas.shape)
    return means, burst_sizes[0], lambdas[:, 0]

tmp, x_vals, y_vals = get_diffs_and_conf(slice(-3, 1.01, 0.5), slice(0, 3.01, 0.5))

ax = plt.imshow(tmp, origin='lower')
plt.yticks(np.arange(*y_vals.shape), np.round(y_vals, 3))
plt.xticks(np.arange(*x_vals.shape), np.round(x_vals, 3))

plt.show()
