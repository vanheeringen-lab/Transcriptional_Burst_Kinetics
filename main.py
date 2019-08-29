import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import simpy
from TBK.MMGPS.run import run_env
from TBK.inference import moment_based, maximum_likelihood
from TBK.BP.bp import beta_poisson3_likelihood, beta_poisson3, beta_poisson4
import numpy as np


if __name__ == "__main__":
    lambd = 2  # rate from inactive -> active
    mu = 0.5  # rate from active -> inactive
    nu = 4  # rate from active -> product
    delta = 1  # rate from product -> degraded

    # generate products through a markovian model
    # products = np.array(Parallel(n_jobs=50)(delayed(run_env)(lambd, mu, nu, delta) for i in range(1000)))

    # generate products through the beta poisson model
    products = beta_poisson3(lambd, mu, nu, size=1000)

    print(f'The parameters are: lambda {lambd}, mu {mu}, nu {nu}, delta {delta}')
    print(f'the parameters based on moment inference is: {moment_based(np.array(products))}')
    print(f'the parameters based on ML:                  {maximum_likelihood(np.array(products), model="BP3")}')
    print(f"The average of gene-products {np.mean(products)},")
    print(f"And theoretically we expect: {((lambd * nu) / ((lambd + mu) * delta))}")
