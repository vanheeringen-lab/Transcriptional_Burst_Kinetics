"""
Function that runs an environment with a gene collecting products.
"""
from typing import Tuple

import simpy

from tbk.mmgps.gene import Gene


def run_env(lambd: float, mu: float, nu: float, delta=1, time=5000)\
        -> Tuple[simpy.Environment, Gene]:
    """
    Run an environment with one gene for a certain amount of time.
    """
    env = simpy.Environment()

    gene = Gene(env, lambd, mu, nu, delta)

    env.run(until=time)

    return env, gene
