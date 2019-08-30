"""
Function that runs an environment with a gene collecting products.
"""
import simpy

from tbk.mmgps.gene import Gene


def run_env(lambd: float, mu: float, nu: float, delta=1, time=5000):
    """
    Run an environment with one gene for a certain amount of time.
    """
    env = simpy.Environment()

    gene = Gene(env, lambd, mu, nu, delta)

    env.run(until=time)

    return len([product for product in gene.products if not product.degraded])
