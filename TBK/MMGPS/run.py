import simpy
from TBK.MMGPS.gene import Gene


def run_env(lambd: float, mu: float, nu: float, delta=1, time=5000):
    """

    """
    env = simpy.Environment()

    gene = Gene(env, lambd, mu, nu, delta)

    env.run(until=time)

    return len([product for product in gene.products if not product.degraded])
