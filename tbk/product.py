"""
Gene product class for the markovian IAP0 model.
"""
import random
import simpy


class Product:
    """
    Gene product class.

    When initialized it starts it starts degradation with rate delta (often all parameters are
    divided by delta, so that all times are relative to the half-time of the product).
    """

    def __init__(self, env: simpy.core.Environment, delta: float):
        """
        Initialization of the product.

        :param env: simpy environment class
        :param de:  delta (the rate of product degradation)
        """
        self.env = env
        self.delta = delta
        self.start = self.env.now
        self.end = None
        self.process = env.process(self.degradation())

    def degradation(self):
        """
        Degrade after 1/delta time on average.
        """
        time = random.expovariate(self.delta)
        yield self.env.timeout(time)
        self.end = self.env.now

    @property
    def age(self):
        """
        Returns how long the product was/is alive.
        """
        if not self.degraded:
            return self.env.now - self.start
        return self.end - self.start

    @property
    def degraded(self):
        """
        Returns whether or not the product has degraded.
        """
        return self.end is not None
