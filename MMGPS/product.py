import random
import simpy


class Product:
    """
    Gene product class.

    When initialized it starts it starts degradation with rate delta (often all parameters are divided by delta, so that
    all times are relative to the half-time of the product).
    """

    def __init__(self, env: simpy.core.Environment, de: float):
        """
        Initialization of the product.

        :param env: simpy environment class
        :param de:  delta (the rate of product degradation)
        """
        self.env = env
        self.de = de
        self.start = self.env.now
        self.end = None
        self.process = env.process(self.degradadation())

    def degradadation(self):
        """
        Degrade after 1/delta time on average.
        """
        t = random.expovariate(self.de)
        yield self.env.timeout(t)
        self.end = self.env.now

    @property
    def age(self):
        if not self.degraded:
            return self.env.now - self.start
        return self.end - self.start

    @property
    def degraded(self):
        return self.end is not None
