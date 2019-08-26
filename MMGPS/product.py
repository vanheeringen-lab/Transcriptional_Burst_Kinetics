import random


class Product:
    """

    """
    def __init__(self, env, de):
        self.env = env
        self.de = de
        self.start = self.end = None
        self.process = env.process(self.degradadation())

    def degradadation(self):
        """

        """
        self.start = self.env.now
        yield self.env.timeout(random.expovariate(self.de))
        self.end = self.env.now

    @property
    def time(self):
        if self.end is None:
            return None
        return self.end - self.start

    @property
    def degraded(self):
        return self.end is None
