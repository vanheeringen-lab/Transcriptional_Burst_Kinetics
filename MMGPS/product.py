import random


class Product:
    """

    """
    def __init__(self, env, de):
        self.env = env
        self.de = de
        self.start = self.env.now
        self.end = None
        self.process = env.process(self.degradadation())

    def degradadation(self):
        """

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
