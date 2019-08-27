import simpy
import random
from product import Product
import numpy as np


class Gene:
    """

    """
    def __init__(self, env, la, mu, nu, de, active=False):
        # store the args in self
        self.env = env
        self.la = la  # lambda
        self.mu = mu  # mu
        self.nu = nu  # nu
        self.de = de  # delta
        self.active = active

        # Start the run process every time a Gene is created.
        self.running = env.process(self.run())
        self.transcribing = env.process(self.transcribe())
        if not self.active:
            self.transcribing.interrupt()

        # setup variables
        self.time_on = self.time_of = 0
        self.products = []
        self.switches = 0

    def run(self):
        """

        """
        while True:
            if self.active:
                self.transcribing = self.env.process(self.transcribe())
                t = random.expovariate(self.mu)
                yield self.env.timeout(t)
                self.time_on += t
            else:
                if self.transcribing.is_alive:
                    self.transcribing.interrupt()
                t = random.expovariate(self.la)
                yield self.env.timeout(t)
                self.time_of += t

            # flip to active / inactive
            self.switches += 1
            self.active ^= True

    def transcribe(self):
        """

        """
        while True:
            try:
                t = random.expovariate(self.nu)
                yield self.env.timeout(t)
                self.products.append(Product(self.env, self.de))
            except simpy.Interrupt:
                break
