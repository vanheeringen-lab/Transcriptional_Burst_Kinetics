import simpy
import random
from .product import Product


class Gene:
    """
    A gene class, which is either active or inactive.

    Depending on whether or not the gene is active or not, it is transcribing gene products.
    """

    def __init__(self, env: simpy.core.Environment, la: float, mu: float, nu: float, de: float, active: bool = False):
        """
        Initialization of the gene.

        :param env:    simpy environment class
        :param la:     lambda (gene activation rate)
        :param mu:     mu (gene inactivation rate)
        :param nu:     nu (product synthesis rate)
        :param de:     delta (product degradation rate)
        :param active: whether or not the gene is active
        """
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
        While the environment doesn't interrupt this function, the gene switches between active and inactive depending
        on the the lambda and mu values. When a gene is activated it starts a transcribe process, and when a gene
        is deactivated, it interrupts it.
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
        Generate gene products.
        """
        while True:
            try:
                t = random.expovariate(self.nu)
                yield self.env.timeout(t)
                self.products.append(Product(self.env, self.de))
            except simpy.Interrupt:
                break
