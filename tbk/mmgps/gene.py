"""
Gene product class for the markovian IAP0 model.
"""
import random

import simpy

from .product import Product


class Gene:
    """
    A gene class, which is either active or inactive.

    Depending on whether or not the gene is active or not, it is transcribing gene products.
    """

    def __init__(
            self,
            env: simpy.core.Environment,
            lambd: float,
            mu: float,
            nu: float,
            delta: float,
            active: bool = False
    ):
        """
        Initialization of the gene.

        :param env:    simpy environment class
        :param lambd:  lambda (gene activation rate)
        :param mu:     mu (gene inactivation rate)
        :param nu:     nu (product synthesis rate)
        :param delta:  delta (product degradation rate)
        :param active: whether or not the gene is active
        """
        # store the args in self
        self.env = env
        self.lambd = lambd  # lambda
        self.mu = mu        # mu
        self.nu = nu        # nu
        self.delta = delta  # delta
        self.active = active

        # Start the run process every time a Gene is created.
        self.running = env.process(self.run())
        self.transcribing = env.process(self.transcribe())
        if not self.active:
            self.transcribing.interrupt()

        # setup variables
        self.time_on = 0
        self.products = []
        self.switches = 0

    def run(self):
        """
        While the environment doesn't interrupt this function, the gene switches between active and
        inactive depending on the the lambda and mu values. When a gene is activated it starts a
        transcribe process, and when a gene is deactivated, it interrupts it.
        """
        while True:
            # switch to on or off
            if self.active:
                self.transcribing = self.env.process(self.transcribe())
            elif self.transcribing.is_alive:
                self.transcribing.interrupt()

            # stay in the on/off state for a certain amount of time
            if self.active:
                time = random.expovariate(self.mu)
                yield self.env.timeout(time)
                self.time_on += time
            else:
                yield self.env.timeout(random.expovariate(self.lambd))

            # now update our state, and keep track of the total amount of switches
            self.switches += 1
            self.active ^= True

    def transcribe(self):
        """
        Generate gene products.
        """
        while True:
            try:
                time = random.expovariate(self.nu)
                yield self.env.timeout(time)
                self.products.append(Product(self.env, self.delta))
            except simpy.Interrupt:
                break

    @property
    def time_off(self):
        """
        Returns the amount of time the gene was in its inactive (off) state.
        """
        return self.env.now - self.time_on
