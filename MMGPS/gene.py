import simpy
import random
from product import Product


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

        # setup variables
        self.time_on = self.time_of = 0
        self.products = []
        self.switches = 0

    def run(self):
        """

        """
        while True:
            if self.active:
                # self.transcribing = self.env.process(self.run())
                time_on = random.expovariate(self.la)
                yield self.env.timeout(time_on)
                self.time_on += time_on
            else:
                # self.transcribing.interrupt()
                # print(self.transcribing)
                time_of = random.expovariate(self.mu)
                yield self.env.timeout(time_of)
                self.time_of += time_of

            # flip to active / inactive
            self.switches += 1
            self.active ^= True

    def transcribe(self):
        """

        """
        while True:
            try:
                yield self.env.timeout(random.expovariate(self.nu))
                self.products.append(Product(self.env, self.de))
            except simpy.Interrupt:
                break
