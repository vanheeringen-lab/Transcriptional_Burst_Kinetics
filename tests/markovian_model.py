"""
Tests for the markovian model of gene-product synthesis
"""

import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(f"{os.getcwd()}/."))

from tbk.mmgps.run import run_env


class TestMarkov(unittest.TestCase):

    def test_on_off_ratio(self):
        """
        Test whether the measured on-off ratio corresponds to the theoretical ratio.
        """
        # setup parameters
        lambd, mu, nu, delta = np.random.randint(1, 8, 4)
        env, gene = run_env(lambd, mu, nu, delta, time=10000)

        # theoretically we expect...
        expected = lambd / (lambd + mu)

        # we got..
        result = gene.time_on / (gene.time_on + gene.time_off)

        self.assertTrue(abs(expected - result) < 0.05)

    def test_product_lifetime(self):
        """
        Test whether a gene-product takes indeed 1 / delta time to be degraded.
        """
        lambd, mu, nu, delta = np.random.randint(1, 8, 4)
        env, gene = run_env(lambd, mu, nu, delta, time=10000)

        # theoretically we expect...
        expected = 1 / delta

        # we got..
        result = np.mean([product.age for product in gene.products])

        self.assertTrue(abs(expected - result) < 0.02)


    def test_expected(self):
        """
        Test whether the mean of the products corresponds to the theoretical mean.
        """
        # setup parameters
        lambd, mu, nu, delta = 2, 4, 3, 1

        # theoretically we expect..
        expected = (lambd * nu) / ((lambd + mu) * delta)

        # the products we have
        genes = [run_env(lambd, mu, nu, delta)[1] for _ in range(1000)]
        products = [len([product for product in gene.products if not product.degraded]) for gene in genes]

        self.assertTrue(abs(expected - np.mean(products)) < 0.1)


if __name__ == '__main__':
    unittest.main()
