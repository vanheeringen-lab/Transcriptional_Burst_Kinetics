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

    def test_expected(self):
        """
        Test whether the mean of the products corresponds to the theoretical mean.
        """
        # setup parameters
        lambd, mu, nu, delta = 2, 4, 3, 1

        # theoretically we expect..
        expected = (lambd * nu) / ((lambd + mu) * delta)

        # the products we have
        products = [run_env(lambd, mu, nu, delta) for _ in range(1000)]

        self.assertTrue(abs(expected - np.mean(products)) < 0.1)


if __name__ == '__main__':
    unittest.main()
