"""
Tests for the beta poisson module
"""

import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(f"{os.getcwd()}/."))

from tbk import bp


class TestBetaPoisson(unittest.TestCase):

    def test_beta_poisson_equal(self):
        """
        Test whether beta poisson 4 is equal to beta poisson 3 with lambda2 set to 1.0
        """
        for i in range(10):
            np.random.seed(i)
            bp3 = bp.beta_poisson3(2, 3, 1)
            np.random.seed(i)
            bp4 = bp.beta_poisson4(2, 3, 1, 1)
            self.assertEqual(bp3, bp4)


if __name__ == '__main__':
    unittest.main()