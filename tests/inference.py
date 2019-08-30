"""
Tests for the inference of parameters
"""

import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(f"{os.getcwd()}/."))

from tbk.inference import moment_based, maximum_likelihood
from tbk.bp.bp import beta_poisson3



class TestInference(unittest.TestCase):

    def test_moment_based_inference(self):
        # TODO: check if these make sense
        values = np.array([0, 1, 2, 3, 4])
        self.assertTrue(np.allclose(moment_based(values), np.array([-2, 0, 2])))

    def test_ML3(self):
        np.random.seed(42)
        params = np.array([2.32735786, 0.25476861, 7.44452277])
        self.assertTrue(np.allclose(params, maximum_likelihood(beta_poisson3(*params, 500)), 0.5))


if __name__ == '__main__':
    unittest.main()
