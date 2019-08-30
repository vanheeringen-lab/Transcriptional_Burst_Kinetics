"""
Tests for the inference of parameters
"""

import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(f"{os.getcwd()}/."))

from tbk.inference import moment_based


class TestInference(unittest.TestCase):

    def test_moment_based_inference(self):
        values = np.array([6, 4, 8, 9, 2, 5, 4, 8, 5, 4, 8, 8, 4, 4, 12, 3, 10, 1, 7, 6, 11, 2, 5,
                           6, 6, 8, 7, 8, 7, 4, 11, 8, 7, 6, 1, 6, 7, 8, 7, 5, 6, 12, 9, 7, 4, 7, 8,
                           9, 7, 8, 3, 8, 11, 10, 12, 4, 3, 2, 7, 3, 4, 9, 9, 9, 8, 2, 4, 7, 3, 7,
                           4, 4, 6, 8, 6, 5, 5, 8, 8, 5, 8, 7, 16, 8, 6, 4, 11, 7, 8, 10, 3, 10, 5,
                           7, 13, 8, 12, 2, 8, 9])


        params = moment_based(values)
        print(params)
        self.assertTrue(np.allclose(params, np.array([2.32735786, 0.25476861, 7.44452277])))


if __name__ == '__main__':
    unittest.main()
