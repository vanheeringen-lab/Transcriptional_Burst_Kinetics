import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(f"{os.getcwd()}/."))

from TBK.BP import bp


class TestBetaPoisson(unittest.TestCase):
    """
    """
    def test_beta_poisson_equal(self):
        """
        """
        for i in range(10):
            np.random.seed(i)
            bp3 = bp.beta_poisson3(1, 2, 3)
            np.random.seed(i)
            bp4 = bp.beta_poisson4(1, 1, 2, 3)
            self.assertEqual(bp3, bp4)


if __name__ == '__main__':
    unittest.main()
