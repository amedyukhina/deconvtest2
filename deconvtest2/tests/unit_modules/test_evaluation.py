import unittest

import numpy as np
from ddt import ddt, data
from deconvtest2.modules.evaluation.evaluation_modules import rmse


@ddt
class TestRMSE(unittest.TestCase):

    @data(
        (np.ones([10, 10, 10]), np.ones([10, 10, 10]), 0),
        (np.ones([10, 10, 10]), np.zeros([10, 10, 10]), 1),

    )
    def test_rmse(self, case):
        img1, img2, target_err = case
        err = rmse(img1, img2)
        self.assertEqual(err, target_err)


if __name__ == '__main__':
    unittest.main()
