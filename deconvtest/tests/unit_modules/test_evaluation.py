import unittest

import numpy as np
from ddt import ddt, data
from skimage.metrics import normalized_root_mse

from ...modules.evaluation.evaluation_modules import rmse, nrmse


@ddt
class TestRMSE(unittest.TestCase):

    @data(
        (np.ones([10, 10, 10]), np.ones([10, 10, 10]), 0),
        (np.ones([10, 10, 10]), np.zeros([10, 10, 10]), 1),
        (np.ones([10, 15, 10]), np.zeros([20, 10, 10]), 1),

    )
    def test_rmse(self, case):
        img1, img2, target_err = case
        err = rmse(img1, img2)
        self.assertEqual(err, target_err)

    def test_against_skimage(self):
        img1 = np.random.randint(0, 100, [100, 100]) * 1.
        img2 = np.random.randint(0, 100, [100, 100]) * 1.
        self.assertEqual(normalized_root_mse(img1, img2, normalization='min-max'), nrmse(img1, img2))


if __name__ == '__main__':
    unittest.main()
