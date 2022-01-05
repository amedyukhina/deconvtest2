import unittest

import numpy as np
from ddt import ddt, data
from skimage.metrics import normalized_root_mse, structural_similarity, peak_signal_noise_ratio

from ...methods.evaluation.nrmse import nrmse
from ...methods.evaluation.rmse import rmse
from ...methods.evaluation.psnr import psnr
from ...methods.evaluation.ssim import ssim


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
        self.assertAlmostEqual(ssim(img1, img2), 1 - target_err, 2)

    def test_against_skimage(self):
        img1 = np.random.randint(0, 100, [100, 100]) * 1.
        img2 = np.random.randint(0, 100, [100, 100]) * 1.
        self.assertEqual(normalized_root_mse(img1, img2, normalization='min-max'), nrmse(img1, img2))
        self.assertEqual(structural_similarity(img1, img2, full=False), ssim(img1, img2))
        self.assertEqual(peak_signal_noise_ratio(img1, img2, data_range=np.max(img1) - np.min(img2)),
                         psnr(img1, img2))


if __name__ == '__main__':
    unittest.main()
