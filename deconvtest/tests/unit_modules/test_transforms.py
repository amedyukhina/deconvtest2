import unittest

import numpy as np
from ddt import ddt, data

from ...modules.transforms.transform_modules import convolve, poisson_noise


@ddt
class TestConvolve(unittest.TestCase):

    @data(
        [np.ones([10, 10, 10]), np.ones([3, 3, 4])],
        [np.ones([10, 10]), np.ones([3, 4])],
        [np.ones([10]), np.ones([3])],
    )
    def test_dimensions(self, case):
        img, psf = case
        self.assertEqual(len(convolve(img, psf).shape), len(img.shape))

    @data(
        [np.ones([10, 10, 10]), np.ones([3, 4])],
        [np.ones([10, 10, 4]), np.ones([3, 4])],
        [np.ones([10, 2]), np.ones([3])],
    )
    def test_error(self, case):
        img, psf = case
        self.assertRaises(ValueError, convolve, img, psf)


@ddt
class TestPoisson(unittest.TestCase):

    @data(
        [np.ones([10, 10, 10]), 3],
        [np.ones([10, 10]), 10],
    )
    def test_dimensions(self, case):
        img, snr = case
        self.assertEqual(len(poisson_noise(img, snr).shape), len(img.shape))

    @data(
        [np.ones([10, 10, 10]), 3],
        [np.ones([10, 10, 10]), 1],
        [np.ones([10, 10, 10]), 2],
        [np.ones([10, 10, 10]), 15],
        [np.ones([10, 10]), 10],
    )
    def test_snr(self, case):
        img, snr = case
        out = poisson_noise(img, snr)
        out_snr = img.max() / np.std(out[np.where(img > 0)])
        self.assertGreaterEqual(0.1, abs(out_snr - snr) / snr)


if __name__ == '__main__':
    unittest.main()
