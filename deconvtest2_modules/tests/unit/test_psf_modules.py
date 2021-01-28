import unittest

import numpy as np
from ddt import ddt, data

from ...psf.psf_modules import gaussian


@ddt
class TestGaussian(unittest.TestCase):
    @data(
        2, 5.5, 3.
    )
    def test_sigma_aspect_type(self, x):
        self.assertIsInstance(gaussian(sigma=x, aspect=x), np.ndarray)

    @data(
        2, 10,
    )
    def test_scale_type(self, x):
        self.assertIsInstance(gaussian(sigma=2, scale=x), np.ndarray)

    @data(
        'a', [2, 3], 'jiaojgw', (2, 3),
    )
    def test_invalid_sigma_aspect_type(self, x):
        self.assertRaises(TypeError, gaussian, sigma=x)

    @data(
        'a', [2, 3], 'jiaojgw', (2, 3), 10., 3.5,
    )
    def test_invalid_scale_type(self, x):
        self.assertRaises(TypeError, gaussian, sigma=2, scale=x)


if __name__ == '__main__':
    unittest.main()
