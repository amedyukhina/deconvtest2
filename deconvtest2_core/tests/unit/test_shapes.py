import unittest

import numpy as np
from ddt import ddt, data

from ...shapes.shapes import gaussian


@ddt
class TestGaussian(unittest.TestCase):
    @data(
        2, [2], [2, 2], np.array([2, 2]), np.ones(3),
    )
    def test_dimension(self, x):
        size = len(np.array([x]).flatten())
        self.assertEqual(len(gaussian(x).shape), size)

    @data(
        2, 10,
    )
    def test_scale_type(self, x):
        self.assertIsInstance(gaussian(sigma=[2], scale=x), np.ndarray)

    @data(
        'a', 'jiaojgw',
    )
    def test_invalid_sigma_type(self, x):
        self.assertRaises(TypeError, gaussian, sigma=x)

    @data(
        'a', [2, 3], 'jiaojgw', (2, 3), 10., 3.5,
    )
    def test_invalid_scale_type(self, x):
        self.assertRaises(TypeError, gaussian, sigma=2, scale=x)


if __name__ == '__main__':
    unittest.main()
