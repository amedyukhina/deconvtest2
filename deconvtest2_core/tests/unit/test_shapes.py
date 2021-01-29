import unittest

import numpy as np
from ddt import ddt, data

from ...shapes.shapes import gaussian, ellipsoid


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


@ddt
class TestEllipsoid(unittest.TestCase):
    @data(
        [20, 10, 10], [2, 2], np.array([2, 2]), np.ones(3),
    )
    def test_dimension(self, x):
        size = len(np.array([x]).flatten())
        self.assertEqual(len(ellipsoid(x).shape), size)

    @data(
        [20, 10, 10], [2, 2], np.ones(3)*5,
    )
    def test_output_volume(self, x):
        volume = np.sum(ellipsoid(x) > 0)
        target_volume = 4./3 * np.pi * np.prod(np.array(x)/2.)
        diff = volume - target_volume
        if abs(diff) < 30:
            diff = 0
        self.assertLess(abs(diff/target_volume), 0.1)

    # def test_rotation(self):
    #     x = ellipsoid([20, 10, 10], phi=0, theta=np.pi / 2)
    #     volume = np.sum(x > 0)
    #     target_volume = 4. / 3 * np.pi * np.prod(np.array([20, 10, 10]) / 2.)
    #     diff = volume - target_volume
    #     if abs(diff) < 30:
    #         diff = 0
    #     print(volume, target_volume, diff)

    # @data(
    #     2, 10,
    # )
    # def test_scale_type(self, x):
    #     self.assertIsInstance(gaussian(sigma=[2], scale=x), np.ndarray)
    #
    # @data(
    #     'a', 'jiaojgw',
    # )
    # def test_invalid_sigma_type(self, x):
    #     self.assertRaises(TypeError, gaussian, sigma=x)
    #
    # @data(
    #     'a', [2, 3], 'jiaojgw', (2, 3), 10., 3.5,
    # )
    # def test_invalid_scale_type(self, x):
    #     self.assertRaises(TypeError, gaussian, sigma=2, scale=x)


if __name__ == '__main__':
    unittest.main()
