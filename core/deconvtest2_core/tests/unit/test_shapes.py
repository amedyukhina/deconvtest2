import unittest

import numpy as np
from ddt import ddt, data

from ...shapes.shapes import gaussian, ellipsoid
from ...utils.measure import bounding_box


def sum_of_border_pixels(arr, margin=3):
    border_pix = 0
    for i in range(len(arr.shape)):
        slc = [slice(None)] * len(arr.shape)
        slc[i] = slice(-margin, None)
        border_pix += np.sum(arr[tuple(slc)])
        slc = [slice(None)] * len(arr.shape)
        slc[i] = slice(0, margin)
        border_pix += np.sum(arr[tuple(slc)])
    return border_pix


@ddt
class TestGaussian(unittest.TestCase):
    @data(
        2, [2], [2, 2], np.array([2, 2]), np.ones(3),
    )
    def test_dimension(self, x):
        size = len(np.array([x]).flatten())
        self.assertEqual(len(gaussian(x).shape), size)


@ddt
class TestEllipsoid(unittest.TestCase):
    @data(
        [20, 10, 10], [2, 2], np.array([2, 2]), np.ones(3),
    )
    def test_dimension(self, x):
        size = len(np.array([x]).flatten())
        self.assertEqual(len(ellipsoid(x).shape), size)

    @data(
        [20, 10, 10],
        [20, 11, 11],
        [50, 21, 21],
        [41, 20, 20],
        [2, 2],
        np.ones(3) * 5,
        [33]*3,
    )
    def test_sizes(self, x):
        ell = ellipsoid(x)
        volume = np.sum(ell > 0)
        target_volume = 4. / 3 * np.pi * np.prod(np.array(x) / 2.)
        diff = volume - target_volume
        if abs(diff) < 30:
            diff = 0
        self.assertLess(abs(diff / target_volume), 0.1)
        self.assertEqual(sum_of_border_pixels(ell, margin=1), 0)

    @data(
        (0, np.pi / 2),
        (np.pi / 2, 0),
        (np.pi / 4, np.pi / 4),
        (np.pi, np.pi * 3 / 4),
    )
    def test_rotation(self, x):
        phi, theta = x
        size = np.array([20, 10, 10])
        ell = ellipsoid(size, phi, theta)
        volume = np.sum(ell > 0)
        target_volume = 4. / 3 * np.pi * np.prod(size / 2.)
        diff = volume - target_volume
        if abs(diff) < 30:
            diff = 0
        self.assertLess(abs(diff / target_volume), 0.1)
        self.assertEqual(sum_of_border_pixels(ell, margin=3), 0)

    @data(
        1, 2, 3, 4, 5,
    )
    def test_margin(self, target_margin):
        ell = ellipsoid([21, 11, 11], np.pi / 4, 0, margin=target_margin)
        indmin, indmax = bounding_box(ell)
        output_margin = (np.array(ell.shape) - (indmax - indmin + 1)) / 2
        for i in range(len(output_margin)):
            self.assertGreaterEqual(target_margin, output_margin[i])


if __name__ == '__main__':
    unittest.main()
