import unittest

import numpy as np
from ddt import ddt, data

from ...transforms.transform_modules import convolve

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


if __name__ == '__main__':
    unittest.main()
