import unittest

import numpy as np
from ddt import ddt, data
from deconvtest2.core.utils.conversion import convert_size, unify_shape


@ddt
class TestConversion(unittest.TestCase):
    @data(
        2, [3], [1, 2], np.ones(2), np.ones(3), [1, 2, 3],
    )
    def test_valid_voxel_size(self, x):
        self.assertEqual(len(convert_size(x)), 3)

    @data(
        [], np.ones(10), [1, 2, 3, 4],
    )
    def test_invalid_voxel_size_length(self, x):
        self.assertRaises(ValueError, convert_size, x)

    @data(
        (np.ones([10, 2, 7]), np.zeros([8, 3, 7]), [10, 3, 7]),
        (np.ones([10, 10, 8]), np.zeros([8, 3, 7]), [10, 10, 8]),
    )
    def test_unify_shape(self, case):
        x, y, shape = case
        x, y = unify_shape(x, y)
        self.assertSequenceEqual(list(x.shape), shape)
        self.assertSequenceEqual(list(y.shape), shape)


if __name__ == '__main__':
    unittest.main()
