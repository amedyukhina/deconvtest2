import unittest

import numpy as np
from ddt import ddt, data

from ...utils.conversion import convert_size


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


if __name__ == '__main__':
    unittest.main()
