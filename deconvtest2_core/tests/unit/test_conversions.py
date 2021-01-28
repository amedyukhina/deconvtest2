import unittest

import numpy as np
from ddt import ddt, data

from ...utils.conversion import convert_voxel_size

@ddt
class TestConversion(unittest.TestCase):
    @data(
        2, [3], [1, 2], np.ones(2), np.ones(3), [1, 2, 3],
    )
    def test_valid_voxel_size(self, x):
        self.assertEqual(len(convert_voxel_size(x)), 3)

    @data(
        'test', 's',
    )
    def test_invalid_voxe_size_type(self, x):
        self.assertRaises(TypeError, convert_voxel_size, x)

    @data(
        [], np.ones(10), [1,2,3,4],
    )
    def test_invalid_voxel_size_length(self, x):
        self.assertRaises(ValueError, convert_voxel_size, x)

if __name__ == '__main__':
    unittest.main()
