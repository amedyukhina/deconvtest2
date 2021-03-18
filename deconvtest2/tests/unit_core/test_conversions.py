import unittest

import numpy as np
from ddt import ddt, data

from ...core.utils.conversion import convert_size, unify_shape, list_to_keys, keys_to_list


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

    def test_dict_conversion(self):
        params = dict({'x': 3,
                       'size': [2, 3, 5]})
        params_converted = keys_to_list(list_to_keys(params))
        for key in params.keys():
            if type(params[key]) is list:
                self.assertSequenceEqual(params[key], params_converted[key])
            else:
                self.assertEqual(params[key], params_converted[key])


if __name__ == '__main__':
    unittest.main()
