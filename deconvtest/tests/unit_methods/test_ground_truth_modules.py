import unittest

import numpy as np
from ddt import ddt, data

from ...methods.ground_truth.ellipsoid import ellipsoid


@ddt
class TestEllipsoid(unittest.TestCase):

    @data(
        ['s', 3, 0, 0],
        [[1, 1, 1], '3', 0, 0],
        [1, 3, [0, 3], 0],
        [1, 3, '[0, 3]', 0],
        [1, 3, 0, [0, 3]],
        [1, 3, 0, 's'],
    )
    def test_types(self, variables):
        self.assertRaises(TypeError, ellipsoid, *variables)

    @data(
        [1, 0.1, 0, 0],
        [3, [0.2, 0.1, 0.1], np.pi / 4, np.pi],
        [[5, 3, 2], 0.1, 0, 0],
    )
    def test_valid_inputs(self, variables):
        ellipsoid(*variables)


if __name__ == '__main__':
    unittest.main()
