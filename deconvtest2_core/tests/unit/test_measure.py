import unittest

import numpy as np
from ddt import ddt, data

from ...utils.measure import bounding_box


@ddt
class TestMeasure(unittest.TestCase):

    def test_bbox(self):
        indmin, indmax = bounding_box(np.ones([10, 10, 10]))
        self.assertSequenceEqual(list(indmin), [0]*3, seq_type=list)
        self.assertSequenceEqual(list(indmax), [9]*3, seq_type=list)

    def test_bbox2(self):
        indmin, indmax = bounding_box(np.zeros([10, 10]))
        self.assertSequenceEqual(list(indmin), [None]*2, seq_type=list)
        self.assertSequenceEqual(list(indmax), [None]*2, seq_type=list)


if __name__ == '__main__':
    unittest.main()
