import unittest

from ddt import ddt, data

from ...utils.utils import check_type


@ddt
class TestUtils(unittest.TestCase):

    @data(
        (['var1', 'var2', 'var3'], [3, [2, 4.5], 'aiosghfr'], [int, [int, list], str]),
        (['var1'], ['s'], [[str, float]]),
    )
    def test_type_check(self, case):
        names, variables, types = case
        check_type(names, variables, types)

    @data(
        (['var1', 'var2', 'var3'], [3, [2, 4.5], 'aiosghfr'], [int, [int, list], float]),
        (['var1'], ['s'], [[int, float]]),
    )
    def test_wrong_type_check(self, case):
        names, variables, types = case
        self.assertRaises(TypeError, check_type, names, variables, types)


if __name__ == '__main__':
    unittest.main()
