import unittest

from ddt import ddt, data
import numpy as np

from ...utils.utils import check_type, list_modules, modules_to_json
import deconvtest2_core


@ddt
class TestUtils(unittest.TestCase):

    @data(
        (['var1', 'var2', 'var3'], [3, [2, 4.5], 'aiosghfr'], [int, [int, list], str]),
        (['var1'], ['s'], [[str, float]]),
        (['var1'], [np.ones(10)], [[list, np.ndarray]]),
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

    def test_list_modules(self):
        modules = list_modules(deconvtest2_core)
        modules = [module[0].__name__ for module in modules]
        self.assertIn('__list_modules', modules)
        self.assertIn('list_modules', modules)

    def test_modules_to_json(self):
        modules = modules_to_json(list_modules(deconvtest2_core))
        modules = [module['name'] for module in modules]
        self.assertIn('__list_modules', modules)
        self.assertIn('list_modules', modules)


if __name__ == '__main__':
    unittest.main()
