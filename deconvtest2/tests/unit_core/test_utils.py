import unittest
import warnings
from typing import Union

import numpy as np
from ddt import ddt, data

from ... import core as deconvtest2_core
from ...core.utils.utils import check_type, list_modules, modules_to_json, is_valid_type


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
        (2, int),
        (3.5, Union[float, int])
    )
    def test_valid_type(self, case):
        variable, valid_type = case
        self.assertTrue(is_valid_type(variable, valid_type))

    @data(
        (2, str),
        ('3.5', Union[float, int])
    )
    def test_valid_type(self, case):
        variable, valid_type = case
        self.assertFalse(is_valid_type(variable, valid_type))

    @data(
        (['var1', 'var2', 'var3'], [3, [2, 4.5], 'aiosghfr'], [int, [int, list], float]),
        (['var1'], ['s'], [[int, float]]),
    )
    def test_wrong_type_check(self, case):
        names, variables, types = case
        self.assertRaises(TypeError, check_type, names, variables, types)

    def test_list_modules(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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
