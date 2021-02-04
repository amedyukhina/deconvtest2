import unittest

from ...list_modules import list_package_contents
import deconvtest2_module_creator


class TestListModules(unittest.TestCase):

    def test_list_modules(self):
        modules = list_package_contents(deconvtest2_module_creator)
        modules = [module[0].__name__ for module in modules]
        self.assertIn('__list_modules', modules)
        self.assertIn('list_package_contents', modules)


if __name__ == '__main__':
    unittest.main()
