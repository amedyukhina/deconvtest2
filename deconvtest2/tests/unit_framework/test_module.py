import unittest

import numpy as np
from ddt import ddt, data
from deconvtest2.framework.module.module import Module
from deconvtest2.framework.module.deconvolution_module import DeconvolutionModule


@ddt
class TestModuleImport(unittest.TestCase):

    @data(
        'gaussian',
        'ellipsoid'
    )
    def test_method(self, method):
        module = Module(method)
        self.assertIsNotNone(module.method)

    @data(
        'regularized_inverse_filter',
    )
    def test_deconv_method(self, method):
        module = DeconvolutionModule(method)
        self.assertIsNotNone(module.method)

    @data(
        'fake_method1',
        'fake_method2'
    )
    def test_method_err(self, method):
        self.assertRaises(ValueError, Module, method)

    @data(
        'fake_method',
    )
    def test_deconv_method_err(self, method):
        self.assertRaises(ValueError, DeconvolutionModule, method)


if __name__ == '__main__':
    unittest.main()
