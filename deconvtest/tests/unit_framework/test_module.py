import unittest
import warnings

import numpy as np
from ddt import ddt, data

from ...framework.module.module import Module
from ...framework.step.deconvolution import Deconvolution
from ...framework.step.evaluation import Evaluation
from ...framework.step.ground_truth import GroundTruth
from ...framework.step.psf import PSF
from ...framework.step.transform import Transform
from ...framework.step.convolution import Convolution


@ddt
class TestModuleImport(unittest.TestCase):

    @data(
        'gaussian',
        'ellipsoid'
    )
    def test_method(self, method):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            module = Module(method)
        self.assertIsNotNone(module.method)

    @data(
        'regularized_inverse_filter',
    )
    def test_deconvolution_method(self, method):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            module = Deconvolution(method)
        self.assertIsNotNone(module.method)

    @data(
        'gaussian',
    )
    def test_psf_method(self, method):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            module = PSF(method)
        self.assertIsNotNone(module.method)

    @data(
        'ellipsoid',
    )
    def test_gt_method(self, method):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            module = GroundTruth(method)
        self.assertIsNotNone(module.method)

    @data(
        'convolve',
    )
    def test_transform_method(self, method):
        module = Convolution(method)
        self.assertIsNotNone(module.method)

    @data(
        'rmse',
    )
    def test_accuracy_method(self, method):
        module = Evaluation(method)
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
    def test_deconvolution_method_err(self, method):
        self.assertRaises(ValueError, Deconvolution, method)

    def test_module_run(self):
        m = Module('ellipsoid')
        self.assertIsInstance(m.run(size=5.), np.ndarray)


if __name__ == '__main__':
    unittest.main()
