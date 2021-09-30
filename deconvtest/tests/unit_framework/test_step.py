import os
import unittest
import warnings

from ddt import ddt, data

from ...framework.workflow.step import Step


@ddt
class TestStep(unittest.TestCase):

    @data(
        'PSF',
        'Evaluation',
        'GroundTruth',
        'Transform',
        'Deconvolution'
    )
    def test_step(self, module):
        step = Step(module)
        self.assertIsNotNone(step.module)

    @data(
        'wrong_step1',
        'wrong_step2'
    )
    def test_wrong_step(self, method):
        self.assertRaises(ValueError, Step, method)

    def test_list_step_methods(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = Step("PSF")
            self.assertIn('gaussian', s.list_available_methods())

    def test_add_method(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = Step('PSF')
            s.add_method('gaussian')
            self.assertIsNotNone(s.method)

    def test_add_wrong_method(self):
        s = Step('PSF')
        self.assertRaises(ValueError, s.add_method, 'wrong_method')

    def test_list_parameter(self):
        s = Step('PSF', 'gaussian')
        self.assertEqual(len(s.list_parameters()), 3)

    def test_list_parameters_emtpy(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = Step('PSF')
            self.assertIsNone(s.list_parameters())

    def test_specify_parameters_missing_argument(self):
        s = Step('PSF', 'gaussian')
        self.assertRaises(ValueError, s.specify_parameters, aspect=[2, 3])

    def test_specify_parameters_no_module(self):
        s = Step('PSF')
        self.assertRaises(ModuleNotFoundError, s.specify_parameters, sigma=[3])

    def test_specify_parameters(self):
        s = Step('PSF', 'gaussian')
        params = s.specify_parameters(sigma=[1, 2, 3], aspect=[3, 2, 4], mode='align')
        self.assertEqual(len(params), 3)
        params = s.specify_parameters(sigma=[1, 2, 3], aspect=[3, 2, 4], mode='permute')
        self.assertEqual(len(params), 9)

    def test_add_parameters(self):
        s = Step('PSF', 'gaussian')
        s.specify_parameters(sigma=[1, 2, 3], aspect=[3, 2, 4], mode='align')
        s.specify_parameters(sigma=4, overwrite=False)
        self.assertEqual(len(s.parameters), 4)

    def test_saving_parameters(self):
        s = Step('PSF', 'gaussian')
        path = 'test.csv'
        s.specify_parameters(sigma=[1, 2, 3], aspect=[3, 2, 4], mode='align')
        s.save_parameters(path)
        self.assertTrue(os.path.exists(path))
        s1 = Step('PSF', 'gaussian')
        s1.load_parameters(path)
        self.assertEqual(len(s.parameters), len(s1.parameters))
        self.assertSequenceEqual(list(s.parameters.columns), list(s1.parameters.columns))
        for c in s.parameters.columns:
            self.assertSequenceEqual(list(s.parameters[c]), list(s1.parameters[c]))
        os.remove(path)

    def test_to_dict(self):
        s = Step('PSF', 'gaussian')
        path = 'test.csv'
        s.specify_parameters(sigma=[1, 2, 3], aspect=[3, 2, 4], mode='align')
        s.save_parameters(path)
        stepdict = s.to_dict()

        s1 = Step('PSF')
        s1.from_dict(stepdict)
        stepdict2 = s1.to_dict()
        self.assertSequenceEqual(list(stepdict.keys()), list(stepdict2.keys()))
        self.assertSequenceEqual(list(stepdict.values()), list(stepdict2.values()))
        os.remove(path)


if __name__ == '__main__':
    unittest.main()
