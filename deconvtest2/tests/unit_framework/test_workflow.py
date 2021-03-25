import os
import unittest
import warnings

import numpy as np
from ddt import ddt, data

from ...framework.workflow.step import Step
from ...framework.workflow.workflow import Workflow


@ddt
class TestStep(unittest.TestCase):

    @data(
        'PSF',
        'Evaluation',
        'GroundTruth',
        'Transform',
        'Deconvolution'
    )
    def test_step(self, method):
        step = Step(method)
        self.assertIsNotNone(step.step)

    @data(
        'wrong_step1',
        'wrong_step2'
    )
    def test_wrong_step(self, method):
        self.assertRaises(ValueError, Step, method)

    def test_list_workflow_steps(self):
        w = Workflow()
        steps = [st.name for st in w.available_steps]
        self.assertIn('PSF', steps)

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

    def test_add_step(self):
        w = Workflow()
        w.add_step(Step('PSF', 'gaussian'))
        self.assertEqual(len(w.steps), 1)

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
        os.remove(path)

    def test_export_import(self):
        s = Step('PSF', 'gaussian')
        path_param1 = 'params1.csv'
        s.specify_parameters(sigma=[1, 2, 3], aspect=[3, 2, 4], mode='align')
        s.save_parameters(path_param1)
        w = Workflow()
        w.add_step(s)

        path = 'test.json'
        w.save(path)

        w2 = Workflow()
        w2.load(path)
        self.assertTrue(os.path.exists(path))
        os.remove(path)
        os.remove(path_param1)

    def test_missing_method(self):
        w = Workflow()
        s = Step('Convolution')
        self.assertRaises(ValueError, w.add_step, s)

    def test_wrong_step_order(self):
        w = Workflow()
        s = Step('Convolution', 'convolve')
        self.assertRaises(IndexError, w.add_step, s)

    def test_wrong_step_number(self):
        w = Workflow(name='test workflow')
        s = Step('GroundTruth', 'ellipsoid')
        w.add_step(s)

        s = Step('PSF', 'gaussian')
        w.add_step(s)

        s = Step('Convolution', 'convolve')
        self.assertRaises(ValueError, w.add_step, s, input_step=[1, 0, 0])

    def test_wrong_step_values(self):
        w = Workflow(name='test workflow')
        s = Step('GroundTruth', 'ellipsoid')
        w.add_step(s)

        s = Step('PSF', 'gaussian')
        w.add_step(s)

        s = Step('Convolution', 'convolve')
        self.assertRaises(IndexError, w.add_step, s, input_step=[1, 2])

    def test_workflow(self):
        w = Workflow(name='test workflow')

        s = Step('GroundTruth', 'ellipsoid')
        path_gt = 'params_ellipsoid.csv'
        s.specify_parameters(size=[10, 15], voxel_size=0.5,
                             theta=[0, np.pi / 2], phi=[0, np.pi, np.pi * 4 / 3], mode='permute', base_name='GT')
        s.save_parameters(path_gt)
        w.add_step(s)

        s = Step('PSF', 'gaussian')
        path_psf = 'params_psf.csv'
        s.specify_parameters(sigma=[1, 2, 3], aspect=[3, 2, 4], mode='align')
        s.save_parameters(path_psf)
        w.add_step(s)

        s = Step('Convolution', 'convolve')
        s.specify_parameters(img='pipeline', psf='pipeline')
        w.add_step(s, input_step=[0, 1])

        s = Step('Transform', 'poisson_noise')
        s.specify_parameters(img='pipeline', snr=[2, 5, 10], base_name='noise')
        path_noise = 'params_noise.csv'
        s.save_parameters(path_noise)
        w.add_step(s)

        s = Step('Evaluation', 'rmse')
        s.specify_parameters(img1='pipeline', img2='pipeline')
        w.add_step(s, input_step=[0, 3])

        path = 'workflow.json'
        w.save(path)
        print(w.steps)
        print(type(w.steps[0]))

        w2 = Workflow()
        w2.load(path)
        # print(w2)
        print(w2.steps)
        print(type(w2.steps[0]))


if __name__ == '__main__':
    unittest.main()
