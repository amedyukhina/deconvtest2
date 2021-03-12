import unittest
import warnings

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
        w.add_step('PSF', 'gaussian')
        self.assertEqual(len(w.steps), 1)


if __name__ == '__main__':
    unittest.main()
