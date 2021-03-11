import unittest

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
    def test_method(self, method):
        step = Step(method)
        self.assertIsNotNone(step.step)

    @data(
        'wrong_step1',
        'wrong_step2'
    )
    def test_method(self, method):
        self.assertRaises(ValueError, Step, method)


if __name__ == '__main__':
    unittest.main()
